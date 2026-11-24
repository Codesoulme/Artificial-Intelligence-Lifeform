"""Swap independently stored versions under an unchanged DeepSeek question."""

import argparse
from collections import Counter
import json
from itertools import product
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_fast_weight_lexical_write import base, compress_dimension, SOURCE, tokens
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_ttt_read_adapter_pilot import generate
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_ttt_ntp import LowRankFastWeight
from spora.research_versioned_memory_pointer import VersionMemoryPointer


SEED = 26321
MEMORY_FRACTION = .95
TRAIN_NAMES = ('吴璐', '简诺', '楚筠', '温子欣', '王南枫', '金雯')
TRAIN_VALUES = ('椰子水', '温牛奶', '冰气泡茶', '麦茶', '橙汁', '奶茶', '山楂露', '绿豆汤', '矿泉水', '乌龙茶')
TRAIN_FORMS = (
    ('{name}喜欢喝什么？只给饮品名。', '', ''),
    ('回忆一下{name}爱喝哪种饮料，简短回答。', '', '。'),
    ('你记得{name}最喜欢哪种饮料吗？请说一句完整的话。', '我记得是', '。'),
    ('{name}最常点哪款？请用一句话说明。', '记录里喝的是', '。'),
)
HELD_FORMS = (
    ('short', '饮品这一栏，{name}平常选哪样？只说名称。'),
    ('sentence', '说到{name}，你还记得常喝的是哪样吗？答成一句话。'),
)
NEUTRAL = ('怎么给刚泡的茶降温？', '今天很累，帮我想一个放松的办法。',
           '小昭爱喝哪种饮料，你有记录吗？', '一直没来过这里的蒋妍最喜欢哪款？')


def separate_versions(payload, *, eta, ridge):
    records = {}
    for name, (keys, factors) in payload['factors'].items():
        lengths = payload['event_lengths'][name]
        if len(lengths) < 2:
            continue
        source = LowRankFastWeight(keys, factors)
        targets = source.recover_targets(len(keys), eta=eta, ridge=ridge)
        versions = []
        for k, v in zip(keys.split(lengths), targets.split(lengths), strict=True):
            if not 1 <= len(k) <= 4:
                raise ValueError('An event exceeds the independent rank budget')
            empty = LowRankFastWeight(k[:0], v[:0])
            state = empty.write(k, v, eta=eta, ridge=ridge)
            versions.append((state.keys, state.values))
        records[name] = versions
    return records


@torch.inference_mode()
def read_versions(records, codebook):
    values = {}
    for name, versions in records.items():
        values[name] = [(F.normalize(LowRankFastWeight(k, v).read(k).to(codebook.device), dim=-1)
                         @ codebook.T).argmax(-1) for k, v in versions]
    return values


def supervised_rows(tokenizer):
    rows = []
    for index, (name, (form, before, after)) in enumerate(product(TRAIN_NAMES, TRAIN_FORMS)):
        for value in (TRAIN_VALUES[(index + offset) % len(TRAIN_VALUES)] for offset in (0, 3, 6, 9)):
            before_ids, value_ids, after_ids = [tokens(tokenizer, text) for text in (before, value, after)]
            if not 1 <= len(value_ids) <= 4:
                raise ValueError('Training memory is outside the slot budget')
            tail = [*before_ids, *value_ids, *after_ids, tokenizer.eos_token_id]
            if tokenizer.decode(tail[:-1]) != before + value + after:
                raise ValueError('Training token boundaries changed the response text')
            actions = [*(('text', token) for token in before_ids),
                       *(('copy', slot) for slot in range(len(value_ids))),
                       *(('text', token) for token in (*after_ids, tokenizer.eos_token_id))]
            rows.append((form.format(name=name), tail, actions, len(value_ids)))
    return rows


@torch.inference_mode()
def frozen_features(model, tokenizer, rows):
    features, frame_lengths = [], []
    for question, tail, _, _ in rows:
        ids = closed_think_inputs(tokenizer, question, 'cuda')['input_ids']
        answer = torch.tensor([tail[:-1]], device=ids.device)
        combined = torch.cat((ids, answer), dim=1)
        hidden = model.model(input_ids=combined, use_cache=False).last_hidden_state
        frame_lengths.append(len(tail))
        features.append(hidden[0, -len(tail):].float().cpu())
    return torch.cat(features), frame_lengths


def fit(model, tokenizer, head, rows, steps):
    features, sizes = frozen_features(model, tokenizer, rows)
    features = features.clone()
    scaffold = {token: head.slots + i for i, token in enumerate(head.scaffold_tokens.tolist())}
    labels = torch.tensor([value if kind == 'copy' else scaffold[value]
                           for _, _, actions, _ in rows for kind, value in actions])
    lengths = torch.tensor([length for (_, _, _, length), size in zip(rows, sizes, strict=True) for _ in range(size)])
    opt = torch.optim.AdamW(head.parameters(), lr=.002)
    generator = torch.Generator().manual_seed(SEED)
    for step in range(steps):
        batch = torch.randperm(len(features), generator=generator)[:64]
        model_out = head(features[batch], lengths[batch])
        loss = F.cross_entropy(model_out, labels[batch])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in (0, steps // 2, steps - 1):
            print('pointer fit', step + 1, float(loss.detach()), flush=True)


def reply(model, tokenizer, head, question, value, *, logits_processor=None):
    def memory_output(_module, inputs, logits):
        return head.mix(inputs[0], logits, value, memory_fraction=MEMORY_FRACTION)

    with model.lm_head.register_forward_hook(memory_output):
        return generate(model, tokenizer, question, logits_processor=logits_processor)


def classify(answer, wanted, foil, eos):
    text = answer['answer']
    stopped = bool(answer['token_ids'] and answer['token_ids'][-1] == eos)
    return {'target_only': wanted in text and foil not in text, 'stopped': stopped,
            'exact_name': stopped and text.strip('。!！.，, \n') == wanted}


def trial(model, tokenizer, head, values, reference):
    summary = {style: Counter() for style, _ in HELD_FORMS}
    results = []
    empty = torch.empty(0, dtype=torch.long, device='cuda')
    eos = tokenizer.eos_token_id
    for name, versions in values.items():
        if len(versions) != 2:
            raise ValueError('The paired audit requires two versions per name')
        expected = [row['value'] for row in reference[name]]
        decoded = [tokenizer.decode(ids.tolist()) for ids in versions]
        if decoded != expected or decoded[0] == decoded[1]:
            raise ValueError('A separated state did not reproduce its reference value')
        for style, form in HELD_FORMS:
            question = form.format(name=name)
            prompts = closed_think_inputs(tokenizer, question, 'cuda')['input_ids'][0].tolist()
            off = generate(model, tokenizer, question)
            outputs = {key: reply(model, tokenizer, head, question, state) for key, state in
                       (('historical', versions[0]), ('current', versions[1]), ('empty', empty))}
            scores = {key: classify(outputs[key], expected[i], expected[1 - i], eos)
                      for i, key in enumerate(('historical', 'current'))}
            counts = summary[style]
            counts.update({'pairs': 1, 'switches_exclusive': all(x['target_only'] for x in scores.values()),
                           'switches_and_stop': all(x['target_only'] and x['stopped'] for x in scores.values()),
                           'both_exact_name': all(x['exact_name'] for x in scores.values()),
                           'empty_matches_off': outputs['empty'] == off,
                           'off_mentions_known_value': any(value in off['answer'] for value in expected)})
            results.append({'name': name, 'style': style, 'question': question, 'prompt_ids': prompts,
                            'expected': expected, 'outputs': {**outputs, 'off': off}, 'scores': scores})
        print('evaluated paired cases', len(results), flush=True)
    neutral = [{'question': question, 'off': generate(model, tokenizer, question),
                'empty': reply(model, tokenizer, head, question, empty)} for question in NEUTRAL]
    if any(row['empty'] != row['off'] for row in neutral) or any(
        row['outputs']['empty'] != row['outputs']['off'] for row in results
    ):
        raise ValueError('Empty memory changed native generation')
    return {style: dict(counts) for style, counts in summary.items()}, results, neutral


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    output = args.output
    if not args.verify:
        output.mkdir(parents=True, exist_ok=False)
    memory_path, head_path = output / 'versions.pt', output / 'pointer.pt'
    base_path = base.MODEL / 'model.safetensors'
    adapter_path = SOURCE / 'adapters.pt'
    state_path, reference_path = args.saved / 'states.pt', args.saved / 'result.json'
    sources = (base_path, adapter_path, state_path, reference_path, base.MODEL / 'tokenizer.json',
               base.MODEL / 'tokenizer_config.json')
    hashes = {str(path): base.sha(path) for path in sources}
    reference = json.loads(reference_path.read_text(encoding='utf-8'))
    if (hashes[str(base_path)] != base.WEIGHT_SHA or hashes[str(state_path)] != reference['state_sha256']
            or hashes[str(adapter_path)] != reference['inputs'][str(adapter_path)]):
        raise ValueError('Source digests do not match the lexical audit')
    expected = json.loads((output / 'result.json').read_text(encoding='utf-8')) if args.verify else None
    if expected is not None and (expected['sources'] != hashes or expected['state_sha256'] != base.sha(memory_path)
                                 or expected['pointer_sha256'] != base.sha(head_path)):
        raise ValueError('Replay sources differ from the audited sources')
    if not args.verify:
        prior = torch.load(state_path, weights_only=True, map_location='cpu')
        settings = torch.load(adapter_path, weights_only=True, map_location='cpu')
        atomic_save(separate_versions(prior, eta=settings['eta'], ridge=settings['ridge']), memory_path)
    records = torch.load(memory_path, weights_only=True, map_location='cpu')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    if set(records) & set(TRAIN_NAMES) or any(row['value'] in TRAIN_VALUES
                                              for name in records for row in reference['records'][name]):
        raise ValueError('Training and held-out names or values overlap')
    model = load_full_streamed()
    print('Loaded frozen DeepSeek', flush=True)
    embeddings = model.get_input_embeddings().weight.detach()
    columns = compress_dimension(embeddings.shape[1]).to(embeddings.device)
    codebook = F.normalize(embeddings.index_select(1, columns).float(), dim=-1)
    values = read_versions(records, codebook)
    del codebook
    rows = supervised_rows(tokenizer)
    scaffold = sorted({value for _, _, actions, _ in rows for kind, value in actions if kind == 'text'})
    head = VersionMemoryPointer(model.config.hidden_size, scaffold)
    if args.verify:
        head.load_state_dict(torch.load(head_path, weights_only=True, map_location='cpu'))
    else:
        fit(model, tokenizer, head, rows, args.steps)
        atomic_save(head.state_dict(), head_path)
    head.eval().to('cuda')
    summary, cases, neutral = trial(model, tokenizer, head, values, reference['records'])
    if any(base.sha(path) != hashes[str(path)] for path in sources):
        raise ValueError('A protected source changed')
    report = {'sources': hashes, 'state_sha256': base.sha(memory_path), 'pointer_sha256': base.sha(head_path),
              'steps': expected['steps'] if expected else args.steps, 'training_rows': len(rows),
              'summary': summary, 'rows': cases, 'neutral': neutral}
    if expected is not None and any(expected[key] != value for key, value in report.items()):
        raise ValueError('Independent reload changed the observed outputs')
    print(summary, flush=True)
    store_json(output / ('reload.json' if args.verify else 'result.json'),
               {**report, 'seconds': perf_counter() - started, 'peak_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('saved', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--steps', type=int, default=180)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
