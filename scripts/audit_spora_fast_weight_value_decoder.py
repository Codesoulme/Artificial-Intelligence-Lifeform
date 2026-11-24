"""Test direct value recovery from low-rank fast weights on unseen names and tokens."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_isolated_fast_weights import NAMES, SOURCE, VALUES, base, empty, restore, write
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_ttt_joint_writer_reader_pilot import fact_cache
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_ttt_ntp import install


TRAIN_NAMES = ('应泠', '梅杳', '申云卿', '柳迟', '章亦成', '杜书桃', '黎英', '夏行舟')
FRESH_NAMES = ('林想', '兰雨', '池樱', '项知遥')
TRAIN_VALUES = ('牛奶', '绿茶', '红茶', '奶茶', '啤酒', '矿泉水', '酒', '冷水', '白酒', '冰块',
                '雪碧', '可乐', '橙汁', '冰水', '白开水', '椰汁', '汽水', '米酒', '乌龙茶',
                '苹果汁', '蜂蜜水', '奶昔', '酸梅汤', '西瓜汁', '热巧克力', '热水')
FRESH_VALUES = ('柠檬茶', '黑咖啡', '葡萄汁', '豆奶', '苏打水', '冰红茶', '姜茶', '白茶')
FACT = '{name}喜欢喝{value}。'
ALTERNATE = '在饮品这一项，{name}选的是{value}。'


def target_vectors(memory, length, eta, ridge):
    """Invert the sequential update using the prefix factors and the current write key."""
    return torch.cat([state.recover_targets(length, eta=eta, ridge=ridge).cpu()
                      for state in memory.values()], dim=-1)


def tokens(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def make_event(model, tokenizer, wrappers, state, adapter, name, value, surface):
    cache = fact_cache(model, tokenizer, wrappers, surface.format(name=name, value=value), value, decoder_only=True)
    latest = write(state, cache, adapter)
    # Record the number of write positions as event metadata, without storing labels in the bank.
    n = len(cache['positions'])
    recovered = target_vectors(latest, n, adapter['eta'], adapter['ridge'])
    actual = torch.cat([cache['targets'][layer].cpu() for layer in wrappers], dim=-1)
    torch.testing.assert_close(recovered, actual, rtol=1e-5, atol=1e-5)
    return latest, n, recovered, tokens(tokenizer, value)


def load_capture_model(layers):
    # Leave a subsequent layer so the last captured target precedes final normalization.
    depth = max(layers) + 2
    config = json.loads((base.MODEL / 'config.json').read_text(encoding='utf-8'))
    model = load_full_streamed(skip_replaced=tuple(range(depth, config['num_hidden_layers'])),
                               stage_output_projection=True)
    del model.model.layers[depth:]
    model.lm_head = torch.nn.Identity()
    return model.eval().requires_grad_(False)


def fit(rows, emb):
    inputs = F.normalize(torch.cat([x for x, _ in rows]), dim=-1)
    labels = torch.tensor([token for _, ids in rows for token in ids])
    y = F.normalize(emb.index_select(0, labels.to(emb.device)).float().cpu(), dim=-1)
    center, shift = inputs.mean(0), y.mean(0)
    x = inputs - center
    # Solve in sample space to limit memory use for the two concatenated transformer layers.
    dual = torch.linalg.solve(x @ x.T + .025 * torch.eye(len(x)), y - shift)
    return {'center': center, 'shift': shift, 'projection': x.T @ dual}


@torch.no_grad()
def decode(head, vectors, emb):
    if vectors is None:
        return []
    query = F.normalize((F.normalize(vectors, dim=-1) - head['center']) @ head['projection'] + head['shift'],
                        dim=-1).to(emb.device)
    scores, hits = None, None
    for start in range(0, len(emb), 4096):
        normalized = F.normalize(emb[start:start + 4096].float(), dim=-1)
        top, index = (query @ normalized.T).max(-1)
        if scores is None:
            scores, hits = top, index + start
        else:
            hits = torch.where(top > scores, index + start, hits)
            scores = torch.maximum(scores, top)
    return hits.tolist()


def summarize(rows):
    return {'cases': len(rows), 'exact': sum(row['answer_ids'] == row['expected_ids'] for row in rows),
            'token_exact': sum(a == b for row in rows for a, b in zip(row['answer_ids'], row['expected_ids'], strict=True)),
            'tokens': sum(len(row['expected_ids']) for row in rows)}


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    report = args.output / 'result.json'
    weights = args.output / 'decoder.pt'
    paths = (base.MODEL / 'model.safetensors', SOURCE / 'adapters.pt', args.saved / 'result.json',
             args.saved / '32-isolated.pt')
    hashes = {str(path): base.sha(path) for path in paths}
    parent = json.loads(paths[2].read_text(encoding='utf-8'))
    stage = parent['stages'][-1]
    if (hashes[str(paths[3])] != stage['storage']['isolated']['sha256']
            or any(hashes[str(path)] != parent['sources'][str(path)] for path in paths[:2])):
        raise ValueError('The saved memories or the DeepSeek source changed')
    expected = json.loads(report.read_text(encoding='utf-8')) if args.verify else None
    if expected is not None and (expected['sources'] != hashes or expected['decoder_sha256'] != base.sha(weights)):
        raise ValueError('Independent replay source changed')
    adapter = torch.load(paths[1], weights_only=True, map_location='cpu')
    model = load_capture_model(adapter['layers'])
    wrappers = install(model, tuple(adapter['layers']))
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    emb = model.get_input_embeddings().weight.detach()
    memory = restore(torch.load(paths[3], weights_only=True, map_location='cpu'), wrappers)
    training = []
    if args.verify:
        head = torch.load(weights, weights_only=True, map_location='cpu')
    else:
        for i, name in enumerate(TRAIN_NAMES):
            for j, value in enumerate(TRAIN_VALUES):
                _, n, feature, ids = make_event(model, tokenizer, wrappers, empty(wrappers), adapter, name, value,
                                                FACT if (i + j) % 2 else ALTERNATE)
                if n != len(ids):
                    raise ValueError(f'Write-position count differs from stand-alone tokenization: {value}')
                training.append((feature, ids))
        print('training events', len(training), flush=True)
        head = fit(training, emb)
        atomic_save(head, weights)
    # Existing banks used one token per event. The value dictionary is used only for grading.
    by_name = {row['name']: row for row in stage['rows']}
    if set(by_name) != set(NAMES) or any(len(tokens(tokenizer, value)) != 1 for value in VALUES):
        raise ValueError('Unexpected saved memory fixture')
    old_reads, swaps = [], []
    for i, name in enumerate(NAMES):
        other = NAMES[i ^ 1]
        for bank_name, output in ((name, old_reads), (other, swaps)):
            feature = target_vectors(memory[bank_name], 1, adapter['eta'], adapter['ridge'])
            ids = decode(head, feature, emb)
            output.append({'name': name, 'bank': bank_name, 'expected_ids': tokens(tokenizer, by_name[bank_name]['expected']),
                           'answer_ids': ids, 'answer': tokenizer.decode(ids, skip_special_tokens=True)})
    fresh, correction = [], []
    for i, name in enumerate(FRESH_NAMES):
        for j in range(2):
            first, second = FRESH_VALUES[i * 2 + j], FRESH_VALUES[(i * 2 + j + 3) % len(FRESH_VALUES)]
            bank, n, feat, ids = make_event(model, tokenizer, wrappers, empty(wrappers), adapter, name, first, ALTERNATE)
            if n != len(ids):
                raise ValueError('Held event is not token-aligned')
            read_ids = decode(head, feat, emb)
            fresh.append({'name': name, 'expected_ids': ids, 'answer_ids': read_ids,
                          'answer': tokenizer.decode(read_ids, skip_special_tokens=True)})
            _, n, feat, ids = make_event(model, tokenizer, wrappers, bank, adapter, name, second, FACT)
            if n != len(ids):
                raise ValueError('Held correction is not token-aligned')
            read_ids = decode(head, feat, emb)
            correction.append({'name': name, 'expected_ids': ids, 'answer_ids': read_ids,
                               'answer': tokenizer.decode(read_ids, skip_special_tokens=True)})
    training_token_ids = sorted({token for value in TRAIN_VALUES for token in tokens(tokenizer, value)})
    if {token for value in VALUES for token in tokens(tokenizer, value)} & set(training_token_ids):
        raise ValueError('The existing evaluation answers overlap the decoder training tokens')
    result = {'sources': hashes, 'decoder_sha256': base.sha(weights), 'training_token_ids': training_token_ids,
              'saved': summarize(old_reads), 'swapped': summarize(swaps), 'fresh': summarize(fresh),
              'correction': summarize(correction), 'empty_ids': decode(head, None, emb),
              'saved_rows': old_reads, 'swap_rows': swaps, 'fresh_rows': fresh, 'correction_rows': correction}
    if any(base.sha(path) != hashes[str(path)] for path in paths):
        raise ValueError('A frozen input changed')
    print({key: result[key] for key in ('saved', 'swapped', 'fresh', 'correction', 'empty_ids')}, flush=True)
    if expected is not None and any(expected[key] != value for key, value in result.items()):
        raise ValueError('Independent reload changed value decoding')
    store_json(args.output / ('reload.json' if args.verify else 'result.json'),
               {**result, 'seconds': perf_counter() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('saved', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
