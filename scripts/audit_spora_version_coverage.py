"""Test whether tracking partial values reduces omissions in versioned replies."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_fast_weight_lexical_write import base, compress_dimension, lexical_write, SOURCE, tokens
from scripts.audit_spora_versioned_memory_generation import classify, NEUTRAL, read_versions, reply, TRAIN_NAMES, TRAIN_VALUES
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_ttt_joint_writer_reader_pilot import fact_cache
from scripts.deepseek_spora_ttt_read_adapter_pilot import generate
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_ttt_ntp import LowRankFastWeight, install
from spora.research_versioned_memory_pointer import VersionCoverageBias, VersionMemoryPointer
from transformers import LogitsProcessorList


EVENTS = {
    'drinks': (('梁汐宁', '茉莉花茶', '茉莉奶绿'), ('姚晓华', '蜂蜜柚子茶', '咸柠七'),
               ('钟砚平', '黑糖姜汁', '生椰拿铁'), ('袁秋明', '桂花酸梅汤', '桂花米酿')),
    'places': (('孙可安', '湖滨阅览室', '南苑自习室'), ('宋闻清', '枫林展厅', '青石广场')),
    'items': (('余非凡', '木纹收纳盒', '藤编文件篮'), ('程秋', '蓝纹笔记本', '方形钥匙包')),
}
FACTS = {'drinks': '登记显示{name}爱喝{value}。',
         'places': '给{name}约的是{value}。', 'items': '{name}挑中的是{value}。'}
QUESTIONS = {
    'drinks': ('{name}那份饮品偏好里写的是哪一种？只说名字。',
               '说起{name}平常爱喝的，记录里是哪一款？用一句话回复。'),
    'places': ('给{name}登记的见面地点叫什么？只说名字。',
               '回想一下，你有记录和{name}约在哪里吗？请用一句话说清楚。'),
    'items': ('{name}最后挑的用品叫什么？只说品名。',
              '你还记得{name}挑的是哪件用品吗？请用一句完整的话。'),
}
REGRESSION = {'林想', '池樱'}
BOOST = 8.


@torch.inference_mode()
def write_fresh(model, tokenizer, columns, *, layer, eta, ridge, events=EVENTS, facts=FACTS, minimum_tokens=2):
    sites = install(model, (layer,))
    projection = compress_dimension(sites[layer].base.in_features).to(columns.device)
    banks = {}
    try:
        for domain, rows in events.items():
            for name, *values in rows:
                versions = []
                for value in values:
                    event = facts[domain].format(name=name, value=value)
                    captured = fact_cache(model, tokenizer, sites, event, value, decoder_only=True)
                    keys = captured['keys'][layer].index_select(-1, projection)
                    codes = F.normalize(captured['value_embeddings'].index_select(-1, columns), dim=-1)
                    if len(keys) != len(tokens(tokenizer, value)) or not minimum_tokens <= len(keys) <= 4:
                        raise ValueError(f'Unexpected span width for {event}')
                    state = lexical_write(LowRankFastWeight(keys[:0], codes[:0]), keys, codes, eta=eta, ridge=ridge)
                    versions.append((state.keys.cpu(), state.values.cpu()))
                banks[name] = versions
    finally:
        model.model.layers[layer].mlp.down_proj = sites[layer].base
    return banks


def covered_reply(model, tokenizer, head, question, value):
    coverage = VersionCoverageBias(value.tolist(), BOOST)
    output = reply(model, tokenizer, head, question, value, logits_processor=LogitsProcessorList([coverage]))
    return {**output, 'biased_steps': coverage.applied}


def condition(model, tokenizer, head, question, versions, values):
    outputs = []
    for index, state in enumerate(versions):
        raw = reply(model, tokenizer, head, question, state)
        covered = covered_reply(model, tokenizer, head, question, state)
        expected, foil = values[index], values[1 - index]
        eos = tokenizer.eos_token_id
        outputs.append({'value': expected, 'foil': foil, 'base_pointer': raw, 'coverage': covered,
                        'scores': {mode: classify(answer, expected, foil, eos) for mode, answer in
                                   (('base_pointer', raw), ('coverage', covered))}})
    return outputs


def run_cases(model, tokenizer, head, memories, prior):
    cases = []
    old_rows = [row for row in prior['rows'] if row['name'] in REGRESSION]
    for row in old_rows:
        name, question, expected = row['name'], row['question'], row['expected']
        answers = condition(model, tokenizer, head, question, memories[name], expected)
        if [answer['base_pointer'] for answer in answers] != [row['outputs'][p] for p in ('historical', 'current')]:
            raise ValueError('Unchanged pointer did not reproduce the earlier regression pair')
        cases.append({'domain': 'regression', 'name': name, 'question': question, 'answers': answers})
    for domain, events in EVENTS.items():
        for name, old, current in events:
            values = memories[name]
            if [tokenizer.decode(ids.tolist()) for ids in values] != [old, current]:
                raise ValueError(f'Fresh version bank decoded a different value for {name}')
            for template in QUESTIONS[domain]:
                question = template.format(name=name)
                answers = condition(model, tokenizer, head, question, values, (old, current))
                cases.append({'domain': domain, 'name': name, 'question': question, 'answers': answers})
        print('evaluated', domain, len(cases), flush=True)
    empty = torch.empty(0, device='cuda', dtype=torch.long)
    neutral = []
    for question in NEUTRAL:
        absent = covered_reply(model, tokenizer, head, question, empty)
        if absent.pop('biased_steps') or absent != generate(model, tokenizer, question):
            raise ValueError('Empty state altered ordinary decoding')
        neutral.append({'question': question, 'output': absent})
    return cases, neutral


def summarize(cases):
    summary = {}
    for row in cases:
        counts = summary.setdefault(row['domain'], Counter())
        counts['pairs'] += 1
        for mode in ('base_pointer', 'coverage'):
            passed = [a['scores'][mode]['target_only'] and a['scores'][mode]['stopped'] for a in row['answers']]
            counts[f'{mode}_pairs'] += all(passed)
            counts[f'{mode}_answers'] += sum(passed)
        counts['biased_steps'] += sum(a['coverage']['biased_steps'] for a in row['answers'])
    return {domain: dict(counts) for domain, counts in summary.items()}


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    prior_result, pointer, earlier = (args.prior / filename for filename in ('result.json', 'pointer.pt', 'versions.pt'))
    adapter = SOURCE / 'adapters.pt'
    checkpoint = base.MODEL / 'model.safetensors'
    sources = (checkpoint, prior_result, pointer, earlier, adapter,
               base.MODEL / 'tokenizer.json', base.MODEL / 'tokenizer_config.json')
    hashes = {str(path): base.sha(path) for path in sources}
    prior = json.loads(prior_result.read_text(encoding='utf-8'))
    if (hashes[str(checkpoint)] != base.WEIGHT_SHA or hashes[str(earlier)] != prior['state_sha256']
            or hashes[str(pointer)] != prior['pointer_sha256'] or any(hashes[str(path)] != prior['sources'][str(path)]
                                                                   for path in (adapter, *sources[-2:]))):
        raise ValueError('An earlier version input does not match its saved digest')
    fresh_path, result_path = args.output / 'versions.pt', args.output / 'result.json'
    expected = json.loads(result_path.read_text(encoding='utf-8')) if args.verify else None
    if expected is not None and (expected['sources'] != hashes or expected['version_sha256'] != base.sha(fresh_path)):
        raise ValueError('A source has changed since the first coverage run')
    seen_names = set(TRAIN_NAMES) | {row['name'] for row in prior['rows']}
    seen_values = set(TRAIN_VALUES) | {value for row in prior['rows'] for value in row['expected']}
    if any(name in seen_names or old in seen_values or current in seen_values
           for events in EVENTS.values() for name, old, current in events):
        raise ValueError('Fresh cases overlap earlier evaluation or training values')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed()
    print('Loaded unchanged DeepSeek and earlier pointer artifacts', flush=True)
    embedding = model.get_input_embeddings().weight.detach()
    columns = compress_dimension(embedding.shape[1]).to(embedding.device)
    codebook = F.normalize(embedding.index_select(1, columns).float(), dim=-1)
    if not args.verify:
        settings = torch.load(adapter, map_location='cpu', weights_only=True)
        atomic_save(write_fresh(model, tokenizer, columns, layer=settings['layers'][0],
                                eta=settings['eta'], ridge=settings['ridge']), fresh_path)
    previous = torch.load(earlier, map_location='cpu', weights_only=True)
    fixed = {name: previous[name] for name in REGRESSION}
    fresh = torch.load(fresh_path, map_location='cpu', weights_only=True)
    values = read_versions({**fixed, **fresh}, codebook)
    del codebook
    state = torch.load(pointer, map_location='cpu', weights_only=True)
    head = VersionMemoryPointer(model.config.hidden_size, state['scaffold_tokens'])
    head.load_state_dict(state)
    head.eval().to('cuda')
    cases, neutral = run_cases(model, tokenizer, head, values, prior)
    if any(hashes[str(path)] != base.sha(path) for path in sources):
        raise ValueError('A protected source changed')
    result = {'sources': hashes, 'version_sha256': base.sha(fresh_path), 'strength': BOOST,
              'summary': summarize(cases), 'cases': cases, 'neutral': neutral}
    if expected is not None and any(expected[key] != value for key, value in result.items()):
        raise ValueError('Coverage replay changed a recorded output')
    print(result['summary'], flush=True)
    store_json(args.output / 'reload.json' if args.verify else result_path,
               {**result, 'seconds': perf_counter() - started, 'peak_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
