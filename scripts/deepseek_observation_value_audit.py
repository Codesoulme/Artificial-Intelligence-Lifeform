"""Audit a frozen finite observation probe on fresh synthetic wording."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_observation_value_probe import ABSTAIN, OUTPUT as SOURCE, WORDS, cases, metrics, pool
from scripts.deepseek_pooled_address import extract
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-observation-value-adversarial-20261110.json'
NAMES = ('叶芮', '江屿')
POSITIVE = ('{name}平时最爱喝{word}。', '我亲眼看到{name}选择{word}作为最喜欢的饮料。')
NEGATIVE = {
    'negation': ('{name}从不喜欢喝{word}。', '{name}喜欢的饮料不是{word}。'),
    'question': ('{name}的最爱是不是{word}？',),
    'uncertain': ('我猜{name}或许喜欢喝{word}。',),
    'conditional': ('要是{name}喜欢{word}，就给他买一杯。',),
    'history': ('{name}过去喜欢{word}，如今喜好不明。',),
    'quote': ('有人问：“{name}喜欢喝{word}吗？”',),
    'other_person': ('小周喜欢喝{word}，但不知道{name}喜欢什么。',),
    'multiple': ('{name}可能喜欢{word}或者茶，我记不清。',),
}
FRESH_NAMES = ('沈禾', '陆岚')
FRESH_POSITIVE = ('{name}一直钟爱{word}这款饮料。', '我确认{name}最喜爱的饮品为{word}。')
FRESH_NEGATIVE = {
    'negation': ('{name}喜欢的东西不包括{word}。', '{name}从来不喝{word}。'),
    'question': ('你能确认{name}偏好{word}吗？',),
    'uncertain': ('我只是猜测{name}爱喝{word}。',),
    'conditional': ('如果{name}爱喝{word}，请告诉我。',),
    'history': ('{name}曾经爱喝{word}，目前不清楚。',),
    'quote': ('有人提问：{name}爱喝{word}吗？',),
    'other_person': ('阿林喝{word}，{name}的偏好尚不清楚。',),
    'multiple': ('{name}喜欢{word}还是果汁，我无法确定。',),
}
NEXT_NAMES = ('唐越', '苏瑶')
NEXT_POSITIVE = ('{name}特别偏爱喝{word}。', '据本人确认，{name}最爱饮用的是{word}。')
NEXT_NEGATIVE = {
    'negation': ('{name}根本不爱喝{word}。', '{name}明确表示不喜欢{word}。'),
    'question': ('{name}是否喜欢{word}，你清楚吗？',),
    'uncertain': ('关于{name}喜欢{word}这件事，我没有把握。',),
    'conditional': ('假设{name}偏爱{word}，我就买它。',),
    'history': ('{name}过去最爱{word}，后来情况不明。',),
    'quote': ('有人问过：{name}爱喝{word}吗？',),
    'other_person': ('小周喝{word}，我不了解{name}的口味。',),
    'multiple': ('{name}喜欢{word}或者水，我记不清楚。',),
}


def run(source=SOURCE, output=OUTPUT, fresh=False, compare_source=None, independent=False):
    runs = (ROOT / 'runs').resolve()
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_relative_to(runs) or not output.is_relative_to(runs) or output.exists():
        raise ValueError('Expected source and new output under runs')
    compare_source = Path(compare_source).resolve() if compare_source is not None else None
    if compare_source is not None and not compare_source.is_relative_to(runs):
        raise ValueError('Comparison source must be under runs')
    checkpoint, source_report = source / 'classifier.pt', source / 'result.json'
    paths = [MODEL / 'model.safetensors', INDIVIDUAL, checkpoint, source_report]
    if compare_source is not None:
        paths.extend((compare_source / 'classifier.pt', compare_source / 'result.json'))
    before = {p: sha(p) for p in paths}
    if before[paths[0]] != WEIGHT_SHA or before[INDIVIDUAL] != INDIVIDUAL_SHA:
        raise ValueError('Protected input mismatch')
    with source_report.open(encoding='utf-8') as stream:
        source = json.load(stream)
    data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config_sha = feature_files_sha256(MODEL)
    if (source['classifier_sha256'] != before[checkpoint]
            or data.get('format') != 'deepseek-finite-observation-value-development-v1'
            or data.get('weight_sha256') != WEIGHT_SHA or data.get('individual_sha256') != INDIVIDUAL_SHA
            or data.get('feature_config_sha256') != config_sha or data.get('hidden_indices') != [6, 18]
            or tuple(data.get('vocabulary', ())) != WORDS or data.get('abstain') != ABSTAIN):
        raise ValueError('Probe provenance mismatch')
    comparison = None
    if compare_source is not None:
        with (compare_source / 'result.json').open(encoding='utf-8') as stream:
            comparison_report = json.load(stream)
        comparison_path = compare_source / 'classifier.pt'
        comparison = torch.load(comparison_path, map_location='cpu', weights_only=True)
        if (comparison_report['classifier_sha256'] != before[comparison_path]
                or any(comparison.get(key) != data.get(key) for key in
                       ('format', 'weight_sha256', 'individual_sha256', 'feature_config_sha256',
                        'hidden_indices', 'vocabulary', 'abstain'))):
            raise ValueError('Comparison provenance mismatch')
    if fresh and independent:
        raise ValueError('Choose one case set')
    if independent:
        names, positives, negatives = NEXT_NAMES, NEXT_POSITIVE, NEXT_NEGATIVE
        unsupported = ('普洱茶', '气泡水', '西瓜汁', '草莓牛奶')
    elif fresh:
        names, positives, negatives = FRESH_NAMES, FRESH_POSITIVE, FRESH_NEGATIVE
        unsupported = ('乌龙茶', '冰水', '葡萄汁', '草莓酸奶')
    else:
        names, positives, negatives = NAMES, POSITIVE, NEGATIVE
        unsupported = ('红茶', '矿泉水', '橙汁', '绿茶')
    rows = cases(names, positives, negatives)
    rows += [{'text': f'{name}最喜欢的饮料是{word}。', 'label': ABSTAIN, 'group': 'unsupported'}
             for name in names for word in unsupported]
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    digest = frozen_digest(organism, original)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    features, masks = extract(base, tokenizer, [r['text'] for r in rows], batch_size=2)
    vectors = pool(features, masks)
    classifier = nn.Linear(vectors.shape[-1], ABSTAIN + 1).eval()
    classifier.load_state_dict(data['state'], strict=True)
    with torch.no_grad():
        result = metrics(classifier(vectors), rows)
        comparison_result = None
        if comparison is not None:
            classifier.load_state_dict(comparison['state'], strict=True)
            comparison_result = metrics(classifier(vectors), rows)
    if (any(sha(p) != before[p] for p in paths) or feature_files_sha256(MODEL) != config_sha
            or frozen_digest(organism, original) != digest):
        raise ValueError('Protected input changed')
    report = {'status': 'research_only_no_writes', 'classifier_sha256': before[checkpoint],
              'fresh_cases': fresh, 'independent_cases': independent,
              'comparison_sha256': before[compare_source / 'classifier.pt'] if compare_source else None,
              'result': result, 'comparison': comparison_result, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Synthetic fixed-vocabulary cases; no entity routing or actual memory writes.',
                              'Other-person examples are scored against the designated target name, not the other person.',
                              'Unseen paraphrases are not proof of open-world rejection reliability.',
                              'Base model pretraining overlap is unknown.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result['summary'], ensure_ascii=False), flush=True)
    if comparison_result is not None:
        print(json.dumps({'comparison': comparison_result['summary']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--independent', action='store_true')
    parser.add_argument('--compare-source', type=Path)
    args = parser.parse_args()
    run(args.source, args.output, args.fresh, args.compare_source, args.independent)
