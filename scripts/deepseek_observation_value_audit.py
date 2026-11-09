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


def run(source=SOURCE, output=OUTPUT):
    runs = (ROOT / 'runs').resolve()
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_relative_to(runs) or not output.is_relative_to(runs) or output.exists():
        raise ValueError('Expected source and new output under runs')
    checkpoint, source_report = source / 'classifier.pt', source / 'result.json'
    paths = (MODEL / 'model.safetensors', INDIVIDUAL, checkpoint, source_report)
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
    rows = cases(NAMES, POSITIVE, NEGATIVE)
    rows += [{'text': f'{name}最喜欢的饮料是{word}。', 'label': ABSTAIN, 'group': 'unsupported'}
             for name in NAMES for word in ('红茶', '矿泉水', '橙汁', '绿茶')]
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
    if (any(sha(p) != before[p] for p in paths) or feature_files_sha256(MODEL) != config_sha
            or frozen_digest(organism, original) != digest):
        raise ValueError('Protected input changed')
    report = {'status': 'research_only_no_writes', 'classifier_sha256': before[checkpoint],
              'result': result, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Synthetic fixed-vocabulary cases; no entity routing or actual memory writes.',
                              'Other-person examples are scored against the designated target name, not the other person.',
                              'Unseen paraphrases are not proof of open-world rejection reliability.',
                              'Base model pretraining overlap is unknown.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.source, args.output)
