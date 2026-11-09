"""Research separate factuality and finite-value heads over frozen features."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_observation_value_probe import (ABSTAIN, EXTRA_NEGATIVE, EXTRA_POSITIVE, HELD_NAMES,
    HELD_NEGATIVE, HELD_POSITIVE, TRAIN_NAMES, TRAIN_NEGATIVE, TRAIN_POSITIVE, WORDS, cases, metrics, pool)
from scripts.deepseek_pooled_address import extract
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-observation-factorized-20261110'
FORMAT = 'deepseek-finite-observation-factorized-development-v1'


class FactorizedProbe(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fact = nn.Linear(width, 1)
        self.value = nn.Linear(width, ABSTAIN + 1)

    def forward(self, features):
        return self.fact(features).squeeze(-1), self.value(features)


def combined_logits(fact, values):
    if fact.ndim != 1 or values.shape != (len(fact), ABSTAIN + 1):
        raise ValueError('Invalid head outputs')
    selected = values.argmax(-1)
    selected = torch.where(fact >= 0, selected, torch.full_like(selected, ABSTAIN))
    return F.one_hot(selected, ABSTAIN + 1).float()


def build_data():
    train = cases(TRAIN_NAMES, TRAIN_POSITIVE, TRAIN_NEGATIVE) + cases(TRAIN_NAMES, EXTRA_POSITIVE, EXTRA_NEGATIVE)
    train += [{'text': template.format(name=name, word=word), 'label': ABSTAIN,
               'mentioned_value': ABSTAIN, 'group': 'unsupported'}
              for name in TRAIN_NAMES for word in ('茉莉花茶', '苏打水', '苹果汁', '冰牛奶', '绿豆汤')
              for template in TRAIN_POSITIVE + EXTRA_POSITIVE]
    held = cases(HELD_NAMES, HELD_POSITIVE, HELD_NEGATIVE)
    held += [{'text': f'{name}喜欢喝{word}。', 'label': ABSTAIN, 'mentioned_value': ABSTAIN, 'group': 'unsupported'}
             for name in HELD_NAMES for word in ('柠檬水', '热巧克力', '汽水', '白开水')]
    return train, held


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    protected = {p: sha(p) for p in (MODEL / 'model.safetensors', INDIVIDUAL)}
    if protected[MODEL / 'model.safetensors'] != WEIGHT_SHA or protected[INDIVIDUAL] != INDIVIDUAL_SHA:
        raise ValueError('Protected input mismatch')
    config_sha = feature_files_sha256(MODEL)
    train, held = build_data()
    if {r['text'] for r in train} & {r['text'] for r in held}:
        raise ValueError('Training and held texts overlap')
    torch.manual_seed(4641)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(organism, original)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    vectors = {}
    for name, rows in (('train', train), ('held', held)):
        features, masks = extract(base, tokenizer, [r['text'] for r in rows], batch_size=2)
        vectors[name] = pool(features, masks)
        print(f'{name} features ready: {len(rows)}', flush=True)
    model = FactorizedProbe(vectors['train'].shape[-1])
    fact_labels = torch.tensor([r['group'] in ('positive', 'unsupported') for r in train], dtype=torch.float32)
    value_labels = torch.tensor([r['mentioned_value'] for r in train])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.01)
    for _ in range(600):
        optimizer.zero_grad(set_to_none=True)
        fact, values = model(vectors['train'])
        loss = F.binary_cross_entropy_with_logits(fact, fact_labels) + F.cross_entropy(values, value_labels)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        results = {}
        for name, rows in (('train', train), ('held', held)):
            fact, values = model(vectors[name])
            result = metrics(combined_logits(fact, values), rows)
            for row, f, value in zip(result['rows'], fact.sigmoid().tolist(), values.argmax(-1).tolist()):
                row['fact_probability'], row['value_prediction'] = f, value
            results[name] = result
    if (any(sha(p) != digest for p, digest in protected.items()) or frozen_digest(organism, original) != before
            or feature_files_sha256(MODEL) != config_sha or any(p.grad is not None for p in base.parameters())):
        raise ValueError('Protected input changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT / 'classifier.pt'
    torch.save({'format': FORMAT, 'state': model.state_dict(), 'vocabulary': WORDS, 'abstain': ABSTAIN,
                'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA,
                'feature_config_sha256': config_sha, 'hidden_indices': [6, 18]}, checkpoint)
    report = {'status': 'research_only_no_writes', 'classifier_sha256': sha(checkpoint),
              'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA, 'feature_config_sha256': config_sha,
              'steps': 600, 'fact_threshold': 0.5, 'results': results,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Fixed eight-word vocabulary and synthetic examples.',
                              'Fact labels treat unsupported affirmative sentences as factual; source trust is not checked.',
                              'No entity recognition, memory writes, correction or persistence.',
                              'Fact and value thresholds are uncalibrated and not safe for actual writes.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({name: result['summary'] for name, result in results.items()}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
