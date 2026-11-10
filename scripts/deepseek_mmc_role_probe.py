"""Compare role-only and role-plus-language probes with more dialogue scenes."""
import json
import math
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

from scripts.audit_mmc_pronoun_pairs import DEV, DEV_SHA, TRAIN, TRAIN_SHA, load
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_mmc_pair_probe import score, select_pairs
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_pooled_address import extract
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load as load_individual
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-mmc-role-probe-20261110'


def role_features(rows):
    indices = []
    for row in rows:
        if type(row['heuristic']) is not int or row['heuristic'] not in (0, 1):
            raise ValueError('Invalid role heuristic')
        if type(row['speaker_count']) is not int or row['speaker_count'] < 1:
            raise ValueError('Invalid speaker count')
        indices.append(2 * int(row['speaker_count'] == 2) + row['heuristic'])
    return F.one_hot(torch.tensor(indices, dtype=torch.long), num_classes=4).float()


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sources = {'weight': MODEL / 'model.safetensors', 'individual': INDIVIDUAL, 'train': TRAIN, 'dev': DEV}
    before = {name: sha(path) for name, path in sources.items()}
    if before != {'weight': WEIGHT_SHA, 'individual': INDIVIDUAL_SHA, 'train': TRAIN_SHA, 'dev': DEV_SHA}:
        raise ValueError('Protected source mismatch')
    train, dev = select_pairs(load(TRAIN, TRAIN_SHA), 200), select_pairs(load(DEV, DEV_SHA), 30)
    if {r['scene'] for r in train} & {r['scene'] for r in dev}:
        raise ValueError('Train/development scene overlap')
    config_sha = feature_files_sha256(MODEL)
    torch.manual_seed(4711)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load_individual(INDIVIDUAL, 'cpu')
    digest = frozen_digest(organism, original)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    vectors = {}
    for name, rows in (('train', train), ('dev', dev)):
        features, masks = extract(base, tokenizer, [r['text'] for r in rows], batch_size=2)
        semantic = pool(features, masks)
        semantic = semantic / math.sqrt(semantic.shape[-1])
        role = role_features(rows)
        vectors[name] = {'role': role, 'semantic': semantic, 'combined': torch.cat((role, semantic), dim=-1)}
        print(f'{name} features ready: {len(rows)}', flush=True)
    labels = torch.tensor([r['label'] for r in train])
    results, states = {'heuristic': score(dev, [r['heuristic'] for r in dev])}, {}
    for mode in ('role', 'semantic', 'combined'):
        torch.manual_seed(4712)
        classifier = nn.Linear(vectors['train'][mode].shape[-1], 2)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
        for _ in range(1000):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(classifier(vectors['train'][mode]), labels) + 0.01 * classifier.weight.square().sum()
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            train_predictions = classifier(vectors['train'][mode]).argmax(-1).tolist()
            predictions = classifier(vectors['dev'][mode]).argmax(-1).tolist()
        states[mode] = classifier.state_dict()
        results[mode] = {'train': score(train, train_predictions), 'dev': score(dev, predictions),
                         'predictions': [{'sample_sha256': row['sample_sha256'], 'label': row['label'],
                                          'predicted': prediction} for row, prediction in zip(dev, predictions)]}
    if (any(sha(path) != before[name] for name, path in sources.items()) or feature_files_sha256(MODEL) != config_sha
            or frozen_digest(organism, original) != digest or any(p.grad is not None for p in base.parameters())):
        raise ValueError('Protected source changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT / 'heads.pt'
    torch.save({'format': 'deepseek-mmc-role-research-v1', 'states': states,
                'weight_sha256': WEIGHT_SHA, 'feature_config_sha256': config_sha, 'hidden_indices': [6, 18]}, checkpoint)
    report = {'status': 'role_pair_development_only', 'results': results, 'heads_sha256': sha(checkpoint),
              'training_sample_sha256': [r['sample_sha256'] for r in train], 'component_sha256': before,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Mention and speaker identity are supplied by dataset, not predicted or authenticated.',
                              'Semantic, role, and combined models have different feature widths and independent optimization.',
                              'Dev set was repeatedly inspected; small balanced sample and single training seed.',
                              'Different annotated clusters are treated as different; labels may be incomplete or noisy.',
                              'No truth, write permission, live memory, new domain or chat quality evaluated.',
                              'Data rights, paraphrase overlap and base pretraining overlap remain unknown.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'heuristic': results['heuristic'], **{m: results[m]['dev'] for m in states}}), flush=True)


if __name__ == '__main__':
    run()
