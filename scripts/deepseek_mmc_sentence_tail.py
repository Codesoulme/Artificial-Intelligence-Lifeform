"""Compare pronoun-pair features with or without the rest of the current sentence."""
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

from scripts.audit_mmc_pronoun_pairs import DEV, DEV_SHA, TRAIN, TRAIN_SHA, load
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_mmc_pair_probe import score, select_pairs
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_pooled_address import extract
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load as load_individual
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-mmc-sentence-tail-20261110'


def paired_rows(documents, per_class):
    truncated = select_pairs(documents, per_class)
    extended = select_pairs(documents, per_class, current_sentence_tail=True)
    for a, b in zip(truncated, extended):
        if (a['sample_sha256'] != b['sample_sha256'] or a['label'] != b['label']
                or not b['text'].startswith(a['text'])):
            raise ValueError('Sentence-tail samples differ beyond the appended text')
    if len(truncated) != len(extended):
        raise ValueError('Sentence-tail sample count mismatch')
    return truncated, extended


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sources = {'weight': MODEL / 'model.safetensors', 'individual': INDIVIDUAL, 'train': TRAIN, 'dev': DEV}
    before = {name: sha(path) for name, path in sources.items()}
    if before != {'weight': WEIGHT_SHA, 'individual': INDIVIDUAL_SHA, 'train': TRAIN_SHA, 'dev': DEV_SHA}:
        raise ValueError('Protected source mismatch')
    train, train_tail = paired_rows(load(TRAIN, TRAIN_SHA), 64)
    dev, dev_tail = paired_rows(load(DEV, DEV_SHA), 30)
    if {r['scene'] for r in train} & {r['scene'] for r in dev}:
        raise ValueError('Train/development scene overlap')
    config_sha = feature_files_sha256(MODEL)
    torch.manual_seed(4701)
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
    for split, before_rows, after_rows in (('train', train, train_tail), ('dev', dev, dev_tail)):
        vectors[split] = {}
        for mode, rows in (('truncated', before_rows), ('sentence_tail', after_rows)):
            features, masks = extract(base, tokenizer, [r['text'] for r in rows], batch_size=2)
            vectors[split][mode] = pool(features, masks)
            print(f'{split} {mode} features ready: {len(rows)}', flush=True)
    labels = torch.tensor([r['label'] for r in train])
    results, models = {'role_heuristic': score(dev, [r['heuristic'] for r in dev])}, {}
    for mode in ('truncated', 'sentence_tail'):
        torch.manual_seed(4702)
        classifier = nn.Linear(vectors['train'][mode].shape[-1], 2)
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.0002, weight_decay=0.01)
        for _ in range(600):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(classifier(vectors['train'][mode]), labels)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            train_predictions = classifier(vectors['train'][mode]).argmax(-1).tolist()
            predictions = classifier(vectors['dev'][mode]).argmax(-1).tolist()
        models[mode] = classifier.state_dict()
        results[mode] = {'train': score(train, train_predictions), 'dev': score(dev, predictions),
                         'predictions': [{'sample_sha256': row['sample_sha256'], 'label': row['label'],
                                          'predicted': prediction} for row, prediction in zip(dev, predictions)]}
    if (any(sha(path) != before[name] for name, path in sources.items()) or feature_files_sha256(MODEL) != config_sha
            or frozen_digest(organism, original) != digest or any(p.grad is not None for p in base.parameters())):
        raise ValueError('Protected source changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT / 'heads.pt'
    torch.save({'format': 'deepseek-mmc-sentence-tail-research-v1', 'states': models, 'weight_sha256': WEIGHT_SHA,
                'feature_config_sha256': config_sha, 'hidden_indices': [6, 18]}, checkpoint)
    report = {'status': 'sentence_tail_development_only', 'results': results, 'heads_sha256': sha(checkpoint),
              'tail_added': {'train': sum(a['text'] != b['text'] for a, b in zip(train, train_tail)),
                             'dev': sum(a['text'] != b['text'] for a, b in zip(dev, dev_tail))},
              'component_sha256': before, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Current sentence boundary and speaker are provided by the dataset; later sentences remain excluded.',
                              'Dataset sentence boundaries may not represent complete real-world turns.',
                              'Same previously inspected development set, small balanced sample and one seed.',
                              'Separate heads are optimized for the two input variants.',
                              'Different clusters are treated as different; projected labels can be incomplete or wrong.',
                              'No automatic mentions, source authentication, memory writes, live chat or generalization evaluated.',
                              'Data rights, paraphrase overlap and model pretraining overlap remain unknown.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'role_heuristic': results['role_heuristic'],
                      **{m: results[m]['dev'] for m in ('truncated', 'sentence_tail')},
                      'tail_added': report['tail_added']}), flush=True)


if __name__ == '__main__':
    run()
