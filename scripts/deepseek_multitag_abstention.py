"""Explore whether weak multi-persona examples improve cautious subject classification."""
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_dulemon_persona_audit import normalize_turn
from scripts.deepseek_dulemon_routing_audit import ARCHIVE, ARCHIVE_SHA
from scripts.deepseek_dulemon_subject_annotation import LABELS, SOURCE, SOURCE_SHA, compare
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_relative_identity_probe import AMBIGUOUS_TRAIN, HELD, NOVEL, TRAIN, UNKNOWN, build_cases
from scripts.deepseek_weak_persona_training import contextualize, sample_training
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-multitag-abstention-20261110'


def sample_multitag(path, excluded_texts, excluded_sessions, count=12):
    if type(count) is not int or count < 1:
        raise ValueError('Positive count required')
    candidates = []
    with zipfile.ZipFile(path) as archive, archive.open('DuLeMon/both/train.json') as stream:
        for session, line in enumerate(stream):
            if session in excluded_sessions:
                continue
            dialogue = json.loads(line).get('conversation')
            if not isinstance(dialogue, list):
                raise ValueError('Invalid conversation')
            for turn in range(1, len(dialogue)):
                previous, current = normalize_turn(dialogue[turn - 1]), normalize_turn(dialogue[turn])
                if current is None or previous is None or previous[0] == current[0]:
                    continue
                role, text, tags = current
                parts = tags.split(',') if tags else []
                if (not parts or any(len(t) < 2 or t[0] not in ('U', 'B') or not t[1:].isdigit() for t in parts)
                        or {t[0] for t in parts} != {'U', 'B'} or text in excluded_texts
                        or previous[1] in excluded_texts or not 4 <= len(text) <= 80
                        or not 1 <= len(previous[1]) <= 100):
                    continue
                digest = hashlib.sha256((role + '\0' + text + '\0' + previous[0] + '\0' + previous[1]).encode()).hexdigest()
                candidates.append({'session': session, 'role': role, 'text': text, 'previous_role': previous[0],
                                   'previous_text': previous[1], 'label': UNKNOWN, 'sample_sha256': digest})
    selected, sessions, texts = [], set(), set()
    for row in sorted(candidates, key=lambda r: r['sample_sha256']):
        if row['session'] in sessions or row['text'] in texts:
            continue
        selected.append(row)
        sessions.add(row['session'])
        texts.add(row['text'])
        if len(selected) == count:
            return selected
    raise ValueError(f'Insufficient multi-persona sessions: {len(selected)}')


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = {'weight': MODEL / 'model.safetensors', 'individual': INDIVIDUAL, 'audit': SOURCE, 'archive': ARCHIVE}
    before = {name: sha(path) for name, path in files.items()}
    if (before['weight'] != WEIGHT_SHA or before['individual'] != INDIVIDUAL_SHA
            or before['audit'] != SOURCE_SHA or before['archive'] != ARCHIVE_SHA):
        raise ValueError('Protected source mismatch')
    with SOURCE.open(encoding='utf-8') as stream:
        all_audit = json.load(stream)['results']['single']['baseline']['rows']
    audit = all_audit[:len(LABELS)]
    excluded = {row[key] for row in all_audit for key in ('text', 'previous_text')}
    natural = sample_training(ARCHIVE, excluded)
    excluded.update(row[key] for row in natural for key in ('text', 'previous_text'))
    multi = sample_multitag(ARCHIVE, excluded, {row['session'] for row in natural})
    splits = {'synthetic': build_cases(TRAIN) + build_cases(AMBIGUOUS_TRAIN), 'natural': natural, 'multi': multi,
              'held': build_cases(HELD), 'novel': build_cases(NOVEL), 'audit': audit}
    config_sha = feature_files_sha256(MODEL)
    torch.manual_seed(4680)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    digest = frozen_digest(organism, original)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    vectors = {}
    for split, rows in splits.items():
        features, masks = extract(base, tokenizer, contextualize(rows, split in ('natural', 'multi', 'audit')), batch_size=2)
        vectors[split] = pool(features, masks)
        print(f'{split} features ready: {len(rows)}', flush=True)
    labels = {split: torch.tensor([r['label'] for r in splits[split]]) for split in ('synthetic', 'natural', 'multi')}
    results, models = {}, {}
    for weight in (0.0, 0.25, 1.0):
        torch.manual_seed(4681)
        classifier = nn.Linear(vectors['synthetic'].shape[-1], 3)
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.0002, weight_decay=0.01)
        for _ in range(600):
            optimizer.zero_grad(set_to_none=True)
            loss = (F.cross_entropy(classifier(vectors['synthetic']), labels['synthetic'])
                    + F.cross_entropy(classifier(vectors['natural']), labels['natural'])
                    + weight * F.cross_entropy(classifier(vectors['multi']), labels['multi'])) / (2 + weight)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            loss.backward()
            optimizer.step()
        key = str(weight)
        models[key] = classifier.state_dict()
        results[key] = {}
        with torch.no_grad():
            for split in ('held', 'novel', 'audit'):
                predicted = classifier(vectors[split]).argmax(-1).tolist()
                expected = [item[0] for item in LABELS] if split == 'audit' else [r['label'] for r in splits[split]]
                rows = [{'sample_sha256': row.get('sample_sha256', str(i)), 'predicted': int(value)}
                        for i, (row, value) in enumerate(zip(splits[split], predicted))]
                results[key][split] = compare(rows, [(label, '') for label in expected])
    if (any(sha(path) != before[name] for name, path in files.items()) or feature_files_sha256(MODEL) != config_sha
            or frozen_digest(organism, original) != digest or any(p.grad is not None for p in base.parameters())):
        raise ValueError('Protected source changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT / 'heads.pt'
    torch.save({'format': 'deepseek-multitag-abstention-research-v1', 'states': models, 'weight_sha256': WEIGHT_SHA,
                'feature_config_sha256': config_sha, 'hidden_indices': [6, 18]}, checkpoint)
    report = {'status': 'multi_persona_abstention_research_only', 'results': results, 'heads_sha256': sha(checkpoint),
              'training_sample_sha256': {name: [r['sample_sha256'] for r in splits[name]] for name in ('natural', 'multi')},
              'component_sha256': before, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Multi-persona tags are noisy proxies, not verified ambiguity or write authorization labels.',
                              'Only 12 multi-persona samples; hyperparameters examined on the existing development audit.',
                              'Training and development exclude exact repeated turns only; paraphrases and pretraining overlap unknown.',
                              'Development labels were produced by one agent and are not independently human verified.',
                              'No memory writes, identity, open dialogue or safety evaluated; data rights not separately verified.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({mode: {split: {k: v for k, v in stats.items() if k != 'rows'}
                            for split, stats in value.items()} for mode, value in results.items()}), flush=True)


if __name__ == '__main__':
    run()
