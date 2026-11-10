"""Test whether noisy persona labels improve a subject probe; never authorize writes."""
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

from scripts.deepseek_context_subject_probe import PREVIOUS, compose
from scripts.deepseek_dulemon_persona_audit import normalize_turn
from scripts.deepseek_dulemon_routing_audit import ARCHIVE, ARCHIVE_SHA
from scripts.deepseek_dulemon_subject_annotation import LABELS, SOURCE, SOURCE_SHA, compare
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_relative_identity_probe import ADDRESSEE, AMBIGUOUS_TRAIN, HELD, NOVEL, SPEAKER, TRAIN, build_cases
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-weak-persona-training-20261110'


def sample_training(path, excluded_texts, per_class=64):
    if type(per_class) is not int or per_class < 1:
        raise ValueError('Positive per-class count required')
    candidates = []
    with zipfile.ZipFile(path) as archive, archive.open('DuLeMon/both/train.json') as stream:
        for session, line in enumerate(stream):
            dialogue = json.loads(line).get('conversation')
            if not isinstance(dialogue, list):
                raise ValueError('Invalid conversation')
            for turn in range(1, len(dialogue)):
                previous, current = normalize_turn(dialogue[turn - 1]), normalize_turn(dialogue[turn])
                if current is None or previous is None or previous[0] == current[0]:
                    continue
                role, text, tags = current
                if (text in excluded_texts or previous[1] in excluded_texts or not 4 <= len(text) <= 80
                        or not 1 <= len(previous[1]) <= 100):
                    continue
                parts = tags.split(',') if tags else []
                if len(parts) != 1 or len(parts[0]) < 2 or parts[0][0] not in ('U', 'B') or not parts[0][1:].isdigit():
                    continue
                label = SPEAKER if (role == 'Usr') == (parts[0][0] == 'U') else ADDRESSEE
                digest = hashlib.sha256((role + '\0' + text + '\0' + previous[0] + '\0' + previous[1]).encode()).hexdigest()
                candidates.append({'session': session, 'role': role, 'text': text, 'previous_role': previous[0],
                                   'previous_text': previous[1], 'label': label, 'sample_sha256': digest})
    counts = {SPEAKER: 0, ADDRESSEE: 0}
    selected, sessions, texts = [], set(), set()
    for row in sorted(candidates, key=lambda r: r['sample_sha256']):
        if row['session'] in sessions or row['text'] in texts or counts[row['label']] >= per_class:
            continue
        selected.append(row)
        sessions.add(row['session'])
        texts.add(row['text'])
        counts[row['label']] += 1
        if all(value == per_class for value in counts.values()):
            return selected
    raise ValueError(f'Insufficient distinct sessions: {counts}')


def contextualize(rows, natural):
    result = []
    for i, row in enumerate(rows):
        if natural:
            result.append(compose(row['previous_role'], row['previous_text'], row['role'], row['text'])[0])
        else:
            role = 'Usr' if i % 2 == 0 else 'Bot'
            result.append(compose('Bot' if role == 'Usr' else 'Usr', PREVIOUS[(i // 2) % len(PREVIOUS)],
                                  role, row['text'])[0])
    return result


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = {'weight': MODEL / 'model.safetensors', 'individual': INDIVIDUAL, 'audit': SOURCE, 'archive': ARCHIVE}
    before = {name: sha(path) for name, path in files.items()}
    if (before['weight'] != WEIGHT_SHA or before['individual'] != INDIVIDUAL_SHA
            or before['audit'] != SOURCE_SHA or before['archive'] != ARCHIVE_SHA):
        raise ValueError('Protected source mismatch')
    with SOURCE.open(encoding='utf-8') as stream:
        source = json.load(stream)
    all_audit = source['results']['single']['baseline']['rows']
    audit = all_audit[:len(LABELS)]
    excluded = {row[key] for row in all_audit for key in ('text', 'previous_text')}
    natural = sample_training(ARCHIVE, excluded)
    splits = {'synthetic': build_cases(TRAIN) + build_cases(AMBIGUOUS_TRAIN), 'natural': natural,
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
        texts = contextualize(rows, split in ('natural', 'audit'))
        features, masks = extract(base, tokenizer, texts, batch_size=2)
        vectors[split] = pool(features, masks)
        print(f'{split} features ready: {len(rows)}', flush=True)
    labels = {split: torch.tensor([row['label'] for row in splits[split]]) for split in ('synthetic', 'natural')}
    results, models = {}, {}
    for mode in ('synthetic_only', 'mixed_weak'):
        torch.manual_seed(4681)
        classifier = nn.Linear(vectors['synthetic'].shape[-1], 3)
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.0002, weight_decay=0.01)
        for _ in range(600):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(classifier(vectors['synthetic']), labels['synthetic'])
            if mode == 'mixed_weak':
                loss = (loss + F.cross_entropy(classifier(vectors['natural']), labels['natural'])) / 2
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            loss.backward()
            optimizer.step()
        models[mode] = classifier.state_dict()
        results[mode] = {}
        with torch.no_grad():
            for split in ('held', 'novel', 'audit'):
                predicted = classifier(vectors[split]).argmax(-1).tolist()
                expected = [item[0] for item in LABELS] if split == 'audit' else [r['label'] for r in splits[split]]
                rows = [{'sample_sha256': row.get('sample_sha256', str(i)), 'predicted': int(value)}
                        for i, (row, value) in enumerate(zip(splits[split], predicted))]
                results[mode][split] = compare(rows, [(label, '') for label in expected])
    if (any(sha(path) != before[name] for name, path in files.items()) or feature_files_sha256(MODEL) != config_sha
            or frozen_digest(organism, original) != digest or any(p.grad is not None for p in base.parameters())):
        raise ValueError('Protected source changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT / 'heads.pt'
    torch.save({'format': 'deepseek-weak-persona-research-v1', 'states': models, 'weight_sha256': WEIGHT_SHA,
                'feature_config_sha256': config_sha, 'hidden_indices': [6, 18]}, checkpoint)
    report = {'status': 'noisy_persona_subject_research_only', 'results': results, 'heads_sha256': sha(checkpoint),
              'training_sample_sha256': [row['sample_sha256'] for row in natural], 'component_sha256': before,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Persona owner is a noisy proxy, not a subject or write authorization label.',
                              'Train/test partition is dataset-provided; only exact audit turn text overlap was excluded.',
                              'Audit labels are agent-annotated development data without independent human review.',
                              'Model pretraining overlap, train/test paraphrase and participant overlap remain unknown.',
                              'Data license not separately verified; no redistribution or open deployment.',
                              'No facts, identity authentication, write decisions or multi-session safety evaluated.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({mode: {split: {k: v for k, v in stats.items() if k != 'rows'}
                            for split, stats in value.items()} for mode, value in results.items()}), flush=True)


if __name__ == '__main__':
    run()
