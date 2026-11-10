"""Compare relative-subject probes on weakly labelled DuLeMon turns."""
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_dulemon_routing_audit import ARCHIVE, ARCHIVE_SHA
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_relative_identity_probe import SPEAKER, ADDRESSEE, UNKNOWN
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-dulemon-persona-audit-20261110.json'
SOURCES = {'baseline': ROOT / 'runs/deepseek-relative-identity-novel-baseline-20261110',
           'expanded': ROOT / 'runs/deepseek-relative-identity-expanded-20261110'}
COUNT_PER_CLASS = 40


def normalize_turn(item):
    if not isinstance(item, str) or ':' not in item:
        return None
    speaker, content = item.split(':', 1)
    if speaker not in ('Usr', 'Bot'):
        return None
    text, separator, annotation = content.partition('\t')
    text = ''.join(text.split())
    return speaker, text, annotation if separator else ''


def sample_archive(path, per_class=COUNT_PER_CLASS):
    if type(per_class) is not int or per_class < 1:
        raise ValueError('Positive sample count required')
    candidates = []
    with zipfile.ZipFile(path) as archive, archive.open('DuLeMon/both/test.json') as stream:
        for session, line in enumerate(stream):
            if not line.strip():
                continue
            dialogue = json.loads(line).get('conversation')
            if not isinstance(dialogue, list):
                raise ValueError('Invalid conversation')
            for turn in range(1, len(dialogue)):
                current, previous = normalize_turn(dialogue[turn]), normalize_turn(dialogue[turn - 1])
                if current is None or previous is None or current[0] == previous[0]:
                    continue
                role, text, tags = current
                parts = tags.split(',') if tags else []
                if (len(parts) != 1 or len(parts[0]) < 2 or parts[0][0] not in ('U', 'B')
                        or not parts[0][1:].isdigit() or not 4 <= len(text) <= 80
                        or not 1 <= len(previous[1]) <= 100):
                    continue
                owner = parts[0][0]
                label = SPEAKER if (role == 'Usr') == (owner == 'U') else ADDRESSEE
                digest = hashlib.sha256((role + '\0' + text + '\0' + previous[0] + '\0' + previous[1]).encode('utf-8')).hexdigest()
                candidates.append({'session': session, 'turn': turn, 'role': role, 'text': text,
                                   'previous_role': previous[0], 'previous_text': previous[1],
                                   'tag': parts[0], 'label': label, 'sample_sha256': digest})
    candidates.sort(key=lambda r: r['sample_sha256'])
    selected, sessions, counts = [], set(), {SPEAKER: 0, ADDRESSEE: 0}
    for row in candidates:
        if row['session'] in sessions or counts[row['label']] >= per_class:
            continue
        selected.append(row)
        counts[row['label']] += 1
        sessions.add(row['session'])
        if all(n == per_class for n in counts.values()):
            break
    if any(n != per_class for n in counts.values()):
        raise ValueError(f'Insufficient distinct sessions per weak label: {counts}')
    return selected


def summarize(rows):
    return {'examples': len(rows), 'weak_label_match': sum(row['predicted'] == row['label'] for row in rows),
            'abstained': sum(row['predicted'] == UNKNOWN for row in rows),
            'cross_person': sum(row['predicted'] != UNKNOWN and row['predicted'] != row['label'] for row in rows)}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = {'weight': MODEL / 'model.safetensors', 'individual': INDIVIDUAL, 'archive': ARCHIVE}
    for name, source in SOURCES.items():
        files[name] = source / 'subject.pt'
        files[name + '_report'] = source / 'result.json'
    before = {key: sha(path) for key, path in files.items()}
    if before['weight'] != WEIGHT_SHA or before['individual'] != INDIVIDUAL_SHA or before['archive'] != ARCHIVE_SHA:
        raise ValueError('Protected source mismatch')
    config_sha = feature_files_sha256(MODEL)
    checkpoints = {}
    for name in SOURCES:
        with files[name + '_report'].open(encoding='utf-8') as stream:
            report = json.load(stream)
        data = torch.load(files[name], map_location='cpu', weights_only=True)
        if (report['subject_sha256'] != before[name] or data.get('format') != 'deepseek-relative-subject-development-v1'
                or data.get('weight_sha256') != WEIGHT_SHA or data.get('feature_config_sha256') != config_sha
                or data.get('hidden_indices') != [6, 18]):
            raise ValueError('Subject checkpoint mismatch')
        checkpoints[name] = data
    rows = sample_archive(ARCHIVE)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, state, _ = load(INDIVIDUAL, 'cpu')
    digest = frozen_digest(organism, state)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    inputs = {'single': [row['text'] for row in rows],
              'previous': [f"前一句（{row['previous_role']}）：{row['previous_text']}\n当前句（{row['role']}）：{row['text']}"
                           for row in rows]}
    results = {}
    for condition, texts in inputs.items():
        features, masks = extract(base, tokenizer, texts, batch_size=2)
        vectors = pool(features, masks)
        results[condition] = {}
        for name, data in checkpoints.items():
            model = nn.Linear(vectors.shape[-1], 3).eval()
            model.load_state_dict(data['state'], strict=True)
            with torch.no_grad():
                predicted = model(vectors).argmax(-1).tolist()
            scored = [dict(row, predicted=int(value)) for row, value in zip(rows, predicted)]
            results[condition][name] = {'summary': summarize(scored), 'rows': scored}
        print(f'{condition} completed', flush=True)
    if (any(sha(path) != before[name] for name, path in files.items())
            or feature_files_sha256(MODEL) != config_sha or frozen_digest(organism, state) != digest):
        raise ValueError('Protected input changed')
    report = {'status': 'weak_persona_proxy_development_audit', 'results': results, 'component_sha256': before,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Persona annotations identify a referenced persona, not necessarily the syntactic or intended subject.',
                              'The annotation-relative label is a weak proxy and is not manually verified coreference ground truth.',
                              'Previous turn is formatted as plain text in a single user message; models were not trained on this format.',
                              'Balanced 80 examples from distinct test sessions; no unknown labels, writes, identity authentication or full conversation.',
                              'DuLeMon data rights remain unverified for redistribution; raw text remains local.',
                              'Base model pretraining overlap is unknown; using this data for analysis makes it a development audit.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({c: {n: v['summary'] for n, v in scores.items()} for c, scores in results.items()}), flush=True)


if __name__ == '__main__':
    run()
