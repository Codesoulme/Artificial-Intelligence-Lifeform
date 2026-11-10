"""Compare role heuristic with a frozen-DeepSeek pronoun-pair development probe."""
import hashlib
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

from scripts.audit_mmc_pronoun_pairs import DEV, DEV_SHA, TRAIN, TRAIN_SHA, load, pair_candidates
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_pooled_address import extract
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load as load_individual
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-mmc-pair-probe-20261110'


def build_text(document, pair, current_sentence_tail=False):
    mapping = document['subtoken_map']
    speakers = [s for segment in document['speakers'] for s in segment]
    previous, current = pair['previous'], pair['current']
    p0, p1 = mapping[previous['start']], mapping[previous['end']]
    c0, c1 = mapping[current['start']], mapping[current['end']]
    if p1 >= c0:
        return None
    begin = mapping[max(0, previous['start'] - 40)]
    end = c1
    if current_sentence_tail:
        sentence_map = document['sentence_map']
        if len(sentence_map) != len(mapping):
            raise ValueError('Sentence map length mismatch')
        cursor = current['end']
        while (cursor + 1 < len(mapping) and sentence_map[cursor + 1] == sentence_map[current['end']]
               and speakers[cursor + 1] == current['speaker']):
            cursor += 1
        end = mapping[cursor]
    token_speaker = {}
    for token, speaker in zip(mapping, speakers):
        if begin <= token <= end:
            token_speaker.setdefault(token, speaker)
    if set(token_speaker) != set(range(begin, end + 1)):
        return None
    aliases, parts, last_speaker = {}, [], None
    for position in range(begin, end + 1):
        speaker = token_speaker[position]
        if speaker not in aliases:
            aliases[speaker] = str(len(aliases) + 1)
        if speaker != last_speaker:
            parts.append('\n' + aliases[speaker] + ':')
            last_speaker = speaker
        if position == p0:
            parts.append('<m1>')
        if position == c0:
            parts.append('<m2>')
        parts.append(document['tokens'][position])
        if position == p1:
            parts.append('</m1>')
        if position == c1:
            parts.append('</m2>')
    return ''.join(parts)


def select_pairs(documents, per_class, current_sentence_tail=False):
    if type(per_class) is not int or per_class < 1:
        raise ValueError('Positive per-class count required')
    candidates = []
    for document in documents:
        for pair in pair_candidates(document):
            text = build_text(document, pair, current_sentence_tail=current_sentence_tail)
            if text is None:
                continue
            identifier = f"{document['doc_key']}:{pair['previous']['start']}:{pair['current']['start']}"
            digest = hashlib.sha256(identifier.encode('utf-8')).hexdigest()
            candidates.append({'sample_sha256': digest, 'scene': document['doc_key'].rsplit('_', 1)[0],
                               'text': text, 'label': int(pair['same_entity']),
                               'heuristic': int(pair['heuristic_same']), 'speaker_count': pair['speaker_count']})
    counts, scenes, selected = {0: 0, 1: 0}, set(), []
    for row in sorted(candidates, key=lambda r: r['sample_sha256']):
        if row['scene'] in scenes or counts[row['label']] >= per_class:
            continue
        selected.append(row)
        scenes.add(row['scene'])
        counts[row['label']] += 1
        if all(value == per_class for value in counts.values()):
            return selected
    raise ValueError(f'Insufficient distinct scenes: {counts}')


def score(rows, predictions):
    if len(rows) != len(predictions):
        raise ValueError('Prediction count mismatch')
    results = {}
    for group in ('all', 'two_speakers', 'other'):
        selected = [(row, int(prediction)) for row, prediction in zip(rows, predictions)
                    if group == 'all' or (row['speaker_count'] == 2) == (group == 'two_speakers')]
        results[group] = {'pairs': len(selected), 'correct': sum(p == r['label'] for r, p in selected),
                          'false_same': sum(p == 1 and r['label'] == 0 for r, p in selected),
                          'false_different': sum(p == 0 and r['label'] == 1 for r, p in selected)}
    return results


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sources = {'weight': MODEL / 'model.safetensors', 'individual': INDIVIDUAL, 'train': TRAIN, 'dev': DEV}
    before = {name: sha(path) for name, path in sources.items()}
    if before != {'weight': WEIGHT_SHA, 'individual': INDIVIDUAL_SHA, 'train': TRAIN_SHA, 'dev': DEV_SHA}:
        raise ValueError('Protected source mismatch')
    train_docs, dev_docs = load(TRAIN, TRAIN_SHA), load(DEV, DEV_SHA)
    train = select_pairs(train_docs, 64)
    dev = select_pairs(dev_docs, 30)
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
    for name, rows in (('train', train), ('dev', dev)):
        features, masks = extract(base, tokenizer, [r['text'] for r in rows], batch_size=2)
        vectors[name] = pool(features, masks)
        print(f'{name} features ready: {len(rows)}', flush=True)
    torch.manual_seed(4702)
    classifier = nn.Linear(vectors['train'].shape[-1], 2)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.0002, weight_decay=0.01)
    labels = torch.tensor([r['label'] for r in train])
    for _ in range(600):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(classifier(vectors['train']), labels)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite loss')
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        predictions = classifier(vectors['dev']).argmax(-1).tolist()
    results = {'role_heuristic': score(dev, [r['heuristic'] for r in dev]), 'probe': score(dev, predictions)}
    if (any(sha(path) != before[name] for name, path in sources.items()) or feature_files_sha256(MODEL) != config_sha
            or frozen_digest(organism, original) != digest or any(p.grad is not None for p in base.parameters())):
        raise ValueError('Protected source changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT / 'head.pt'
    torch.save({'format': 'deepseek-mmc-pair-research-v1', 'state': classifier.state_dict(),
                'weight_sha256': WEIGHT_SHA, 'feature_config_sha256': config_sha, 'hidden_indices': [6, 18]}, checkpoint)
    report = {'status': 'pronoun_pair_development_only', 'results': results, 'head_sha256': sha(checkpoint),
              'sample_sha256': {name: [r['sample_sha256'] for r in rows] for name, rows in (('train', train), ('dev', dev))},
              'component_sha256': before, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Mentions and their positions are provided by the dataset, not predicted by the system.',
                              'Different annotated clusters are treated as different, though labels may be incomplete or noisy.',
                              'Selected balanced pairs, at most one per scene; dev was already inspected, not final evaluation.',
                              'Input is clipped at the second mention, and speaker identities are supplied as local aliases.',
                              'No write, truth, live identity, unseen-domain or conversation capability established.',
                              'Data license, model pretraining overlap and paraphrase overlap remain unknown.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    run()
