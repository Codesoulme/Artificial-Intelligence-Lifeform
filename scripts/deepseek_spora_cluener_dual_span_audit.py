"""Evaluate two experimental subject locators on a new CLUENER subset.

This is a span-identification audit, not a language-memory read/write test.
The report contains text digests and metrics rather than redistributed text.
"""
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import pyarrow.parquet as parquet
import torch
from torch import nn
from torch.nn import functional as F
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_cluener_span_audit import SOURCE, SOURCE_SHA, character_mask, labeled_names, select_rows
from scripts.deepseek_cluener_locator_train import TRAIN, TRAIN_SHA
from scripts.deepseek_learned_name_span import single_span_mask
from scripts.deepseek_spora_cluener_locator_retrain import OUTPUT as LOCATOR
from spora.identity_fingerprints import IdentityFingerprints

OUTPUT = ROOT / 'runs/deepseek-spora-cluener-dual-span-audit-20261111.json'


def select_fresh(source, prior):
    excluded = {r['text_sha256'] for r in prior}
    groups = {'single': [], 'multiple': [], 'none': []}
    seen = set()
    for index, row in enumerate(source):
        text = row.get('text')
        if not isinstance(text, str) or not 8 <= len(text) <= 60:
            continue
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if digest in seen or digest in excluded:
            continue
        seen.add(digest)
        spans = labeled_names(row)
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            continue
        group = 'none' if not spans else 'single' if len(spans) == 1 else 'multiple'
        groups[group].append({'text': text, 'spans': spans, 'digest': digest,
                              'source_index': index, 'group': group})
    selected = []
    for group, count in (('single', 20), ('multiple', 10), ('none', 10)):
        candidates = sorted(groups[group], key=lambda r: r['digest'])
        if len(candidates) < count:
            raise ValueError('Insufficient fresh source examples')
        selected.extend(candidates[:count])
    return selected


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if runtime.sha(SOURCE) != SOURCE_SHA or runtime.sha(TRAIN) != TRAIN_SHA:
        raise ValueError('CLUENER source digest mismatch')
    source = parquet.read_table(SOURCE).to_pylist()
    prior, _ = select_rows(source)
    selected = select_fresh(source, prior)
    train_texts = {r['text'] for r in parquet.read_table(TRAIN).to_pylist()}
    if any(row['text'] in train_texts for row in selected):
        raise ValueError('Fresh selected text overlaps locator training data')
    candidate_path = LOCATOR / 'locator.pt'
    candidate_report = json.loads((LOCATOR / 'result.json').read_text(encoding='utf-8'))
    candidate_sha = runtime.sha(candidate_path)
    if candidate_sha != candidate_report['locator_sha256']:
        raise ValueError('Candidate locator digest mismatch')
    candidate_data = torch.load(candidate_path, map_location='cpu', weights_only=True)
    source_sha = runtime.sha(runtime.CHECKPOINT)
    torch.set_num_threads(2)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = runtime.load_chat()
        if candidate_data.get('feature_variant_sha256') != chat.router.get('feature_variant_sha256'):
            raise ValueError('Candidate locator feature variant mismatch')
        questions = [row['text'] for row in selected]
        texts = [chat.tokenizer.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False,
                                                    add_generation_prompt=True) for q in questions]
        encoded = chat.tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False,
                                 return_offsets_mapping=True)
        features, masks = runtime.extract(chat.base, chat.tokenizer, questions, batch_size=1)
        vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
        default = nn.Linear(vectors.shape[-1], 1).eval()
        default.load_state_dict(chat.router['locator_state'], strict=True)
        candidate = nn.Linear(vectors.shape[-1], 1).eval()
        candidate.load_state_dict(candidate_data['state'], strict=True)
        with torch.no_grad():
            scores = (default(vectors).squeeze(-1).sigmoid(), candidate(vectors).squeeze(-1).sigmoid())
        rows = []
        for i, row in enumerate(selected):
            text_start = texts[i].find(row['text'])
            if text_start < 0:
                raise ValueError('Source text missing from template')
            gold = torch.zeros(len(row['text']), dtype=torch.bool)
            for left, right in row['spans']:
                gold[left:right] = True
            candidates = []
            for values, threshold in ((scores[0], chat.router['locator_threshold']),
                                      (scores[1], candidate_data['threshold'])):
                token_mask, single = single_span_mask(values[i], masks[i], threshold)
                chars = character_mask(encoded['offset_mapping'][i], token_mask, text_start, len(row['text']))
                candidates.append({'single': single, 'exact_gold': bool(torch.equal(chars, gold)),
                                   'selected_characters': int(chars.sum()), 'mask': chars})
            fingerprint_matches = [False, False]
            if row['group'] == 'single':
                left, right = row['spans'][0]
                reference = row['text'][left:right]
                binding = IdentityFingerprints(1)
                binding.bind(0, reference)  # oracle enrollment only; never used by the locator
                for j, item in enumerate(candidates):
                    text = ''.join(row['text'][k] for k in item['mask'].nonzero(as_tuple=True)[0].tolist())
                    try:
                        fingerprint_matches[j] = bool(item['single'] and binding.lookup(text) == 0)
                    except ValueError:
                        fingerprint_matches[j] = False
            rows.append({'group': row['group'], 'source_index': row['source_index'], 'text_sha256': row['digest'],
                         'default_single': candidates[0]['single'], 'candidate_single': candidates[1]['single'],
                         'default_gold_exact': candidates[0]['exact_gold'],
                         'candidate_gold_exact': candidates[1]['exact_gold'],
                         'default_fingerprint_match': fingerprint_matches[0],
                         'candidate_fingerprint_match': fingerprint_matches[1],
                         'dual_fingerprint_match': all(fingerprint_matches)})
    if runtime.sha(runtime.CHECKPOINT) != source_sha or runtime.sha(candidate_path) != candidate_sha:
        raise ValueError('Protected input changed')
    single = [r for r in rows if r['group'] == 'single']
    summary = {'single_total': len(single),
               'default_fingerprint_match': sum(r['default_fingerprint_match'] for r in single),
               'candidate_fingerprint_match': sum(r['candidate_fingerprint_match'] for r in single),
               'dual_fingerprint_match': sum(r['dual_fingerprint_match'] for r in single)}
    for group in ('multiple', 'none'):
        group_rows = [r for r in rows if r['group'] == group]
        summary[group] = {'total': len(group_rows),
                          'default_single_span': sum(r['default_single'] for r in group_rows),
                          'candidate_single_span': sum(r['candidate_single'] for r in group_rows)}
    report = {'status': 'read_only_news_span_audit', 'summary': summary, 'rows': rows,
              'selection': 'next 20 single/10 multiple/10 none by text SHA, excluding prior selected 40 and exact training texts',
              'source_sha256': SOURCE_SHA, 'training_source_sha256': TRAIN_SHA,
              'source_checkpoint_sha256': source_sha, 'candidate_locator_sha256': candidate_sha,
              'seconds': time.perf_counter() - started,
              'limitations': ['CLUENER news text, not a dialogue or memory question benchmark.',
                              'One reference name is oracle-enrolled per single-name example.',
                              'Previous experiments used the same validation split for development.',
                              'Annotation errors, tokenizer boundary merges, and base pretraining overlap are possible.',
                              'No memory read/write, identity enrollment workflow or real user state evaluated.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
