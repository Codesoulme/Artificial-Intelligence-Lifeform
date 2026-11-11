"""Controlled held-out-name dialogue probe; no memory reads or deployment."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import pyarrow.parquet as parquet
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer

from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_cluener_span_audit import SOURCE, SOURCE_SHA, labeled_names, select_rows
from scripts.deepseek_cluener_locator_train import TRAIN, TRAIN_SHA
from scripts.deepseek_learned_name_span import single_span_mask
from scripts.deepseek_spora_cluener_dual_span_audit import select_fresh
from scripts.deepseek_spora_cluener_locator_retrain import OUTPUT as LOCATOR
from scripts.deepseek_spora_prefix_features import load_prefix
from scripts.deepseek_spora_prefix_parity_audit import OUTPUT as PARITY
from spora.identity_fingerprints import IdentityFingerprints

OUTPUT = ROOT / 'runs/deepseek-spora-conversation-locator-audit-v2-20261112.json'


def select_names(source, train_texts, count=20):
    prior, _ = select_rows(source)
    recent = select_fresh(source, prior)
    excluded = {r['text_sha256'] for r in prior} | {r['digest'] for r in recent}
    candidates = []
    for row in source:
        text = row.get('text')
        if not isinstance(text, str) or text in train_texts:
            continue
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        spans = labeled_names(row)
        if digest in excluded or len(spans) != 1:
            continue
        start, end = spans[0]
        name = text[start:end]
        if not 2 <= len(name) <= 4 or any(not '\u4e00' <= char <= '\u9fff' for char in name):
            continue
        candidates.append((digest, name))
    names = []
    for digest, name in sorted(candidates):
        if name not in {item[1] for item in names}:
            names.append((digest, name))
            if len(names) == count:
                return names
    raise ValueError('Insufficient distinct names')


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if runtime.sha(SOURCE) != SOURCE_SHA or runtime.sha(TRAIN) != TRAIN_SHA:
        raise ValueError('Source digest mismatch')
    source = parquet.read_table(SOURCE).to_pylist()
    training = {row['text'] for row in parquet.read_table(TRAIN).to_pylist()}
    names = select_names(source, training)
    cases = []
    for digest, name in names:
        for group, subject in (('known', name), ('near_suffix', name + '子'), ('near_prefix', '小' + name)):
            cases.append({'group': group, 'known_name': name, 'subject': subject,
                          'source_digest': digest, 'question': f'{subject}现在喜欢喝什么？'})
    parity = json.loads(PARITY.read_text(encoding='utf-8'))
    source_sha = runtime.sha(runtime.CHECKPOINT)
    if (parity['source_checkpoint_sha256'] != source_sha or parity['examples'] != 10
            or parity['span_matches'] != 10 or parity['max_entity_score_difference'] > 0.001):
        raise ValueError('Prefix parity evidence missing or inconsistent')
    candidate_path = LOCATOR / 'locator.pt'
    candidate_report = json.loads((LOCATOR / 'result.json').read_text(encoding='utf-8'))
    candidate_sha = runtime.sha(candidate_path)
    if candidate_sha != candidate_report['locator_sha256']:
        raise ValueError('Candidate locator digest mismatch')
    candidate_data = torch.load(candidate_path, map_location='cpu', weights_only=True)
    torch.set_num_threads(2)
    started = time.perf_counter()
    model, router, variant = load_prefix()
    if variant != parity['feature_variant_sha256'] or variant != candidate_data['feature_variant_sha256']:
        raise ValueError('Feature variant mismatch')
    tokenizer = AutoTokenizer.from_pretrained(runtime.MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    questions = [case['question'] for case in cases]
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False,
                                           add_generation_prompt=True) for q in questions]
    encoded = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False,
                        return_offsets_mapping=True)
    features, masks = runtime.extract(model, tokenizer, questions, batch_size=1)
    vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
    heads = []
    for weights in (router['locator_state'], candidate_data['state']):
        head = nn.Linear(vectors.shape[-1], 1).eval()
        head.load_state_dict(weights, strict=True)
        heads.append(head)
    with torch.no_grad():
        scores = [head(vectors).squeeze(-1).sigmoid() for head in heads]
    rows = []
    for i, case in enumerate(cases):
        question = case['question']
        start = texts[i].find(question)
        binding = IdentityFingerprints(1)
        binding.bind(0, case['known_name'])
        accepted, exact_subject, single_span = [], [], []
        for values, threshold in zip(scores, (router['locator_threshold'], candidate_data['threshold'])):
            selected, single = single_span_mask(values[i], masks[i], threshold)
            chars = set()
            for token in selected.nonzero(as_tuple=True)[0].tolist():
                left, right = encoded['offset_mapping'][i, token].tolist()
                chars.update(range(max(0, left - start), min(len(question), right - start)))
            picked = ''.join(question[j] for j in sorted(chars))
            exact_subject.append(bool(single and picked == case['subject']))
            single_span.append(bool(single))
            try:
                accepted.append(bool(single and binding.lookup(picked) == 0))
            except ValueError:
                accepted.append(False)
        rows.append({'group': case['group'], 'source_digest': case['source_digest'],
                     'question_sha256': hashlib.sha256(question.encode('utf-8')).hexdigest(),
                     'default_match': accepted[0], 'candidate_match': accepted[1],
                     'dual_match': all(accepted), 'default_exact_subject': exact_subject[0],
                     'candidate_exact_subject': exact_subject[1], 'default_single_span': single_span[0],
                     'candidate_single_span': single_span[1]})
    summary = {}
    for group in ('known', 'near_suffix', 'near_prefix'):
        subset = [row for row in rows if row['group'] == group]
        summary[group] = {'total': len(subset), **{key: sum(row[key] for row in subset)
                          for key in ('default_match', 'candidate_match', 'dual_match', 'default_exact_subject',
                                      'candidate_exact_subject', 'default_single_span', 'candidate_single_span')}}
    if runtime.sha(runtime.CHECKPOINT) != source_sha or runtime.sha(candidate_path) != candidate_sha:
        raise ValueError('Protected input changed')
    report = {'status': 'controlled_conversation_span_probe', 'summary': summary, 'rows': rows,
              'source_sha256': SOURCE_SHA, 'training_source_sha256': TRAIN_SHA,
              'source_checkpoint_sha256': source_sha, 'candidate_locator_sha256': candidate_sha,
              'feature_variant_sha256': variant, 'seconds': time.perf_counter() - started,
              'limitations': ['Artificial questions in a single repeated template, not natural dialogue.',
                              'Names from CLUENER validation; split and related cases were used for development.',
                              'Near-name subjects are defined as distinct solely by experiment.',
                              'Reference name is manually enrolled separately for each case.',
                              'Measures predicted span and fingerprint only, not actual memory reads or answer text.',
                              'Reduced prefix parity checked on ten different examples only.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
