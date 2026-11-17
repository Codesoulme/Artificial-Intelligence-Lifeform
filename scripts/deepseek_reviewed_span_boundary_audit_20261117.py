"""Diagnose reviewed span representability using existing predictions, without inference."""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from transformers import AutoTokenizer
from scripts.audit_p236_duconv import ARCHIVE, ARCHIVE_SHA
from scripts.deepseek_local_baseline import MODEL
from scripts.deepseek_spora_consensus_reviewed_dialogue_audit import rows_from_review
from scripts.deepseek_spora_duconv_dev_identity_audit import reviewed_sample, select_sample
from scripts.deepseek_spora_joint_span_probe import MAX_SPAN
from scripts.deepseek_spora_subject_span_type_probe import token_span

PREDICTIONS = ROOT / 'runs/deepseek-spora-consensus-reviewed-dialogue-20261117.json'
OUT = ROOT / 'runs/deepseek-reviewed-span-boundary-audit-20261117.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(row, offsets, allowed, start, predictions):
    subject_mask = (allowed & (offsets[:, 0] >= start)
                    & (offsets[:, 1] <= start + len(row['text']))
                    & (offsets[:, 1] > offsets[:, 0]))
    gold, exact = token_span(offsets, start, row['left'], row['right'], subject_mask)
    indices = gold.nonzero(as_tuple=True)[0]
    contiguous = bool(len(indices) and int(indices[-1] - indices[0] + 1) == len(indices))
    usable = bool(exact and contiguous and len(indices) <= MAX_SPAN)
    linear = predictions['linear']['selected']
    context = predictions['context']['selected']
    category = ('both_reject' if not linear and not context else
                'linear_reject' if not linear else 'context_reject' if not context else
                'consensus' if predictions['consensus']['selected'] else 'disagree')
    return {'digest': row['digest'], 'category': category, 'exact_boundary': exact,
            'contiguous': contiguous, 'tokens': len(indices), 'usable': usable,
            'starts_at_sentence_start': row['left'] == 0,
            'consensus_exact': predictions['consensus']['exact']}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    sources = {'archive': ARCHIVE, 'predictions': PREDICTIONS,
               **{name: MODEL / name for name in ('tokenizer.json', 'tokenizer_config.json', 'config.json')}}
    hashes = {name: sha(path) for name, path in sources.items()}
    if hashes['archive'] != ARCHIVE_SHA:
        raise ValueError('Source archive mismatch')
    prior = json.loads(PREDICTIONS.read_text(encoding='utf8'))
    predictions = {row['digest']: row for row in prior['rows']}
    rows = rows_from_review(reviewed_sample(select_sample()))
    if len(predictions) != len(rows) or set(predictions) != {row['digest'] for row in rows}:
        raise ValueError('Prediction sample mismatch')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    details = []
    # Match feature_batch and extract batching, masks and template construction.
    for offset in range(0, len(rows), 8):
        batch = rows[offset:offset + 8]
        texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': r['text']}],
                 tokenize=False, add_generation_prompt=True) for r in batch]
        encoded = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False,
                            return_offsets_mapping=True)
        for i, row in enumerate(batch):
            if predictions[row['digest']]['group'] != row['group']:
                raise ValueError('Group mismatch')
            if row['group'] != 'single':
                continue
            start = texts[i].find(row['text'])
            if start < 0 or row['left'] < 0:
                raise ValueError('Missing source text or subject')
            details.append(diagnose(row, encoded['offset_mapping'][i],
                                    encoded['attention_mask'][i].bool(), start,
                                    predictions[row['digest']]['predictions']))
    groups = defaultdict(list)
    for row in details:
        groups[row['category']].append(row)
    summary = {name: {'count': len(items), 'usable': sum(x['usable'] for x in items),
                      'exact_boundary': sum(x['exact_boundary'] for x in items),
                      'at_sentence_start': sum(x['starts_at_sentence_start'] for x in items),
                      'token_counts': dict(sorted(Counter(x['tokens'] for x in items).items())),
                      'consensus_exact': sum(x['consensus_exact'] for x in items)}
               for name, items in sorted(groups.items())}
    if any(sha(path) != hashes[name] for name, path in sources.items()):
        raise ValueError('Source changed during audit')
    result = {'summary': summary, 'rows': details, 'source_sha256': hashes,
              'max_span': MAX_SPAN, 'limitations': [
                  'Previously reviewed development data, single annotator; not independent evaluation.',
                  'Token boundaries do not establish the cause of model rejection.',
                  'No model inference, memory writes, identity authentication, or facts checked.']}
    with OUT.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
