"""Audit immediate-repeat routing on labeled CrossWOZ validation turns."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZipFile

from audit_crosswoz_train import git_blob_sha
from deepseek_awake_recall_intent import DEFAULT_DATABASE
from deepseek_continuous_memory_chat import load_chat
from deepseek_memory_dialogue_context import features_for
from deepseek_memory_turn_priority import gate_probabilities
from deepseek_output_aligned_memory import sha, write
from spora.awake_recall_intent import RecallIntent
from spora.research_continuous_observation import ELLIPTICAL_QUERY


VAL_BLOB = '6de13a813f308802dab90d851132a17bc5064a73'
CONTACT = re.compile(r'https?://|www\.|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?<!\d)1[3-9]\d{9}(?!\d)', re.I)


def candidates(archive_path, limit):
    if archive_path.stat().st_size != 1_368_203 or git_blob_sha(archive_path.read_bytes()) != VAL_BLOB:
        raise ValueError('CrossWOZ archive differs from the pinned validation blob')
    with ZipFile(archive_path) as archive:
        if archive.namelist() != ['val.json'] or archive.getinfo('val.json').file_size != 38_909_408:
            raise ValueError('Unexpected validation archive layout')
        corpus = json.loads(archive.read('val.json'))
    grouped = dict(acknowledgement=[], switch=[], request=[])
    for identifier, dialogue in corpus.items():
        messages = dialogue['messages']
        for index, (previous, current) in enumerate(zip(messages, messages[1:]), 1):
            if (previous['role'], current['role']) != ('sys', 'usr'):
                continue
            question = current['content'].strip()
            if not 1 <= len(question) <= 130 or CONTACT.search(question):
                continue
            acts = {(act[0], act[1]) for act in current['dialog_act']}
            if acts and acts <= {('General', 'thank'), ('General', 'bye')}:
                kind = 'acknowledgement'
            elif any(action == 'Select' for action, _ in acts):
                kind = 'switch'
            elif any(action == 'Request' for action, _ in acts):
                kind = 'request'
            else:
                continue
            key = hashlib.sha256(f'spora-awake-val1/{identifier}/{index}'.encode()).hexdigest()
            grouped[kind].append((key, identifier, index, question))
    for kind, entries in grouped.items():
        selected = set()
        for key, dialogue, index, question in sorted(entries):
            if dialogue in selected:
                continue
            selected.add(dialogue)
            yield dict(key=key, dialogue=dialogue, index=index, kind=kind, question=question)
            if len(selected) == limit:
                break


def main(archive, head_path, output, limit):
    rows = list(candidates(archive, limit))
    digest = sha(DEFAULT_DATABASE)
    chat = load_chat(DEFAULT_DATABASE, use_inherited_router=True, use_retrained_gate=True)
    width = chat.decoder.base.lm_head.weight.shape[1]
    head = RecallIntent.load(head_path, width, chat.recall_fingerprint)
    summary = {}
    details = []
    for row in rows:
        features = features_for(chat.decoder, chat.tokenizer, row, False)
        old, = gate_probabilities(chat.inheritance, [row], {row['key']: features})
        existing = old >= chat.gate_cutoff
        with_rule = existing and bool(ELLIPTICAL_QUERY.fullmatch(row['question']))
        candidate = head(*features[:2])
        counts = summary.setdefault(row['kind'], dict(total=0, existing=0, existing_with_rule=0, candidate=0))
        counts['total'] += 1
        counts['existing'] += existing
        counts['existing_with_rule'] += with_rule
        counts['candidate'] += candidate
        details.append(dict(dialogue=row['dialogue'], index=row['index'], kind=row['kind'],
                            existing=existing, existing_with_rule=with_rule, candidate=candidate))
    if sha(DEFAULT_DATABASE) != digest:
        raise ValueError('Source memory database changed during audit')
    write(output, dict(summary=summary, rows=details, head_sha256=sha(head_path), validation_blob=VAL_BLOB))
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('head', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=32)
    args = parser.parse_args()
    if args.output.exists() or args.limit < 1:
        parser.error('An unused output path and positive sample limit are required')
    main(args.archive, args.head, args.output, args.limit)
