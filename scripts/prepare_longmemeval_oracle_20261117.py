"""Prepare an evidence-only, evaluation-only LongMemEval interface.

Oracle sessions contain only gold-selected evidence. They are NOT a full
retrieval benchmark. Strip source identifiers and gold flags before inference.
"""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/research/longmemeval-cleaned-20261117/longmemeval_oracle.json'
OUT = ROOT / 'runs/research/longmemeval-oracle-prepared-20261117'
EXPECTED_SHA256 = '821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c'
DATE_FORMAT = '%Y/%m/%d (%a) %H:%M'


def prepare_case(record, case_id):
    """Return public input and private grading metadata, keeping them disjoint."""
    ids = record['haystack_session_ids']
    dates = record['haystack_dates']
    sessions = record['haystack_sessions']
    if not (len(ids) == len(dates) == len(sessions)) or len(set(ids)) != len(ids):
        raise ValueError('Mismatched or duplicate session ids')
    if not set(record['answer_session_ids']).issubset(ids):
        raise ValueError('Missing evidence session')
    datetime.strptime(record['question_date'], DATE_FORMAT)
    ordered = sorted(range(len(ids)), key=lambda i: (datetime.strptime(dates[i], DATE_FORMAT), i))
    public_sessions = []
    gold_sessions = []
    gold_turns = []
    for new_index, old_index in enumerate(ordered):
        items = sessions[old_index]
        cleaned = []
        for turn_index, turn in enumerate(items):
            if turn['role'] not in ('user', 'assistant') or not isinstance(turn['content'], str):
                raise ValueError('Unexpected message shape')
            cleaned.append({'role': turn['role'], 'content': turn['content']})
            if turn.get('has_answer') is True:
                gold_turns.append([new_index, turn_index])
        if ids[old_index] in record['answer_session_ids']:
            gold_sessions.append(new_index)
        public_sessions.append({'observed_at': dates[old_index], 'turns': cleaned})
    public = {'case_id': case_id, 'question_date': record['question_date'],
              'question': record['question'], 'sessions': public_sessions}
    private = {'case_id': case_id, 'original_question_id': record['question_id'],
               'question_type': record['question_type'], 'answer': record['answer'],
               'abstention': str(record['question_id']).endswith('_abs'),
               'evidence_session_indexes': gold_sessions, 'evidence_turn_indexes': gold_turns}
    return public, private


def write_jsonl(path, rows):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    data = args.source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError('Source digest mismatch; audit a new version before preparing it')
    records = json.loads(data)
    if len(records) != 500 or len({r['question_id'] for r in records}) != len(records):
        raise ValueError('Unexpected source records')
    prepared = [prepare_case(record, f'case-{i:04d}') for i, record in enumerate(records)]
    public, gold = zip(*prepared)
    manifest = {'source': 'https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned',
                'source_file': 'longmemeval_oracle.json', 'source_sha256': digest,
                'license': 'MIT', 'case_count': len(records),
                'question_types': dict(Counter(g['question_type'] for g in gold)),
                'abstention_cases': sum(g['abstention'] for g in gold),
                'limitations': ['English benchmark-constructed conversations, not organic Chinese dialogue.',
                                'Oracle contains only selected evidence; not a complete retrieval test.',
                                'Question types and answers must remain outside the inference process.',
                                'Observed session order is not necessarily event occurrence order.',
                                'Evaluation-only: never use this benchmark to train or tune the model.']}
    args.out.mkdir(parents=True)
    write_jsonl(args.out / 'input.jsonl', public)
    write_jsonl(args.out / 'gold.private.jsonl', gold)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == '__main__':
    main()
