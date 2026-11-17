"""Audit pinned machine-translated Chinese SQuAD validation data locally."""
import hashlib
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))
import pyarrow.parquet as parquet

REV = '5ba5df5f19fe63b5a779a17494b51d6a92c4f54b'
URL = f'https://huggingface.co/datasets/real-jiakai/chinese-squadv2/resolve/{REV}/data/validation-00000-of-00001.parquet'
SHA = 'c74c4f811ba6f4e40a69424e40d070bf4773e26f5a006a64993a61aae4984abc'
SIZE = 2206673
SOURCE = ROOT / 'runs/chinese-squadv2-validation-5ba5df5f.parquet'
OUT = ROOT / 'runs/chinese-squadv2-validation-audit-20261117.json'


def audit(rows):
    counts, seen = Counter(), set()
    for row in rows:
        counts['rows'] += 1
        if row['id'] in seen:
            raise ValueError('Duplicate id')
        seen.add(row['id'])
        context, question, impossible, answers = row['context'], row['question'], row['is_impossible'], row['answers']
        if not isinstance(context, str) or not isinstance(question, str) or type(impossible) is not bool or not isinstance(answers, dict):
            raise ValueError('Invalid row')
        texts, starts = answers['text'], answers['answer_start']
        if len(texts) != len(starts):
            raise ValueError('Answer length mismatch')
        counts['unanswerable'] += int(impossible)
        counts['answerable'] += int(not impossible)
        counts['flag_mismatch'] += int(impossible != (len(texts) == 0))
        who = any(x in question for x in ('谁', '哪位', '何人'))
        counts['who_answerable' if not impossible else 'who_unanswerable'] += int(who)
        for text, start in zip(texts, starts):
            counts['answers'] += 1
            counts['invalid_offset'] += int(not isinstance(text, str) or type(start) is not int or start < 0
                                            or context[start:start + len(text)] != text)
    return dict(counts)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if SOURCE.exists():
        data = SOURCE.read_bytes()
    else:
        req = urllib.request.Request(URL, headers={'User-Agent': 'SPORA-research'})
        with urllib.request.urlopen(req, timeout=60) as response:
            data = response.read(SIZE + 1)
        if len(data) != SIZE or hashlib.sha256(data).hexdigest() != SHA:
            raise ValueError('Download mismatch')
        with SOURCE.open('xb') as f:
            f.write(data)
    if len(data) != SIZE or hashlib.sha256(data).hexdigest() != SHA:
        raise ValueError('Source mismatch')
    counts = audit(parquet.read_table(SOURCE).to_pylist())
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SHA:
        raise ValueError('Source changed')
    result = {'source_url': URL, 'sha256': SHA, 'license': 'CC-BY-SA-4.0', 'counts': counts,
              'limitations': ['Chinese text and annotations were machine translated and model revised.',
                  'Original SQuAD impossibility labels may not remain valid after translation.',
                  'Offset validity is not semantic correctness or source independence.',
                  'No model training or evaluation; raw text is kept local.']}
    with OUT.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
