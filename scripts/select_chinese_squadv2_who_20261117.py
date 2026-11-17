"""Build a small local development sample including translated no-answer questions."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import pyarrow.parquet as parquet
from scripts.audit_chinese_squadv2_20261117 import SOURCE, SHA

OUT = ROOT / 'runs/chinese-squadv2-who-selection-20261117.json'


def select(rows, per_group=16):
    pool = {'known': [], 'unknown': []}
    for row in rows:
        context, question, impossible = row['context'], row['question'], row['is_impossible']
        if (not isinstance(context, str) or not isinstance(question, str) or len(context) > 500
                or not any(x in question for x in ('谁', '哪位', '何人')) or type(impossible) is not bool):
            continue
        answers = row['answers']
        texts, starts = answers['text'], answers['answer_start']
        if len(texts) != len(starts) or impossible != (len(texts) == 0):
            continue
        if impossible:
            left = right = None
        else:
            pairs = set(zip(texts, starts))
            if len(pairs) != 1:
                continue
            answer, left = next(iter(pairs))
            if (not isinstance(answer, str) or type(left) is not int or left < 0
                    or not 2 <= len(answer) <= 8 or context[left:left + len(answer)] != answer
                    or context.count(answer) != 1 or answer in question):
                continue
            right = left + len(answer)
        group = 'unknown' if impossible else 'known'
        pool[group].append({'digest': hashlib.sha256(row['id'].encode('utf8')).hexdigest(),
             'context_digest': hashlib.sha256(context.encode('utf8')).hexdigest(),
             'text': context + '\n问题：' + question, 'left': left, 'right': right,
             'label': None if impossible else 1, 'kind': group})
    result, used = [], set()
    for group in ('known', 'unknown'):
        count = 0
        for row in sorted(pool[group], key=lambda x: x['digest']):
            if row['context_digest'] in used:
                continue
            used.add(row['context_digest'])
            result.append(row)
            count += 1
            if count == per_group:
                break
        if count != per_group:
            raise ValueError('Insufficient distinct contexts')
    return result, {group: len(items) for group, items in pool.items()}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SHA:
        raise ValueError('Source mismatch')
    rows, eligible = select(parquet.read_table(SOURCE).to_pylist())
    result = {'selected': len(rows), 'eligible': eligible, 'source_sha256': SHA,
              'rows': [{k: row[k] for k in ('digest', 'context_digest', 'kind')} for row in rows],
              'limitations': ['Machine-translated and model-revised data; labels require human verification.',
                  'No-answer means original dataset marks the paragraph insufficient, not a real-world fact false.',
                  'Who-questions may refer to organizations or groups, not necessarily persons.',
                  'Raw contexts and questions remain local.']}
    with OUT.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({'selected': len(rows), 'eligible': eligible}), flush=True)


if __name__ == '__main__':
    main()
