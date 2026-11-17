"""Select short, offset-consistent CMRC who-questions for local development audit."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_cmrc2018_dev_20261117 import SOURCE, BLOB_SHA1, blob_sha1

OUT = ROOT / 'runs/cmrc-who-development-selection-v2-20261117.json'
LIMIT = 32


def select(dataset, limit=LIMIT):
    pool = []
    for article in dataset['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            if not isinstance(context, str) or len(context) > 500:
                continue
            for qa in paragraph['qas']:
                question, answers = qa['question'], qa['answers']
                if not any(marker in question for marker in ('谁', '哪位', '何人')) or not answers:
                    continue
                if any(not isinstance(a.get('text'), str) or type(a.get('answer_start')) is not int
                       or not a['text'] or a['answer_start'] < 0
                       or context[a['answer_start']:a['answer_start'] + len(a['text'])] != a['text']
                       for a in answers):
                    continue
                unique = {(a['text'], a['answer_start']) for a in answers}
                if len(unique) != 1:
                    continue
                answer, start = next(iter(unique))
                if not 2 <= len(answer) <= 8 or context.count(answer) != 1 or answer in question:
                    continue
                key = hashlib.sha256(qa['id'].encode('utf8')).hexdigest()
                pool.append({'digest': key, 'context_digest': hashlib.sha256(context.encode('utf8')).hexdigest(),
                             'text': context + '\n问题：' + question, 'left': start, 'right': start + len(answer),
                             'label': 1, 'kind': 'known'})
    selected, contexts = [], set()
    for row in sorted(pool, key=lambda r: r['digest']):
        if row['context_digest'] in contexts:
            continue
        contexts.add(row['context_digest'])
        selected.append(row)
        if len(selected) == limit:
            break
    return selected, len(pool)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    data = SOURCE.read_bytes()
    if blob_sha1(data) != BLOB_SHA1:
        raise ValueError('Source mismatch')
    rows, eligible = select(json.loads(data.decode('utf8')))
    result = {'eligible_before_context_dedup': eligible, 'selected': len(rows),
              'rows': [{'digest': r['digest'], 'context_digest': r['context_digest']} for r in rows],
              'source_sha256': hashlib.sha256(data).hexdigest(),
              'limitations': ['Who-questions may target organizations or other non-person answers.',
                  'Unique short answer does not establish identity, evidence sufficiency, or authorization.',
                  'No negative examples; development data is not an independent final test.',
                  'Original texts and answer strings are not included.']}
    with OUT.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({'eligible': eligible, 'selected': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
