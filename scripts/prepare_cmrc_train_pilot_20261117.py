"""Prepare local, attribution-preserving CMRC train selection by article and answer."""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_cmrc2018_dev_20261117 import REV, SOURCE as DEV, BLOB_SHA1 as DEV_BLOB, blob_sha1
from scripts.cmrc_who_development_selection_20261117 import select as select_dev

TRAIN = ROOT / 'runs/cmrc2018-train-c0eb1b6b.json'
OUT = ROOT / 'runs/cmrc-train-pilot-manifest-20261117.json'
TRAIN_BLOB = 'cd56f7d38f6f4a6f92df2616479bbf134a37bebf'
TRAIN_SIZE = 7408757
TRAIN_URL = f'https://raw.githubusercontent.com/ymcui/cmrc2018/{REV}/squad-style-data/cmrc2018_train.json'


def download_train():
    if TRAIN.exists():
        data = TRAIN.read_bytes()
    else:
        req = urllib.request.Request(TRAIN_URL, headers={'User-Agent': 'SPORA-research'})
        with urllib.request.urlopen(req, timeout=60) as response:
            data = response.read(TRAIN_SIZE + 1)
        if len(data) != TRAIN_SIZE or blob_sha1(data) != TRAIN_BLOB:
            raise ValueError('Download mismatch')
        with TRAIN.open('xb') as f:
            f.write(data)
    if len(data) != TRAIN_SIZE or blob_sha1(data) != TRAIN_BLOB:
        raise ValueError('Local train source mismatch')
    return data


def select_train(dataset, dev_dataset, *, limit=64):
    dev_rows, _ = select_dev(dev_dataset)
    excluded_contexts = {r['context_digest'] for r in dev_rows}
    excluded_answers = {r['text'][r['left']:r['right']] for r in dev_rows}
    # Exclude every development article title, not merely selected examples.
    excluded_titles = {a['title'] for a in dev_dataset['data']}
    selected, seen_titles, seen_answers = [], set(), set()
    pool = []
    for article in dataset['data']:
        title = article['title']
        if title in excluded_titles:
            continue
        title_hash = hashlib.sha256(title.encode('utf8')).hexdigest()
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            context_hash = hashlib.sha256(context.encode('utf8')).hexdigest()
            if len(context) > 500 or context_hash in excluded_contexts:
                continue
            for qa in paragraph['qas']:
                question, answers = qa['question'], qa['answers']
                if not any(x in question for x in ('谁', '哪位', '何人')) or not answers:
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
                if (not 2 <= len(answer) <= 8 or context.count(answer) != 1 or answer in question
                        or answer in excluded_answers):
                    continue
                pool.append({'digest': hashlib.sha256(qa['id'].encode('utf8')).hexdigest(),
                    'context_digest': context_hash, 'title_digest': title_hash,
                    'answer_digest': hashlib.sha256(answer.encode('utf8')).hexdigest(),
                    'text': context + '\n问题：' + question, 'left': start, 'right': start + len(answer),
                    'label': 1, 'kind': 'known'})
    for row in sorted(pool, key=lambda x: x['digest']):
        if row['title_digest'] in seen_titles or row['answer_digest'] in seen_answers:
            continue
        selected.append(row)
        seen_titles.add(row['title_digest'])
        seen_answers.add(row['answer_digest'])
        if len(selected) == limit:
            break
    return selected, len(pool)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    train = download_train()
    dev = DEV.read_bytes()
    if blob_sha1(dev) != DEV_BLOB:
        raise ValueError('Development source mismatch')
    rows, eligible = select_train(json.loads(train.decode('utf8')), json.loads(dev.decode('utf8')))
    if len(rows) != 64:
        raise ValueError('Insufficient eligible independent articles')
    result = {'rows': [{k: r[k] for k in ('digest', 'title_digest', 'context_digest', 'answer_digest')} for r in rows],
              'eligible_before_dedup': eligible, 'selected': len(rows),
              'train_sha256': hashlib.sha256(train).hexdigest(), 'dev_sha256': hashlib.sha256(dev).hexdigest(),
              'source_url': TRAIN_URL, 'license': 'CC-BY-SA-4.0',
              'limitations': ['Exact title and answer exclusion cannot detect aliases or semantic duplicates.',
                  'Who-questions can target groups and organizations; no unanswerable labels.',
                  'Only 32 previously selected development answers are excluded by exact string.',
                  'No claim of pretraining non-overlap or independent final test.']}
    if hashlib.sha256(TRAIN.read_bytes()).hexdigest() != result['train_sha256'] or hashlib.sha256(DEV.read_bytes()).hexdigest() != result['dev_sha256']:
        raise ValueError('Source changed')
    with OUT.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({'selected': len(rows), 'eligible': eligible}), flush=True)


if __name__ == '__main__':
    main()
