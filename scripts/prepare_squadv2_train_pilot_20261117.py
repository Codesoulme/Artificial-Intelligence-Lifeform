"""Select mixed translated QA examples while excluding validation titles and contexts."""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import pyarrow.parquet as parquet
from scripts.audit_chinese_squadv2_20261117 import REV, SOURCE as DEV, SHA as DEV_SHA

TRAIN = ROOT / 'runs/chinese-squadv2-train-5ba5df5f.parquet'
OUT = ROOT / 'runs/squadv2-train-pilot-manifest-20261117.json'
TRAIN_SHA = 'c575af699c291d3dedd57b41f349836389d31f6869c47e72040f9a8f8fdae9b7'
TRAIN_SIZE = 38503943
URL = f'https://huggingface.co/datasets/real-jiakai/chinese-squadv2/resolve/{REV}/data/train-00000-of-00001.parquet'


def load_train():
    if TRAIN.exists():
        data = TRAIN.read_bytes()
    else:
        req = urllib.request.Request(URL, headers={'User-Agent': 'SPORA-research'})
        with urllib.request.urlopen(req, timeout=120) as response:
            data = response.read(TRAIN_SIZE + 1)
        if len(data) != TRAIN_SIZE or hashlib.sha256(data).hexdigest() != TRAIN_SHA:
            raise ValueError('Download mismatch')
        with TRAIN.open('xb') as f:
            f.write(data)
    if len(data) != TRAIN_SIZE or hashlib.sha256(data).hexdigest() != TRAIN_SHA:
        raise ValueError('Training source mismatch')


def select_train(rows, validation, per_group=48):
    excluded_titles = {r['title_en'] for r in validation}
    excluded_contexts = {r['context_en'] for r in validation}
    pool = {'known': [], 'unknown': []}
    for row in rows:
        context, question, impossible = row['context'], row['question'], row['is_impossible']
        if (row['title_en'] in excluded_titles or row['context_en'] in excluded_contexts
                or not isinstance(context, str) or not isinstance(question, str) or len(context) > 500
                or not any(x in question for x in ('谁', '哪位', '何人')) or type(impossible) is not bool):
            continue
        texts, starts = row['answers']['text'], row['answers']['answer_start']
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
        kind = 'unknown' if impossible else 'known'
        pool[kind].append({'digest': hashlib.sha256(row['id'].encode('utf8')).hexdigest(),
                           'title_digest': hashlib.sha256(row['title_en'].encode('utf8')).hexdigest(),
                           'context_digest': hashlib.sha256(row['context_en'].encode('utf8')).hexdigest(),
                           'text': context + '\n问题：' + question, 'left': left, 'right': right,
                           'label': None if impossible else 1, 'kind': kind})
    chosen, used_titles, used_contexts = [], set(), set()
    for kind in ('known', 'unknown'):
        count = 0
        for row in sorted(pool[kind], key=lambda x: x['digest']):
            if row['title_digest'] in used_titles or row['context_digest'] in used_contexts:
                continue
            chosen.append(row)
            used_titles.add(row['title_digest'])
            used_contexts.add(row['context_digest'])
            count += 1
            if count == per_group:
                break
        if count != per_group:
            raise ValueError('Insufficient disjoint training articles')
    return chosen, {k: len(v) for k, v in pool.items()}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    load_train()
    if hashlib.sha256(DEV.read_bytes()).hexdigest() != DEV_SHA:
        raise ValueError('Validation source mismatch')
    rows, eligible = select_train(parquet.read_table(TRAIN).to_pylist(), parquet.read_table(DEV).to_pylist())
    result = {'rows': [{k: r[k] for k in ('digest', 'title_digest', 'context_digest', 'kind')} for r in rows],
              'selected': len(rows), 'eligible': eligible, 'train_sha256': TRAIN_SHA, 'dev_sha256': DEV_SHA,
              'source_url': URL, 'license': 'CC-BY-SA-4.0',
              'limitations': ['Machine translated/revised labels not independently verified.',
                  'Exact English titles and contexts excluded; aliases and semantic overlap remain possible.',
                  'No explicit cross-dataset entity exclusion or pretraining non-overlap proof.',
                  'No real memories, conversation, identity or authorization labels.']}
    if hashlib.sha256(TRAIN.read_bytes()).hexdigest() != TRAIN_SHA or hashlib.sha256(DEV.read_bytes()).hexdigest() != DEV_SHA:
        raise ValueError('Source changed')
    with OUT.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({'selected': len(rows), 'eligible': eligible}), flush=True)


if __name__ == '__main__':
    main()
