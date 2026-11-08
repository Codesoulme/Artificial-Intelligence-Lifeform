"""Development-only P112 probe. Never exports examples, predictions, or model weights."""
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 11620261109
DIM = 2048
FOLDS = 5


def norm(s):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', s)).lower()


def digest(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def load_data():
    rows = [json.loads(x) for x in (ROOT / 'runs/p112-story-quality-sample/samples.jsonl').read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    mapping = json.loads((ROOT / 'runs/p112-story-quality-blind/mapping.json').read_text(encoding='utf-8-sig'))
    by_key = {}
    for r in rows:
        t = norm(r['text'])
        if digest(t) != r['normalized_sha256'] or digest(t[:96]) != r['group_sha256'] or digest(r['text']) != r['source_original_sha256']:
            raise ValueError('sample identity mismatch')
        key = (r['condition'], r['source_row_index'], r['normalized_sha256'])
        if key in by_key:
            raise ValueError('duplicate sample')
        by_key[key] = r
    seen = set()
    for reviewer in range(1, 5):
        grades = json.loads((ROOT / f'runs/p112-story-quality-blind/grade-{reviewer}.json').read_text(encoding='utf-8-sig'))
        for g in grades:
            bid = g['blind_id']
            if bid in seen or bid not in mapping:
                raise ValueError('duplicate or missing blind mapping')
            seen.add(bid)
            m = mapping[bid]
            key = (m['condition'], m['source_row_index'], m['normalized_sha256'])
            if key not in by_key or 'label' in by_key[key]:
                raise ValueError('unmatched or duplicate assignment')
            if g['judgment'] not in ('clean', 'minor', 'major', 'uncertain'):
                raise ValueError('unknown judgment')
            by_key[key]['label'] = g['judgment']
            by_key[key]['reviewer'] = str(reviewer)
    if len(rows) != 160 or len(mapping) != 160 or len(seen) != 160 or set(mapping) != seen or any('label' not in r for r in rows):
        raise ValueError('incomplete assignment')
    return rows


def groups(texts, threshold=0.6):
    """Transitive closure of same prefix or 5-character shingle Jaccard >= .6."""
    texts = [norm(t) for t in texts]
    shingles = [set(t[i:i+5] for i in range(max(1, len(t)-4))) for t in texts]
    parent = list(range(len(texts)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(texts)):
        for j in range(i):
            union = len(shingles[i] | shingles[j])
            if texts[i][:96] == texts[j][:96] or (union and len(shingles[i] & shingles[j]) / union >= threshold):
                parent[root(i)] = root(j)
    return [root(i) for i in range(len(texts))]


def split_groups(y, group, seed=SEED, folds=FOLDS):
    buckets = defaultdict(list)
    for i, g in enumerate(group): buckets[g].append(i)
    rng = np.random.default_rng(seed)
    items = list(buckets.values())
    rng.shuffle(items)
    items.sort(key=lambda ix: (-sum(y[i] for i in ix), -len(ix)))
    counts = np.zeros((folds, 2), dtype=int)
    assignment = np.zeros(len(y), dtype=int)
    for ix in items:
        a = sum(y[i] for i in ix); b = len(ix) - a
        scores = []
        for f in range(folds):
            trial = counts.copy(); trial[f] += [a, b]
            scores.append(np.var(trial[:, 0]) / max(1, sum(y))**2 + np.var(trial[:, 1]) / max(1, len(y)-sum(y))**2)
        f = min(range(folds), key=lambda k: (scores[k], counts[k].sum(), k))
        counts[f] += [a, b]; assignment[ix] = f
    if any(not (0 < sum(y[i] for i in range(len(y)) if assignment[i] == f) < sum(assignment == f)) for f in range(folds)):
        raise ValueError('insufficient groups for stratified evaluation')
    return assignment


def features(text):
    """Fixed signed hashing of character 2/3-grams; no fitted vocabulary."""
    s = norm(text)
    v = np.zeros(DIM, dtype=float)
    for n in (2, 3):
        for i in range(max(0, len(s)-n+1)):
            h = hashlib.blake2b(s[i:i+n].encode('utf-8'), digest_size=8, person=b'p116char').digest()
            k = int.from_bytes(h, 'little')
            v[k % DIM] += 1 if (k >> 11) & 1 else -1
    length = np.linalg.norm(v)
    return v / length if length else v


def predict_model(x, y, xt):
    """L2 logistic regression, zero weights, balanced class loss; fixed hyperparameters."""
    w = np.zeros(x.shape[1]); b = 0.0
    weights = np.where(y == 1, len(y)/(2*sum(y)), len(y)/(2*(len(y)-sum(y))))
    for _ in range(500):
        p = 1 / (1 + np.exp(-np.clip(x @ w + b, -30, 30)))
        error = (p-y)*weights
        w -= 0.5 * (x.T @ error / len(y) + 0.01*w)
        b -= 0.5 * np.mean(error)
    return (xt @ w + b >= 0).astype(int)


def rule(text):
    """Fixed surface rule: extremely short or character-loop repetition."""
    s = norm(text)
    return int(len(s) < 80 or bool(re.search(r'(.{2,12})\1{3,}', s)))


def wilson(k, n):
    if not n: return None
    z = 1.959963984540054; p = k/n; d = 1+z*z/n
    center = (p+z*z/(2*n))/d
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [round(center-half, 4), round(center+half, 4)]


def metrics(y, pred):
    y = np.asarray(y); pred = np.asarray(pred)
    tp = int(((y == 1)&(pred == 1)).sum()); fn = int(((y == 1)&(pred == 0)).sum())
    fp = int(((y == 0)&(pred == 1)).sum()); tn = int(((y == 0)&(pred == 0)).sum())
    def rate(k, n): return {'count': k, 'denominator': n, 'rate': round(k/n, 4) if n else None, 'wilson95': wilson(k,n)}
    return {'tp':tp, 'fn':fn, 'fp':fp, 'tn':tn, 'recall':rate(tp,tp+fn), 'retention':rate(tn+fn,len(y)), 'false_reject':rate(fp,fp+tn), 'miss':rate(fn,tp+fn), 'precision':rate(tp,tp+fp)}


def main():
    out = ROOT / 'runs/p116-story-quality-probe'
    if out.exists(): raise FileExistsError('output directory exists; refuse overwrite')
    rows = load_data()
    uncertain = sum(r['label'] == 'uncertain' for r in rows)
    rows = [r for r in rows if r['label'] != 'uncertain']
    y = np.array([int(r['label'] == 'major') for r in rows])
    g = groups([r['text'] for r in rows]); fold = split_groups(y,g)
    x = np.stack([features(r['text']) for r in rows])
    model = np.zeros(len(rows),dtype=int)
    for f in range(FOLDS):
        train = fold != f; test = ~train
        assert not (set(np.array(g)[train]) & set(np.array(g)[test]))
        model[test] = predict_model(x[train],y[train],x[test])
    predictions = {'keep_all':np.zeros(len(rows),dtype=int), 'surface_rule':np.array([rule(r['text']) for r in rows]), 'zero_logistic':model}
    result = {'seed':SEED, 'n':len(rows), 'uncertain_excluded':uncertain, 'labels':dict(Counter(r['label'] for r in rows)), 'groups':len(set(g)), 'group_sizes':dict(Counter(Counter(g).values())), 'folds':[{'n':int(sum(fold==f)), 'major':int(sum(y[fold==f]))} for f in range(FOLDS)], 'overall':{}, 'source':{}, 'reviewer':{}, 'source_reviewer':{}}
    for name,p in predictions.items(): result['overall'][name] = metrics(y,p)
    for field in ('condition','reviewer'):
        target = result['source' if field == 'condition' else 'reviewer']
        for val in sorted(set(r[field] for r in rows)):
            ix = np.array([r[field]==val for r in rows]); target[val] = {name:metrics(y[ix],p[ix]) for name,p in predictions.items()}
    for a in sorted(set(r['condition'] for r in rows)):
        for b in sorted(set(r['reviewer'] for r in rows)):
            ix = np.array([r['condition']==a and r['reviewer']==b for r in rows])
            result['source_reviewer'][a+'-'+b] = {'n':int(ix.sum()),'major':int(y[ix].sum())}
    out.mkdir(parents=True)
    (out/'aggregate.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__': main()
