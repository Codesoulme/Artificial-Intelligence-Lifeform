"""Create an unapproved, train-only screening and blinded-review package."""
import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/wikimedia-intro-candidate-v3/train.jsonl'
EXPECTED = '2312663f7ace88aebb704171c5dc306e5b86958105576717a5cf297b28eebfdc'
LINK_SOURCE = ROOT / 'runs/wikimedia-intro-candidate-v1/train.jsonl'
LINK_SHA = 'ac528a753ace33f9567e57359df86671a4fa80de79954ccb31e2b03d9f04fc7f'
OUT = ROOT / 'runs/wikimedia-intro-screening-p77-v2'
PRIOR = [
    ('runs/research/wikimedia-wikipedia-zh/p71-sample.json', 40, 'ec8f6c0dd1418b8fcd454278fb4f2bc7cd0ce8312f29c80b672533942601338f'),
    ('runs/research/wikimedia-wikipedia-zh/p72-sample.json', 40, 'ec8f6c0dd1418b8fcd454278fb4f2bc7cd0ce8312f29c80b672533942601338f'),
    ('runs/wikimedia-intro-candidate-v1/review-sample.json', 40, 'ac528a753ace33f9567e57359df86671a4fa80de79954ccb31e2b03d9f04fc7f'),
    ('runs/wikimedia-intro-candidate-v2/review-sample.json', 60, 'addf534bdad82afb96f62af75fe476230f7519234938e824eef66827b3cec69c'),
    ('runs/wikimedia-intro-candidate-v3/review-sample.json', 60, EXPECTED),
]
SEED = 'p77-train-screen-review-20261108-v1'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def normalized(text):
    return re.sub(r'[^\w]+', '', unicodedata.normalize('NFKC', text).casefold())


def grams(text):
    text = normalized(text)
    return {text[i:i+5] for i in range(len(text)-4)}


def flags(row):
    text = unicodedata.normalize('NFKC', row['text']).strip()
    title = unicodedata.normalize('NFKC', row['title']).strip()
    found = []
    # This intentionally quarantines aliases and many legitimate opening styles.
    if not title or normalized(title) not in normalized(text[:60]):
        found.append('subject_not_anchored')
    if re.match(r'^(?:是|为|為|又称|又稱|曾|于|於|他|她|它|其|该|該|此|这|這|并|並|由|在|作为|作為)', text):
        found.append('dependent_opening')
    if re.search(r'(?:包括|如下|分别为|分別為|可分为|可分為|有以下|主要有)\s*[:：]|(?:^|[。;；])\s*(?:[•●▪]|\d+[.)、]|[一二三四五六七八九十]+、)|(?:[^。]{0,30}[、]){3,}', text):
        found.append('list_residue')
    if re.search(r'收入|薪[资資水]|年薪|身家|财产|財產|住址|家庭住址|手机号|手機號|身份证|身份證|未成年|未满\s*\d|未滿\s*\d|性侵|强奸|強姦|猥亵|猥褻|家暴|家庭暴力|裸照|私密|病史|怀孕|懷孕|流产|流產|自杀|自殺|自残|自殘|受害者|受害人|受害女|受害男|儿童虐待|兒童虐待', text):
        found.append('personal_sensitive_risk')
    return found


def excluded_component(rows, prior):
    """Connected component through five-gram Jaccard >= .8; also exact page IDs."""
    sets = [grams(row['text']) for row in rows]
    index = collections.defaultdict(list)
    for i, tokens in enumerate(sets):
        for token in tokens:
            index[token].append(i)
    ids = {str(row.get('page_id', row.get('id', ''))) for row in prior}
    excluded = {i for i, row in enumerate(rows) if str(row['page_id']) in ids}
    frontier = [grams(re.split(r'\n\s*\n', row['text'], maxsplit=1)[0]) for row in prior]
    frontier += [sets[i] for i in excluded]
    while frontier:
        query = frontier.pop()
        if not query:
            continue
        counts = collections.Counter(i for token in query for i in index.get(token, ()))
        for i, common in counts.items():
            if i not in excluded and common / (len(query) + len(sets[i]) - common) >= .8:
                excluded.add(i)
                frontier.append(sets[i])
    return excluded


def rank(seed, row):
    return hashlib.sha256((seed + '\0' + str(row['page_id']) + '\0' + row['group']).encode()).hexdigest()


def load_prior():
    rows, provenance = [], []
    for name, count, expected in PRIOR:
        path = ROOT / name
        data = json.loads(path.read_text(encoding='utf8'))
        if data.get('source_sha256') != expected or len(data.get('samples', [])) != count:
            raise ValueError(f'Invalid prior review: {name}')
        for row in data['samples']:
            if not row.get('page_id', row.get('id')) or not row.get('text'):
                raise ValueError(f'Missing prior page or text: {name}')
        rows.extend(data['samples'])
        provenance.append({'path': name, 'sha256': sha(path), 'rows': count})
    return rows, provenance


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if sha(SOURCE) != EXPECTED:
        raise ValueError('Unexpected train source')
    if sha(LINK_SOURCE) != LINK_SHA:
        raise ValueError('Unexpected train linking source')
    prior, provenance = load_prior()
    rows = [json.loads(line) for line in SOURCE.read_text(encoding='utf8').splitlines()]
    linking_rows = [json.loads(line) for line in LINK_SOURCE.read_text(encoding='utf8').splitlines()]
    if len(rows) != 8150 or any(not all(row.get(k) for k in ('page_id', 'group', 'title', 'url', 'text')) for row in rows):
        raise ValueError('Unexpected source rows')
    linked = excluded_component(linking_rows, prior)
    linked_groups = {linking_rows[i]['group'] for i in linked}
    if not {r['group'] for r in rows} <= {r['group'] for r in linking_rows}:
        raise ValueError('Third version is not a subset of linking rows')
    excluded = {i for i, row in enumerate(rows) if row['group'] in linked_groups}
    tiers = {'screened': [], 'quarantine': []}
    reasons = collections.Counter()
    for i, row in enumerate(rows):
        f = flags(row)
        reasons.update(f)
        tiers['quarantine' if f else 'screened'].append({**row, 'screening_flags': f, 'prior_review_group': i in excluded})
    sample = []
    eligible_counts = {}
    for tier, count in [('screened', 40), ('quarantine', 40)]:
        eligible = [row for row in tiers[tier] if not row['prior_review_group']]
        eligible_counts[tier] = len(eligible)
        if len(eligible) < count:
            raise ValueError(f'Insufficient unseen {tier} rows')
        sample.extend({'tier': tier, **row} for row in sorted(eligible, key=lambda row: rank(SEED, row))[:count])
    OUT.mkdir(parents=False, exist_ok=False)
    for tier, items in tiers.items():
        with (OUT / f'{tier}.jsonl').open('x', encoding='utf8', newline='\n') as f:
            for row in items:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    with (OUT / 'review-sample.json').open('x', encoding='utf8') as f:
        json.dump({'seed': SEED, 'samples': sample}, f, ensure_ascii=False, indent=2)
    manifest = {'status': 'mechanical_screen_only_not_approved', 'source_sha256': EXPECTED, 'link_source_sha256': LINK_SHA,
                'script_sha256': sha(Path(__file__)), 'prior_reviews': provenance,
                'counts': {k: len(v) for k, v in tiers.items()}, 'overlapping_flag_counts': dict(reasons),
                'prior_component_excluded_rows': len(excluded), 'eligible_counts': eligible_counts,
                'sample_counts': {'screened': 40, 'quarantine': 40}, 'seed': SEED,
                'selection': 'sha256(seed + NUL + page_id + NUL + group), ascending within tier',
                'files': {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}}
    with (OUT / 'manifest.json').open('x', encoding='utf8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
