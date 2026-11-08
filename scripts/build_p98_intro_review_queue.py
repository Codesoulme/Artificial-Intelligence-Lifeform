"""Create a reproducible local review queue; never approves use in training."""
import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/wikimedia-intro-review-p98-v1'
SEED = 'p98-intro-review-20261109-v1'
RAW = 'ec8f6c0dd1418b8fcd454278fb4f2bc7cd0ce8312f29c80b672533942601338f'
REV = 'b04c8d1ceb2f5cd4588862100d08de323dccfbaa'
V1 = 'ac528a753ace33f9567e57359df86671a4fa80de79954ccb31e2b03d9f04fc7f'
V3 = '2312663f7ace88aebb704171c5dc306e5b86958105576717a5cf297b28eebfdc'
SCREEN = 'f2a30945dab18b728666bd151a179b720f2ae3a1464a337548a384cdba01879f'
SAMPLE = '4694eaafdcd030b88c74ed9d37cf155140bfc2fa62f6a8e2e0934bac71084c9e'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def norm(text):
    return re.sub(r'[^\w]+', '', unicodedata.normalize('NFKC', text).casefold())


def grams(text):
    s = norm(text)
    return {s[i:i+5] for i in range(len(s)-4)}


def components(rows):
    """Connected components of train rows under page equality or 5-gram Jaccard >= .8."""
    sets = [grams(r['text']) for r in rows]
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a,b)] = min(a,b)
    inverted = collections.defaultdict(list)
    pages = {}
    for i, (row, tokens) in enumerate(zip(rows, sets)):
        pid = str(row['page_id'])
        if pid in pages: union(i, pages[pid])
        pages[pid] = i
        counts = collections.Counter(j for token in tokens for j in inverted[token])
        for j, common in counts.items():
            if common / (len(tokens) + len(sets[j]) - common) >= .8:
                union(i, j)
        for token in tokens: inverted[token].append(i)
    return sets, [find(i) for i in range(len(rows))]


def related_indices(rows, sets, historical):
    """Link historical first paragraphs directly to the v1 train graph."""
    inv = collections.defaultdict(list)
    by_page = collections.defaultdict(list)
    for i, (r, tokens) in enumerate(zip(rows, sets)):
        by_page[str(r['page_id'])].append(i)
        for t in tokens: inv[t].append(i)
    linked = set()
    for r in historical:
        linked.update(by_page.get(str(r.get('page_id', r.get('id'))), []))
        tokens = grams(re.split(r'\n\s*\n', r['text'], maxsplit=1)[0])
        counts = collections.Counter(i for token in tokens for i in inv.get(token, ()))
        for i, common in counts.items():
            if common / (len(tokens) + len(sets[i]) - common) >= .8:
                linked.add(i)
    return linked


def read_json(path):
    return json.loads(path.read_text(encoding='utf8'))


def rows(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]


def require(condition, msg):
    if not condition: raise ValueError(msg)


def verify_file(path, expected):
    actual = digest(path)
    require(actual == expected, f'Checksum mismatch: {path}')
    return actual


def choose(candidates, count=200, seed=SEED):
    def rank(r):
        return text_hash(seed + '\0' + str(r['page_id']) + '\0' + r['group'])
    ordered = sorted(candidates, key=lambda r: (rank(r), str(r['page_id']), r['group']))
    chosen, seen_groups, seen_pages = [], set(), set()
    for r in ordered:
        if r['_component'] in seen_groups or str(r['page_id']) in seen_pages: continue
        chosen.append(r)
        seen_groups.add(r['_component']); seen_pages.add(str(r['page_id']))
        if len(chosen) == count: return chosen
    raise ValueError(f'Only {len(chosen)} distinct page/components available')


def main():
    require(not OUT.exists(), f'Output already exists: {OUT}')
    base = ROOT / 'runs/wikimedia-intro-screening-p77-v2'
    pm = read_json(base/'manifest.json')
    require(pm['status'] == 'mechanical_screen_only_not_approved', 'Unexpected P77 status')
    require(pm['source_sha256'] == V3 and pm['link_source_sha256'] == V1, 'P77 source mismatch')
    require(pm['files']['screened.jsonl'] == SCREEN and pm['files']['review-sample.json'] == SAMPLE, 'P77 output mismatch')
    verify_file(base/'screened.jsonl', SCREEN); verify_file(base/'review-sample.json', SAMPLE)
    verify_file(ROOT/'scripts/stratify_intro_train.py', pm['script_sha256'])
    rawpath = ROOT/'runs/research/wikimedia-wikipedia-zh/shard-00002.parquet'
    verify_file(rawpath, RAW)  # opaque bytes only: never decode the mixed source
    metadata = read_json(rawpath.parent/'metadata.json')
    md = [x for x in metadata if x['path'] == '20231101.zh/train-00002-of-00006.parquet']
    require(len(md) == 1 and md[0]['lfs']['oid'] == RAW and md[0]['size'] == rawpath.stat().st_size, 'Source metadata mismatch')
    manifests = {}
    for version, expected in [(1,V1),(2,'addf534bdad82afb96f62af75fe476230f7519234938e824eef66827b3cec69c'),(3,V3)]:
        d = ROOT/f'runs/wikimedia-intro-candidate-v{version}'
        m = read_json(d/'manifest.json'); manifests[version] = m
        require(m['status'] == 'candidate_not_approved_for_training' and m['source_sha256'] == RAW and m['revision'] == REV and m['source_repository'] == 'wikimedia/wikipedia' and m['files']['train.jsonl'] == expected, f'Invalid v{version} manifest')
        verify_file(d/'train.jsonl', expected)
        verify_file((d/'generator.py') if version != 3 else (ROOT/'scripts/prepare_wikimedia_intro_candidate.py'), m['generator_sha256'])
    v1 = rows(ROOT/'runs/wikimedia-intro-candidate-v1/train.jsonl')
    v2 = rows(ROOT/'runs/wikimedia-intro-candidate-v2/train.jsonl')
    v3 = rows(ROOT/'runs/wikimedia-intro-candidate-v3/train.jsonl')
    for ver, dataset in [(1,v1),(2,v2),(3,v3)]:
        require(len(dataset) == manifests[ver]['counts']['train_rows'], 'Train count mismatch')
        require(len({str(r['page_id']) for r in dataset}) == len(dataset), 'Duplicate page')
        for r in dataset:
            require(all(r.get(k) for k in ('group','page_id','url','title','text')), 'Missing source field')
            require(text_hash(norm(r['text'])) == r['group'] and int(r['group'][:8],16)%20 != 0, 'Group or split mismatch')
            u = urlparse(r['url'])
            require(u.scheme == 'https' and u.netloc == 'zh.wikipedia.org' and u.path.startswith('/wiki/'), 'Unexpected page URL')
    def as_map(dataset): return {str(r['page_id']):r for r in dataset}
    m1,m2,m3 = map(as_map,(v1,v2,v3))
    require(all(m1.get(k)==v for k,v in m2.items()) and all(m2.get(k)==v for k,v in m3.items()), 'Train versions not nested')
    prior = []
    provenance = []
    for p in pm['prior_reviews']:
        path = ROOT/p['path']; verify_file(path,p['sha256']); obj = read_json(path)
        require(len(obj['samples']) == p['rows'], 'Historical sample count mismatch')
        require(obj['source_sha256'] in (RAW,V1,V3,manifests[2]['files']['train.jsonl']), 'Historical source mismatch')
        prior.extend(obj['samples']); provenance.append(p)
    oldsample = read_json(base/'review-sample.json')
    require(len(oldsample['samples']) == 80 and collections.Counter(x['tier'] for x in oldsample['samples']) == {'screened':40,'quarantine':40}, 'P77 sample mismatch')
    require(all({k:v for k,v in x.items() if k not in ('tier','screening_flags','prior_review_group')} == m3.get(str(x['page_id'])) for x in oldsample['samples']), 'P77 sample source mismatch')
    sets, comp = components(v1)
    index = {str(r['page_id']):i for i,r in enumerate(v1)}
    prior_comp = {comp[i] for i in related_indices(v1, sets, prior)}
    old_comp = {comp[i] for i in related_indices(v1, sets, oldsample['samples'])}
    historical_pages = {str(x.get('page_id',x.get('id'))) for x in prior + oldsample['samples']}
    # Re-run the versioned P77 rules and compare every train row to its saved tier.
    from stratify_intro_train import flags
    screened = rows(base/'screened.jsonl')
    require(len(screened) == pm['counts']['screened'], 'Screen count mismatch')
    saved = {str(r['page_id']):r for r in screened}
    require(len(saved) == len(screened), 'Duplicate screened page')
    for r in v3:
        pid = str(r['page_id']); expected = not flags(r)
        require((pid in saved) == expected, 'Screen classification mismatch')
        if expected:
            s = saved[pid]
            require({k:v for k,v in s.items() if k not in ('screening_flags','prior_review_group')} == r and s['screening_flags'] == [], 'Screen record mismatch')
            require(s['prior_review_group'] == (comp[index[pid]] in prior_comp), 'Historical exclusion mismatch')
    eligible=[]
    for r in screened:
        pid=str(r['page_id']); c=comp[index[pid]]
        if c not in prior_comp and c not in old_comp and pid not in historical_pages:
            eligible.append({**r, '_component':c})
    selected = choose(eligible)
    # Stable component identifier is the lexicographically smallest v1 group in each component.
    labels = {}
    for r,c in zip(v1,comp): labels[c] = min(labels.get(c,r['group']),r['group'])
    OUT.mkdir(exist_ok=False)
    for n in range(4):
        items=[]
        for r in selected[n*50:(n+1)*50]:
            items.append({'page_id':r['page_id'], 'group':r['group'], 'similarity_component':labels[r['_component']], 'text_sha256':text_hash(r['text']), 'source_url':r['url'], 'title':r['title'], 'text':r['text'], 'review_status':'pending', 'review':None})
        with (OUT/f'review-{n+1:02}.json').open('x',encoding='utf8',newline='\n') as f:
            json.dump({'status':'pending_not_approved_for_training','source_sha256':RAW,'train_sha256':V3,'screened_sha256':SCREEN,'samples':items},f,ensure_ascii=False,indent=2); f.write('\n')
    output = {'status':'pending_not_approved_for_training','seed':SEED,'selection':'ascending sha256(seed + NUL + page_id + NUL + group); first per connected component and page','similarity':'NFKC casefold alphanumeric character 5-gram Jaccard >= 0.8; transitive closure across v1 train; equal page IDs joined','raw_source_sha256':RAW,'source_revision':REV,'source_metadata_sha256':digest(rawpath.parent/'metadata.json'),'p77_manifest_sha256':digest(base/'manifest.json'),'p77_screened_sha256':SCREEN,'p77_review_sample_sha256':SAMPLE,'train_v1_sha256':V1,'train_v3_sha256':V3,'candidate_manifest_sha256':{str(i):digest(ROOT/f'runs/wikimedia-intro-candidate-v{i}/manifest.json') for i in (1,2,3)},'historical_reviews':provenance,'script_sha256':digest(Path(__file__)),'counts':{'v1_train':len(v1),'v3_train':len(v3),'screened':len(screened),'historical_components':len(prior_comp),'p77_sample_components':len(old_comp),'eligible_rows':len(eligible),'eligible_components':len({x['_component'] for x in eligible}),'selected':len(selected),'shards':4,'per_shard':50},'files':{f'review-{n:02}.json':digest(OUT/f'review-{n:02}.json') for n in range(1,5)}}
    with (OUT/'manifest.json').open('x',encoding='utf8',newline='\n') as f: json.dump(output,f,ensure_ascii=False,indent=2); f.write('\n')
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
