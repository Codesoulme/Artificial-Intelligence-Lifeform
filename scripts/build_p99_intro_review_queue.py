"""Build a fresh pending review queue from the verified P98 input pipeline."""
import argparse
import collections
import json
from pathlib import Path

import build_p98_intro_review_queue as p98
from build_p98_intro_review_queue import *  # Reuse the P98 verification primitives.

ROOT = p98.ROOT
P98_MANIFEST_SHA256 = 'a4b95240e505a29e6d8b6621caade5b75c7c6faf3179940ff8b7962ba068a0b2'
SEED = 'p99-intro-review-20261109-v1'


def load_p98(v1, screened, comp, labels):
    base = ROOT / 'runs/wikimedia-intro-review-p98-v1'
    p98.verify_file(base / 'manifest.json', P98_MANIFEST_SHA256)
    manifest = p98.read_json(base / 'manifest.json')
    p98.require(manifest['status'] == 'pending_not_approved_for_training', 'P98 status mismatch')
    p98.require(manifest['train_v1_sha256'] == p98.V1 and manifest['train_v3_sha256'] == p98.V3 and manifest['p77_screened_sha256'] == p98.SCREEN, 'P98 source mismatch')
    p98.require(manifest['counts']['selected'] == 200 and manifest['counts']['shards'] == 4 and manifest['counts']['per_shard'] == 50, 'P98 counts mismatch')
    index = {str(row['page_id']): i for i, row in enumerate(v1)}
    source = {str(row['page_id']): row for row in screened}
    prior = []
    for n in range(1, 5):
        name = f'review-{n:02}.json'
        p98.verify_file(base / name, manifest['files'][name])
        shard = p98.read_json(base / name)
        p98.require(shard['status'] == 'pending_not_approved_for_training' and shard['source_sha256'] == p98.RAW and shard['train_sha256'] == p98.V3 and shard['screened_sha256'] == p98.SCREEN and len(shard['samples']) == 50, 'P98 shard mismatch')
        for item in shard['samples']:
            pid = str(item['page_id'])
            p98.require(pid in index and pid in source, 'P98 page absent from source')
            row = source[pid]
            p98.require(all(item[k] == row[field] for k, field in [('page_id','page_id'), ('group','group'), ('source_url','url'), ('title','title'), ('text','text')]), 'P98 record mismatch')
            p98.require(item['text_sha256'] == p98.text_hash(row['text']) and item['similarity_component'] == labels[comp[index[pid]]], 'P98 record hash or component mismatch')
            prior.append(item)
    p98.require(len({str(r['page_id']) for r in prior}) == 200 and len({r['similarity_component'] for r in prior}) == 200, 'P98 duplicates')
    return prior


def output_path(name):
    # A single new directory under runs; no traversal, existing files, or overwrites.
    p98.require(name not in ('', '.', '..') and Path(name).name == name and '/' not in name and '\\' not in name and ':' not in name, 'Output must be a directory name under runs')
    out = ROOT / 'runs' / name
    p98.require(not out.exists(), f'Output already exists: {out}')
    return out


def exclude_components(eligible, excluded):
    return [row for row in eligible if row['_component'] not in excluded]


def build(out):
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
    # Stable component identifier is the lexicographically smallest v1 group in each component.
    labels = {}
    for r,c in zip(v1,comp): labels[c] = min(labels.get(c,r['group']),r['group'])
    p98_rows = load_p98(v1, screened, comp, labels)
    p98_comp = {comp[index[str(r['page_id'])]] for r in p98_rows}
    eligible = exclude_components(eligible, p98_comp)
    selected = choose(eligible, 400, SEED)
    out.mkdir(exist_ok=False)
    for n in range(4):
        items = []
        for r in selected[n*100:(n+1)*100]:
            items.append({'page_id':r['page_id'], 'group':r['group'], 'similarity_component':labels[r['_component']], 'text_sha256':text_hash(r['text']), 'source_url':r['url'], 'title':r['title'], 'text':r['text'], 'review_status':'pending', 'review':None})
        with (out/f'review-{n+1:02}.json').open('x',encoding='utf8',newline='\n') as f:
            json.dump({'status':'pending_not_approved_for_training','source_sha256':RAW,'train_sha256':V3,'screened_sha256':SCREEN,'samples':items},f,ensure_ascii=False,indent=2); f.write('\n')
    output = {'status':'pending_not_approved_for_training','seed':SEED,'selection':'ascending sha256(seed + NUL + page_id + NUL + group); first per connected component and page','similarity':'NFKC casefold alphanumeric character 5-gram Jaccard >= 0.8; transitive closure across v1 train; equal page IDs joined','raw_source_sha256':RAW,'source_revision':REV,'source_metadata_sha256':digest(rawpath.parent/'metadata.json'),'p77_manifest_sha256':digest(base/'manifest.json'),'p77_screened_sha256':SCREEN,'p77_review_sample_sha256':SAMPLE,'p98_manifest_sha256':P98_MANIFEST_SHA256,'p98_files':read_json(ROOT/'runs/wikimedia-intro-review-p98-v1/manifest.json')['files'],'train_v1_sha256':V1,'train_v3_sha256':V3,'candidate_manifest_sha256':{str(i):digest(ROOT/f'runs/wikimedia-intro-candidate-v{i}/manifest.json') for i in (1,2,3)},'historical_reviews':provenance,'script_sha256':digest(Path(__file__)),'p98_script_sha256':digest(ROOT/'scripts/build_p98_intro_review_queue.py'),'counts':{'v1_train':len(v1),'v3_train':len(v3),'screened':len(screened),'historical_components':len(prior_comp),'p77_sample_components':len(old_comp),'p98_components':len(p98_comp),'eligible_rows':len(eligible),'eligible_components':len({x['_component'] for x in eligible}),'selected':len(selected),'shards':4,'per_shard':100},'files':{f'review-{n:02}.json':digest(out/f'review-{n:02}.json') for n in range(1,5)}}
    with (out/'manifest.json').open('x',encoding='utf8',newline='\n') as f: json.dump(output,f,ensure_ascii=False,indent=2); f.write('\n')
    print(json.dumps(output,ensure_ascii=False,indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='wikimedia-intro-review-p99-v1', help='new directory name under runs')
    args = parser.parse_args()
    out = output_path(args.output)
    build(out)


if __name__ == '__main__':
    main()
