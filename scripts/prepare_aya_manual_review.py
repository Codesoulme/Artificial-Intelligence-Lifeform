"""Create fixed, disjoint local review shards; not approval or training."""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'runs/aya-chinese-candidate-v1'
OUT = ROOT/'runs/aya-review-p83'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    source = DATA/'train.jsonl'
    manifest = json.loads((DATA/'manifest.json').read_text(encoding='utf8'))
    if sha(source) != manifest['files'][source.name]:
        raise ValueError('Candidate changed')
    rows = [json.loads(x) for x in source.read_text(encoding='utf8').splitlines()]
    # P73's 40 reviewed rows are intentionally excluded from this new sample.
    previous = set(random.Random(741109).sample(range(len(rows)),40))
    previous_groups = {rows[i]['group'] for i in previous}
    by_group = {}
    for index, row in enumerate(rows):
        if row['group'] not in previous_groups:
            by_group.setdefault(row['group'], (index,row))
    selected = random.Random(831109).sample(sorted(by_group),240)
    OUT.mkdir(parents=True)
    for shard in range(4):
        path = OUT/f'shard-{shard+1}.jsonl'
        with path.open('x',encoding='utf8') as f:
            for group in selected[shard*60:(shard+1)*60]:
                index,row = by_group[group]
                canonical = json.dumps(row['messages'],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf8')
                item = dict(source_index=index,group=group,messages_sha256=hashlib.sha256(canonical).hexdigest(),
                            messages=row['messages'])
                f.write(json.dumps(item,ensure_ascii=False)+'\n')
    report = dict(status='manual_review_pending_not_approved_for_training',seed=831109,
                  source_sha256=sha(source),candidate_manifest_sha256=sha(DATA/'manifest.json'),
                  generator_sha256=sha(Path(__file__)),selected_groups=240,shard_count=4,rows_per_shard=60,
                  excluded_previous_groups=len(previous_groups),
                  files={p.name:sha(p) for p in OUT.glob('shard-*.jsonl')},
                  scope='Train only; one row per normalized prompt group. Four independent AI reviewers; no human or factual verification claimed.')
    (OUT/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
