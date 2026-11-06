"""Check new partitions against previously learned text and vocabulary inputs."""
import argparse,collections,json
from pathlib import Path
from audit_conversation_data import signatures
from prepare_conversation_v2 import ROOT,sha

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='runs/conversation-data-v4')
    args=parser.parse_args();current=ROOT/args.data;out=current/'lineage-audit.json'
    if out.exists():raise FileExistsError(out)
    train_dirs=[ROOT/'runs/conversation-data-v2',current]
    held_dirs=[ROOT/'runs/conversation-data-v3',current]
    manifests={str(p.relative_to(ROOT)):sha(p/'manifest.json') for p in set(train_dirs+held_dirs)}
    held=[];index=collections.defaultdict(list);groups={};digests={};issues=[];counts=collections.Counter();seen=set()
    for directory in held_dirs:
        for split in ('development','test'):
            for kind in ('pretrain','dialogue'):
                with (directory/f'{kind}-{split}.jsonl').open(encoding='utf-8') as stream:
                    for line in stream:
                        row=json.loads(line);sigs=signatures(row['text']);n=len(held)
                        item=dict(directory=directory.name,split=split,kind=kind,group=row['group'],sha256=row['sha256'],size=len(sigs))
                        held.append(item);groups.setdefault(row['group'],[]).append(n);digests.setdefault(row['sha256'],[]).append(n)
                        for s in sigs:index[s].append(n)
    for directory in train_dirs:
        for kind in ('pretrain','dialogue'):
            with (directory/f'{kind}-train.jsonl').open(encoding='utf-8') as stream:
                for line in stream:
                    row=json.loads(line);counts[directory.name+'_'+kind]+=1
                    if row['sha256'] in seen:continue
                    seen.add(row['sha256']);sigs=signatures(row['text'])
                    candidates=collections.Counter(i for s in sigs for i in index.get(s,()))
                    exact=set(groups.get(row['group'],[]))|set(digests.get(row['sha256'],[]))
                    for i,n in candidates.items():
                        denominator=min(len(sigs),held[i]['size'])
                        if denominator>=8 and n/denominator>=.8:exact.add(i)
                    for i in sorted(exact):issues.append(dict(train_directory=directory.name,train_kind=kind,train_sha256=row['sha256'],heldout=held[i],shared_shingles=candidates[i]))
            print(json.dumps(dict(checked=directory.name,kind=kind,issues=len(issues))),flush=True)
    result=dict(passed=not issues,counts=counts,unique_train_documents=len(seen),manifest_sha256=manifests,issues=issues,
        scope='Checks v2 TRAIN (covers vocabulary fitting and B-training superset) plus current TRAIN against v3 and current held partitions. Exact/group and sampled16-character shingles>=80%containment over>=8 shingles. Does not prove semantic isolation. No model outputs used.')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],issues=len(issues),counts=counts)),flush=True)

if __name__=='__main__':main()
