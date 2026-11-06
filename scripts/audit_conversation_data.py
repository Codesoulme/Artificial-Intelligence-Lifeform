"""Cross-partition exact/group checks and bounded lexical duplicate audit."""
import argparse,json,hashlib,collections
from pathlib import Path
from prepare_conversation_v2 import ROOT,normalized,sha
from curate_conversation import reject

def signatures(text,width=16,stride=16):
    text=normalized(text)
    return {hashlib.blake2b(text[i:i+width].encode(),digest_size=8).digest() for i in range(0,max(0,len(text)-width+1),stride)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='runs/conversation-data-v3')
    parser.add_argument('--output')
    args=parser.parse_args();data=ROOT/args.data;out=ROOT/args.output if args.output else data/'audit.json'
    if out.exists():raise FileExistsError(out)
    groups=collections.defaultdict(set);digests=collections.defaultdict(set);held=[];index=collections.defaultdict(list);counts=collections.Counter();problems=[]
    for split in ('development','test'):
        for kind in ('pretrain','dialogue'):
            with (data/f'{kind}-{split}.jsonl').open(encoding='utf-8') as stream:
                for line in stream:
                    row=json.loads(line);number=len(held);sigs=signatures(row['text'])
                    held.append(dict(kind=kind,split=split,group=row['group'],sha256=row['sha256'],size=len(sigs)))
                    for s in sigs:index[s].append(number)
                    groups[split].add(row['group']);digests[split].add(row['sha256']);counts[kind+'_'+split]+=1
    for kind in ('pretrain','dialogue'):
        with (data/f'{kind}-train.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line);groups['train'].add(row['group']);digests['train'].add(row['sha256']);counts[kind+'_train']+=1
                if reject(row,kind):raise AssertionError('Filter violation')
                sigs=signatures(row['text']);shared=collections.Counter(i for s in sigs for i in index.get(s,()))
                for i,n in shared.items():
                    denominator=min(len(sigs),held[i]['size'])
                    if denominator>=8 and n/denominator>=.8:
                        problems.append(dict(train_sha256=row['sha256'],heldout=held[i],shared=n,ratio=n/denominator))
    cross_groups={f'{a}/{b}':len(groups[a]&groups[b]) for a,b in [('train','development'),('train','test'),('development','test')]}
    cross_exact={f'{a}/{b}':len(digests[a]&digests[b]) for a,b in [('train','development'),('train','test'),('development','test')]}
    report=dict(counts=counts,cross_groups=cross_groups,cross_exact=cross_exact,sampled_shingle_overlaps=problems,
        data_artifact_sha256={f'{kind}-{split}.jsonl':sha(data/f'{kind}-{split}.jsonl') for kind in ('pretrain','dialogue') for split in ('train','development','test')},
        passed=not any(cross_groups.values()) and not any(cross_exact.values()) and not problems,
        scope='All normalized groups/exact digests; 16-character nonoverlapping sampled shingles, >=8 shared and80% containment. Misses shifted/paraphrased near duplicates; not semantic isolation proof.')
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='sampled_shingle_overlaps'}),flush=True)
    print('overlaps',len(problems),flush=True)

if __name__=='__main__':main()
