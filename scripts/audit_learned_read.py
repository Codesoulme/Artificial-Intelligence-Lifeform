"""Cost and error audit on 4/8/16 fully occupied neural history pages."""
import argparse,json,statistics,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spora.growing import new_state,grow,observe,query
from spora.checkpoint import load_checkpoint
from spora.memory_read import MemoryReadPlan
from spora.learned_read import train_factors,LearnedReadPlan


def timing(call):
    started=time.perf_counter();value=call();return value,(time.perf_counter()-started)*1000


def audit(seed,plan):
    model,_,_,_=load_checkpoint(ROOT/f'checkpoints/p5b-orthogonal90-seed{seed}.pt')
    state=new_state(model);generator=torch.Generator().manual_seed(plan['audit_seed']+500+seed)
    events=[];rows=[]
    for i in range(1024):
        if i and i%64==0:state=grow(model,state)
        key=int(torch.randint(64,(1,),generator=generator));value=int(torch.randint(16,(1,),generator=generator))
        state=observe(model,state,key,value,1);events.append((key,value))
        if (i+1)//64 not in (4,8,16) or (i+1)%64:continue
        pages=(i+1)//64
        # Both memory sites are exercised by the integration unit tests; this
        # numerical/cost experiment measures the last site's history read.
        matrices=[page[-1][0] for page in state.pages]
        factors,training=train_factors(matrices,plan['router']['rank'],plan['router']['steps'],seed+2415000)
        learned,setup_ms=timing(lambda:LearnedReadPlan(matrices,factors))
        exact,exact_setup_ms=timing(lambda:MemoryReadPlan(matrices))
        q=F.normalize(torch.randn(128,model.config.memory_width,generator=generator),dim=-1)
        keys=torch.randn(pages,model.config.memory_width,generator=generator)*.1
        modes={}
        with torch.no_grad():
            truth,_=exact.read(q,keys)
            for label,call in [('exact',lambda:exact.read(q,keys)),
                              ('p8_bounded',lambda:exact.read(q,keys,'bounded')),
                              ('learned',lambda:learned.read(q,keys))]:
                (value,stats),_=timing(call)
                error=float((value-truth).norm(dim=-1).max())
                modes[label]={'error_l2_max':error,'stats':stats,'samples_ms':[]}
            # Interleave repetitions; include routing and fallback costs.
            for _ in range(15):
                for label,call in [('exact',lambda:exact.read(q,keys)),('p8_bounded',lambda:exact.read(q,keys,'bounded')),('learned',lambda:learned.read(q,keys))]:
                    _,elapsed=timing(call);modes[label]['samples_ms'].append(elapsed)
            for result in modes.values():result['median_ms']=statistics.median(result.pop('samples_ms'))
            # Synthetic highly separated query: proves the certificate can
            # skip work; not counted as natural routing success.
            strong=q[:1];separated=torch.zeros_like(keys);separated[0]=strong[0]*3000
            value,contrast=learned.read(strong,separated)
            reference,_=exact.read(strong,separated)
            contrast['error_l2_max']=float((value-reference).norm(dim=-1).max())
            # Exercise actual occupied memories rather than empty allocation.
            k=torch.tensor([e[0] for e in events]);target=torch.tensor([e[1] for e in events]);t=torch.arange(i+1)
            predictions=[]
            for start in range(0,len(k),128):predictions.append(query(model,state,k[start:start+128],torch.ones_like(k[start:start+128]),t[start:start+128]).argmax(-1))
            history=float((torch.cat(predictions)==target).float().mean())
        row={'seed':seed,'pages':pages,'events':i+1,'history_accuracy':history,'training':training,
             'setup_ms':setup_ms,'exact_setup_ms':exact_setup_ms,'learned_plan_bytes':learned.tensor_bytes,
             'exact_plan_bytes':exact.tensor_bytes,'factor_bytes':sum(t.numel()*t.element_size() for pair in factors for t in pair),
             'modes':modes,'constructed_separation':contrast,
             'speedup':modes['exact']['median_ms']/modes['learned']['median_ms']}
        row['correctness_passed']=max(modes['learned']['error_l2_max'],contrast['error_l2_max'])<=plan['router']['read_error_l2']
        rows.append(row);print(json.dumps(row),flush=True)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);torch.set_num_threads(2)
    plan=json.loads((ROOT/'configs/p12-p13-plan.json').read_text());rows=[]
    for seed in plan['seeds']:rows+=audit(seed,plan)
    result={'cases':rows,'correctness_passed':all(r['correctness_passed'] for r in rows),
            'enable_by_default':all(r['speedup']>1 and r['correctness_passed'] for r in rows),
            'language_training':False,'query_note':'Random numerical address queries; semantic language not trained. Contrast uses intentionally separated page keys.'}
    (output/'audit.json').write_text(json.dumps(result,indent=2))
    if not result['correctness_passed']:raise SystemExit(1)


if __name__=='__main__':main()
