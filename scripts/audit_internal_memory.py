"""Held-out numerical policy and continuing provenance memory audits."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spora.internal_memory import WriteControl,attach,observe_sourced,read_provenance,consolidation_request,SOURCES
from spora.growing import new_state,grow,query,save,load
from spora.checkpoint import load_checkpoint
from spora.state_codec import flatten
from spora.durable import TensorStore
from spora.sleep_memory import consolidate_store
from train_write_control import evaluate
from audit_durable import fingerprint


def controller(seed):
    path=ROOT/f'runs/p12-v2-seed{seed}/model.pt'
    d=torch.load(path,map_location='cpu',weights_only=True)
    m=WriteControl(**d['config']);m.load_state_dict(d['state']);return m.eval(),d['training']


@torch.no_grad()
def memory_audit(seed,world,output,audit_seed=2440000):
    model,_,_,_=load_checkpoint(ROOT/f'checkpoints/p5b-orthogonal90-seed{seed}.pt')
    gate,_=controller(seed)
    state=attach(model,new_state(model),gate)
    generator=torch.Generator().manual_seed(audit_seed+world)
    values=torch.randint(16,(64,),generator=generator).tolist()
    current={};events=[];decision_ok=[];blocked_exact=[];sleep=None;qualities=[];effective_times=[]
    for i in range(256):
        key=i%64;phase=i//64;kind=key%8;original=values[key]
        source='direct';quality=.8 if kind==6 else .95;effective=i
        if phase==0:
            value=original;action='new'
        elif phase==1:
            if kind==0:value=(original+1)%16;quality=1.;action='correct'
            elif kind==1:value=(original+2)%16;source='report';quality=.2;action='defer'
            elif kind==2:value=(original+3)%16;source='inference';quality=1.;action='defer'
            elif kind==3:value=(original+4)%16;source='simulation';quality=1.;action='defer'
            elif kind==4:value=original;action='repeat'
            elif kind==5:value=(original+5)%16;quality=1.;effective=0;action='defer'
            elif kind==6:value=original;quality=1.;action='support'
            else:value=(original+6)%16;source='report';quality=.99;action='correct'
        elif phase==2:
            value=(original+7)%16;source='report';quality=.1;action='defer'
        else:
            value=(original+8)%16;quality=1.;action='correct'
        if i and i%64==0:state=grow(model,state)
        prior=state
        state,result=observe_sourced(model,state,key,value,source,quality,effective)
        decision_ok.append(result['action']==action)
        if action in ('new','correct'):current[key]=value
        if action in ('defer','repeat','support'):
            blocked_exact.append(all(torch.equal(a,b) for a,b in zip(prior.current,state.current)))
        events.append((key,value,source,action))
        qualities.append(quality);effective_times.append(effective)
        if i==63:
            path=output/f'seed{seed}-world{world}.sqlite'
            with TensorStore.create(path,model,state) as store:
                request=consolidation_request(model,state)
                sleep=consolidate_store(store,model,state,store.revision,request['pages'][0],steps=600,seed=2439000+world)
                model,state,_,_=store.read()
        if (i+1)%64==0:
            path=output/f'seed{seed}-world{world}.pt'
            save(path,model,state)
            old=fingerprint(model,state);model,state,_=load(path)
            if fingerprint(model,state)!=old:raise AssertionError('Restart mismatch')
    keys=torch.tensor([e[0] for e in events]);times=torch.arange(256)
    history=query(model,state,keys,torch.ones_like(keys),times).argmax(-1)
    unknown=query(model,state,(keys+1)%64,torch.ones_like(keys),times).argmax(-1)
    current_keys=torch.arange(64)
    actual=query(model,state,current_keys,torch.zeros(64,dtype=torch.long),torch.zeros(64,dtype=torch.long)).argmax(-1)
    sources=[read_provenance(model,state,key,i) for i,(key,_,_,_) in enumerate(events)]
    result={'seed':seed,'world':world,'events':len(events),
        'decision_accuracy':sum(decision_ok)/len(decision_ok),
        'current_accuracy':float((actual==torch.tensor([current[k] for k in range(64)])).float().mean()),
        'history_accuracy':float((history==torch.tensor([e[1] for e in events])).float().mean()),
        'wrong_key_unknown':float((unknown==16).float().mean()),
        'source_accuracy':sum(r.get('source')==e[2] for r,e in zip(sources,events))/len(events),
        'stored_decision_accuracy':sum(r.get('action')==e[3] for r,e in zip(sources,events))/len(events),
        'quality_mae':sum(abs((r.get('quality') or 0)-q) for r,q in zip(sources,qualities))/len(events),
        'quality_max_error':max(abs((r.get('quality') or 0)-q) for r,q in zip(sources,qualities)),
        'effective_time_max_error':max(abs((r.get('valid_at') if r.get('valid_at') is not None else -1024)-t) for r,t in zip(sources,effective_times)),
        'nonwriting_current_exact':all(blocked_exact),'sleep':sleep,'restart_exact':True}
    result['passed']=all(result[x]>=.95 for x in ('decision_accuracy','current_accuracy','history_accuracy','wrong_key_unknown','source_accuracy','stored_decision_accuracy')) and all(blocked_exact) and sleep['accepted']
    result['passed']=result['passed'] and result['effective_time_max_error']==0 and result['quality_max_error']<=.5/255+1e-6
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);torch.set_num_threads(2)
    plan=json.loads((ROOT/'configs/p12-p13-plan.json').read_text())
    revision=json.loads((ROOT/'configs/p12-provenance-v2.json').read_text())
    plan=dict(plan,audit_seed=revision['audit_seed'],provenance_revision=revision)
    report={'plan':plan,'classifiers':[],'memory':[]}
    for seed in plan['seeds']:
        m,training=controller(seed);metrics=evaluate(m,plan['audit_seed'])
        passed=min(metrics['guarded_accuracy'].values())>=plan['write_accuracy_each_class_min'] and metrics['nonfactual_writes']==0
        row={'seed':seed,'training':training,'audit':metrics,'passed':passed};report['classifiers'].append(row);print(json.dumps(row),flush=True)
        for world in range(2):
            row=memory_audit(seed,world,output,plan['audit_seed']);report['memory'].append(row);print(json.dumps(row),flush=True)
    report['passed']=all(x['passed'] for x in report['classifiers']+report['memory'])
    (output/'audit.json').write_text(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
