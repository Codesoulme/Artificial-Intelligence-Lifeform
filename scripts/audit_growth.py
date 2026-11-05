"""Audit one continuing individual across four neural history pages."""
import argparse,json,os,subprocess,sys,tempfile,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spora.checkpoint import load_checkpoint,save_checkpoint
from spora.growing import new_state,grow,observe,query,save,load,scalars,migrate

def isolated(path,keys,modes,times):
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT);env['PYTHONUTF8']='1'
    with tempfile.TemporaryDirectory() as cwd:
        p=subprocess.run([sys.executable,'-m','spora.growing_infer','--checkpoint',str(path.resolve())],
            input=json.dumps({'keys':keys,'modes':modes,'times':times})+'\n',text=True,capture_output=True,env=env,cwd=cwd,timeout=60)
    if p.returncode:raise RuntimeError(p.stderr)
    return json.loads(p.stdout)

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--worlds',type=int,default=8);p.add_argument('--seed',type=int,default=1410000)
    p.add_argument('--entities',type=int,choices=[4,64],default=4);args=p.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);torch.set_num_threads(2);start=time.perf_counter()
    model,_,_,data=load_checkpoint(args.checkpoint)
    gen=torch.Generator().manual_seed(args.seed)
    result={'foundation':str(Path(args.checkpoint).resolve()),'metadata':data['metadata'],'seed':args.seed,'entities':args.entities,'worlds':[]}
    for w in range(args.worlds):
        pool=torch.randperm(64,generator=gen)[:args.entities];keys=pool[torch.randint(args.entities,(256,),generator=gen)]
        values=torch.randint(16,(256,),generator=gen);trusted=(torch.rand(256,generator=gen)>.25).float()
        keys[:4]=pool[:4];trusted[:4]=1
        state=new_state(model);current={};history_keys=[];history_values=[];stages=[]
        for index in range(256):
            if index and index%64==0:
                k=torch.tensor(history_keys);m=torch.ones_like(k);t=torch.arange(index)
                before=query(model,state,k,m,t);state=grow(model,state)
                after=query(model,state,k,m,t)
                if not torch.equal(before,after):raise AssertionError('Growth changed existing query logits')
            state=observe(model,state,int(keys[index]),int(values[index]),float(trusted[index]))
            history_keys.append(int(keys[index]));history_values.append(int(values[index]))
            if trusted[index]:current[int(keys[index])]=int(values[index])
            if (index+1)%64==0:
                k=torch.tensor(history_keys);m=torch.ones_like(k);t=torch.arange(index+1);target=torch.tensor(history_values)
                prediction=query(model,state,k,m,t).argmax(-1)
                wrong=query(model,state,(k+19)%64,m,t).argmax(-1)
                ck=torch.tensor(list(current));cv=torch.tensor(list(current.values()))
                cp=query(model,state,ck,torch.zeros_like(ck),torch.zeros_like(ck)).argmax(-1)
                unknown_keys=torch.tensor([k for k in range(64) if k not in current],dtype=torch.long)
                up=query(model,state,unknown_keys,torch.zeros_like(unknown_keys),torch.zeros_like(unknown_keys)).argmax(-1) if unknown_keys.numel() else None
                stages.append({'events':index+1,'history':float((prediction==target).float().mean()),
                    'oldest64':float((prediction[:64]==target[:64]).float().mean()),'invalid_unknown':float((wrong==16).float().mean()),
                    'current':float((cp==cv).float().mean()),'current_unknown':float((up==16).float().mean()) if up is not None else None,'scalars':scalars(state)})
                # Resume the same individual between every page, not a new random model.
                path=output/f'world{w}.pt';save(path,model,state,{'stage':'P6 growth audit','world':w})
                model,state,_=load(path)
        cold=isolated(path,history_keys[:8]+history_keys[-8:],[1]*16,list(range(8))+list(range(248,256)))
        with torch.no_grad():expected=query(model,state,torch.tensor(history_keys[:8]+history_keys[-8:]),torch.ones(16,dtype=torch.long),torch.tensor(list(range(8))+list(range(248,256)))).softmax(-1).tolist()
        result['worlds'].append({'world':w,'stages':stages,'cold_exact':cold['probabilities']==expected})
        print(json.dumps({'world':w,'stages':stages,'cold_exact':cold['probabilities']==expected}),flush=True)
    result['passed']=all(x['cold_exact'] and all(min(s[k] for k in ('history','oldest64','invalid_unknown','current','current_unknown') if s[k] is not None)>=.95 for s in x['stages']) for x in result['worlds'])
    result['seconds']=time.perf_counter()-start
    (output/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({'passed':result['passed'],'seconds':result['seconds']}),flush=True)

if __name__=='__main__':main()
