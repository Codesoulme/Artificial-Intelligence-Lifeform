"""Account for sealed-state savings and the cost of materializing chosen pages."""
import json,sys,time,statistics
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spora.growing import load,save,query,scalars,GrowingState
from spora.sleep_memory import history_site
from spora.language_bridge import LanguageBridge
from spora.durable import TensorStore


def timing(functions,device,repeats=21):
    for _ in range(3):
        for f in functions.values():f()
    times={name:[] for name in functions};names=list(functions)
    for i in range(repeats):
        order=names[i%len(names):]+names[:i%len(names)]
        for name in order:
            if device=='cuda':torch.cuda.synchronize()
            started=time.perf_counter();functions[name]()
            if device=='cuda':torch.cuda.synchronize()
            times[name].append((time.perf_counter()-started)*1000)
    return {n:{'median_ms':statistics.median(v),'samples':v} for n,v in times.items()}


@torch.no_grad()
def main():
    output=ROOT/'runs/p10-sleep-profile'
    if output.exists():raise FileExistsError(output)
    output.mkdir();torch.set_num_threads(2);torch.manual_seed(2230000)
    folder=ROOT/'runs/p10-sleep-audit/seed7-world0'
    result={'rows':[],'files':{},'training_steps':0,
            'limitations':['reconstructing a selected low-rank page costs compute',
                           'byte interface eagerly materializes sealed pages; no semantic language claim']}
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        model,sealed,_=load(folder/'after-stage2.pt',device)
        dense_pages=[]
        for page_index in range(3):
            _,before,_=load(folder/f'before-stage{page_index}.pt',device)
            dense_pages.append(before.pages[page_index])
        dense_pages.append(sealed.pages[3])
        dense=GrowingState(sealed.current,dense_pages,sealed.cursor)
        expanded=GrowingState(sealed.current,[[history_site(page,site) for site in range(len(model.memories))]
                                              for page in sealed.pages],sealed.cursor)
        if device=='cpu':
            dense_path=output/'dense.pt';sealed_path=output/'sealed.pt'
            save(dense_path,model,dense);save(sealed_path,model,sealed)
            result['files']={'dense_bytes':dense_path.stat().st_size,'sealed_bytes':sealed_path.stat().st_size,
                             'dense_scalars':scalars(dense),'sealed_scalars':scalars(sealed)}
            for name,state in [('dense',dense),('sealed',sealed)]:
                with TensorStore.create(output/f'{name}.sqlite',model,state) as store:store.compact()
                result['files'][name+'_sqlite_bytes']=(output/f'{name}.sqlite').stat().st_size
        for count in (1,64):
            keys=torch.randint(64,(count,),device=device);modes=torch.ones_like(keys)
            times=torch.randint(192,(count,),device=device)
            expected=query(model,expanded,keys,modes,times)
            actual=query(model,sealed,keys,modes,times)
            assert torch.equal(actual,expected)
            row={'device':device,'query_count':count,'materialized_equivalent_exact':True,
                 'timing':timing({'original_dense':lambda:query(model,dense,keys,modes,times),
                                  'sealed':lambda:query(model,sealed,keys,modes,times)},device)}
            result['rows'].append(row)
        bridge=LanguageBridge(model).to(device);tokens=torch.randint(0,256,(4,128),device=device)
        expected=bridge(tokens,expanded);actual=bridge(tokens,sealed)
        result['rows'].append({'device':device,'query_count':'untrained byte interface 4x128',
            'max_abs_logit_error':float((expected-actual).abs().max()),
            'timing':timing({'expanded_same_memory':lambda:bridge(tokens,expanded),
                             'sealed':lambda:bridge(tokens,sealed)},device,repeats=9)})
    result['passed']=all(r.get('materialized_equivalent_exact',r.get('max_abs_logit_error',1)<1e-4) for r in result['rows'])
    (output/'profile.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
