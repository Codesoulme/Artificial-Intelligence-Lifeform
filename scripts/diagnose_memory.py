"""Transient development probes; never save or alter a model checkpoint."""
from pathlib import Path
import hashlib
import json
import torch
from torch.nn import functional as F
from spora.checkpoint import load_checkpoint
from spora.model import fast_forward
from spora.tasks import mappings, absorb


def main():
    root=Path(__file__).resolve().parents[1]
    destination=root/'reports'/'memory-diagnosis-2026-11-05.json'
    if destination.exists():
        raise FileExistsError('Preserve observed diagnostics; choose a new version for a new study')
    torch.set_num_threads(4)
    seed_results=[]
    for seed in (7,11,23):
        path=root/f'checkpoints/p1-seed{seed}.pt'
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        model,_,_,_=load_checkpoint(path)
        generator=torch.Generator().manual_seed(400001)
        scores={(steps,reset):{'corrected':0,'unrelated':0} for steps in (1,2,4,8) for reset in (False,True)}
        norms=[[] for _ in model.memories]
        worlds=128
        for offset in range(0,worlds,16):
            keys,values=mappings(generator,16,8,model.config,'cpu')
            fast,velocity=absorb(model,keys,values)
            revised=(values[:,0]+1)%model.config.value_count
            for site_index,site in enumerate(model.memories):
                key=site.address(model.embedding(keys[:,0]+model.config.key_start))
                target=site.target(model.embedding(revised+model.config.value_start))
                loss=.5*(fast_forward(fast[site_index],key)-target.detach()).square().sum()
                gradients=torch.autograd.grad(loss,fast[site_index])
                norm=torch.sqrt(sum(g.flatten(1).square().sum(1) for g in gradients)+1e-12)
                norms[site_index].extend(norm.detach().tolist())
            for reset in (False,True):
                ff=[[p.detach().clone().requires_grad_(True) for p in site] for site in fast]
                uu=[[torch.zeros_like(p) if reset else p.detach().clone() for p in site] for site in velocity]
                for step in range(1,9):
                    ff,uu=model.write_event(ff,uu,keys[:,0],revised)
                    if (step,reset) in scores:
                        with torch.no_grad():
                            predicted=model.predict_many(keys,ff).argmax(-1)
                        scores[step,reset]['corrected']+=(predicted[:,0]==revised).sum().item()
                        scores[step,reset]['unrelated']+=(predicted[:,1:]==values[:,1:]).sum().item()
        diagnostics=[]
        for index,site in enumerate(model.memories):
            n=torch.tensor(norms[index])
            q=site.address(model.embedding(torch.arange(model.config.key_count)+model.config.key_start))
            q=F.normalize(q,dim=-1)
            gram=q@q.T
            offdiag=gram[~torch.eye(model.config.key_count,dtype=torch.bool)]
            diagnostics.append({'site':index,'learned_rate':float(F.softplus(site.log_lr).detach()),
                'mean_correction_gradient_norm':float(n.mean()),'fraction_clipped':float((n>site.clip_norm).float().mean()),
                'mean_clip_scale':float((site.clip_norm/n).clamp(max=1).mean()),
                'mean_abs_key_cosine':float(offdiag.abs().mean().detach()),
                'max_abs_key_cosine':float(offdiag.abs().max().detach())})
        assert before==hashlib.sha256(path.read_bytes()).hexdigest()
        seed_results.append({'seed':seed,'checkpoint_sha256':before,'worlds':worlds,
            'results':[{'gradient_steps_per_event':steps,'reset_initial_momentum':reset,
                        'correction_accuracy':score['corrected']/worlds,
                        'unrelated_accuracy':score['unrelated']/(worlds*7)}
                       for (steps,reset),score in scores.items()],
            'diagnostics':diagnostics})
    output={'date':'2026-11-05','purpose':'development diagnosis, not a sealed final evaluation',
        'generator_seed':400001,'chosen_after':'32-world seed7 exploratory probe at seed400000',
        'same_observation_reused':True,'no_external_teacher':True,'no_checkpoint_writes':True,
        'caveats':['More inner steps consume additional compute; not an equal-compute comparison',
                   'Results only cover 8 symbolic associations and correction of the first',
                   'Not evidence that capacity saturation or general correction is solved'],
        'training_seeds':seed_results}
    destination.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    for result in seed_results:
        print(json.dumps({'seed':result['seed'],'results':result['results'],'diagnostics':result['diagnostics']}),flush=True)
    print('Saved diagnostic report:',destination)


if __name__=='__main__':
    main()
