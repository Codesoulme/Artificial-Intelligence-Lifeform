"""Prepare a random byte interface around an audited organism; no text training."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spora.growing import load,query,scalars
from spora.language_bridge import LanguageBridge,causal_loss
from spora.language_training import load_interface,ByteTrainer

def main():
    p=argparse.ArgumentParser();p.add_argument('--individual',required=True);p.add_argument('--output',required=True);p.add_argument('--context-layers',type=int,default=0);args=p.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(1710000)
    source=Path(args.individual);before=hashlib.sha256(source.read_bytes()).hexdigest()
    organism,state,meta=load(source);bridge=LanguageBridge(organism,context_layers=args.context_layers)
    # Synthetic integer tokens only: readiness for a training step, no Chinese examples.
    tokens=torch.randint(0,256,(2,128));prefix=tokens.clone();prefix[:,64:]=(prefix[:,64:]+1)%256
    with torch.no_grad():
        a=bridge(tokens,state);b=bridge(prefix,state)
        causal=torch.equal(a[:,:64],b[:,:64])
    start=time.perf_counter();loss=causal_loss(bridge(tokens,state),tokens);loss.backward()
    gradient=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in bridge.trainable_parameters())
    organism_frozen=all(p.grad is None for p in organism.parameters())
    bridge.zero_grad(set_to_none=True)
    # Save the pristine random interface; smoke test performed no optimizer step.
    interface={k:v.detach().cpu().clone() for k,v in bridge.state_dict().items() if not k.startswith('organism.')}
    interface_format='spora-byte-interface-v2' if args.context_layers else 'spora-byte-interface-v1'
    torch.save({'format':interface_format,'context_layers':args.context_layers,'seed':1710000,'organism_sha256':before,'state':interface,
                'trained_tokens':0,'vocab':'UTF-8 bytes + BOS/EOS/PAD'},output/'untrained-interface.pt')
    restored=LanguageBridge(organism,context_layers=args.context_layers)
    payload=torch.load(output/'untrained-interface.pt',weights_only=True)
    load_interface(restored,payload,before)
    with torch.no_grad():same=torch.equal(a,restored(tokens,state))
    report={'trained_language_tokens':0,'Chinese_training_started':False,'causal_prefix_exact':causal,
            'trainable_gradients_finite':gradient,'organism_frozen':organism_frozen,'interface_restart_exact':same,
            'organism_file_unchanged':before==hashlib.sha256(source.read_bytes()).hexdigest(),
            'organism_sha256':before,'interface_parameters':sum(p.numel() for p in bridge.trainable_parameters()),
            'organism_parameters':organism.parameter_count(),'individual_state_scalars':scalars(state),
            'random_byte_loss_not_language_score':float(loss.detach()),'cpu_forward_backward_seconds':time.perf_counter()-start,
            'notes':['Memory access has a gradient, but byte-to-fact semantic alignment is untrained',
                     'Frozen organism preserves old answers by construction; new language ability must be measured later',
                     'Published adapter remains pristine; disposable GPU trial uses one random-byte optimizer step']}
    if torch.cuda.is_available():
        gpu_organism,gpu_state,_=load(source,'cuda');gpu_bridge=LanguageBridge(gpu_organism).cuda()
        load_interface(gpu_bridge,payload,before)
        torch.cuda.reset_peak_memory_stats();gpu_tokens=torch.randint(0,256,(4,128),device='cuda')
        torch.cuda.synchronize();gpu_start=time.perf_counter()
        # Include Adam moment allocation in the memory bound. This disposable
        # random-byte trial cannot change the pristine published interface.
        trainer=ByteTrainer(gpu_bridge,gpu_state,before,{'data':'random-integer-bytes'},1710000)
        gpu_loss=trainer.step(gpu_tokens.cpu(),4);torch.cuda.synchronize()
        report['gpu_smoke']={'device':torch.cuda.get_device_name(),'batch':4,'length':128,
                             'forward_backward_seconds':time.perf_counter()-gpu_start,
                             'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                             'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                             'timing_includes_optimizer':True,'timing_is_cold_start':True,
                             'memory_excludes_driver_context':True,
                             'finite':bool(torch.isfinite(torch.tensor(gpu_loss))),'optimizer_steps':1,
                             'random_byte_updates_only':True,'published_interface_optimizer_steps':0}
        if not report['gpu_smoke']['finite']:raise AssertionError('GPU smoke is not finite')
    report['passed']=all(report[k] for k in ('causal_prefix_exact','trainable_gradients_finite','organism_frozen','interface_restart_exact','organism_file_unchanged'))
    (output/'readiness.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
