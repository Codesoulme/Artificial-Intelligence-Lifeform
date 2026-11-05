"""Preserved-lineage integration, portable recovery and forced process exits."""
import argparse,copy,hashlib,json,subprocess,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spora.growing import load,save,query,grow
from spora.internal_memory import attach,observe_sourced,read_provenance
from spora.learned_read import prepare_router
from spora.durable import TensorStore
from spora.state_codec import flatten
from audit_internal_memory import controller
from audit_time_runtime import history_equal,protected_files
from audit_durable import fingerprint,worker_line


def protected():
    hashes=protected_files()
    report=json.loads((ROOT/'reports/P11-results.json').read_text())
    for name,info in report['candidate']['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=info['sha256']:raise AssertionError('Time candidate changed')
        hashes[name]=info['sha256']
    return hashes


@torch.no_grad()
def change(model,state):
    state,r=observe_sourced(model,state,12,8,'direct',1.)
    if not r['current_write']:raise AssertionError('Expected finite legacy correction')
    state,r=observe_sourced(model,state,12,2,'simulation',1.)
    if r['current_write']:raise AssertionError('Imagined content became fact')
    return state


def child(path,stage):
    torch.set_num_threads(2)
    with TensorStore(path) as store:
        m,s,_,rev=store.read();s=change(m,s)
        def hook(at):
            if at==stage:
                print(json.dumps({'paused':at}),flush=True);sys.stdin.readline()
        store.commit(m,s,rev,hook=hook)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--child');p.add_argument('--stage');a=p.parse_args()
    if a.child:return child(a.child,a.stage)
    output=Path(a.output).resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);torch.set_num_threads(2);protected_hashes=protected()
    parent=ROOT/'checkpoints/individual-time-candidate.pt'
    original,old,_=load(parent);model,state,meta=load(parent);control,_=controller(7)
    state=attach(model,state,control)
    exact=history_equal(original,old,model,state)
    old_tensors=flatten(original,old)[1];new_tensors=flatten(model,state)[1]
    old_exact=all(torch.equal(v,new_tensors[k]) for k,v in old_tensors.items())
    base=output/'base.sqlite'
    state,training=prepare_router(model,state)
    with TensorStore.create(base,model,state,meta) as store:store.checkpoint()
    baseline=fingerprint(model,state)
    nm,ns=copy.deepcopy(model),copy.deepcopy(state);ns=change(nm,ns);changed=fingerprint(nm,ns)
    report={'originals':protected_hashes,'old_logits_exact':exact,'old_tensors_exact':old_exact,
        'legacy_provenance_unknown':read_provenance(model,state,12,0)['source'] is None,
        'prepare_training':training,'crashes':[]}
    stages=json.loads((ROOT/'configs/p9-p10-plan.json').read_text())['storage']['crash_stages']
    for stage in stages:
        path=output/f'{stage}.sqlite'
        with TensorStore(base) as store:store.backup(path)
        process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--child',str(path),'--stage',stage],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            line=worker_line(process)
            if not line or json.loads(line).get('paused')!=stage:raise RuntimeError('Worker missed boundary')
            expected=changed if stage=='after_commit' else baseline
            with TensorStore(path) as store:
                m,s,_,_=store.read();reader=fingerprint(m,s)==expected
            process.kill();process.wait(timeout=10)
            with TensorStore(path) as store:
                m,s,_,rev=store.read();recovery=fingerprint(m,s)==expected
                refm,refs=(nm,ns) if stage=='after_commit' else (model,state)
                next_s,_=observe_sourced(m,s,13,4,'inference',1.)
                next_ref,_=observe_sourced(refm,refs,13,4,'inference',1.)
                continuation=fingerprint(m,next_s)==fingerprint(refm,next_ref)
                stats=store.commit(m,next_s,rev)
            row={'stage':stage,'reader_exact':reader,'recovery_exact':recovery,'continuation_exact':continuation,'commit':stats}
            report['crashes'].append(row);print(json.dumps(row),flush=True)
        finally:
            if process.poll() is None:process.kill();process.wait(timeout=10)
            for stream in (process.stdin,process.stdout,process.stderr):stream.close()
    with torch.no_grad():
        answers=query(nm,ns,torch.tensor([12]*6),torch.tensor([0,1,1,1,1,1]),torch.tensor([0,0,1,2,72,73])).argmax(-1).tolist()
        expected_answers=[8,5,9,3,8,2]
    provenance=[read_provenance(nm,ns,12,t) for t in (0,72,73)]
    report['demo']={'answers':answers,'expected':expected_answers,'provenance':provenance,
                    'router_invalidated':ns.internal.router_factors is None}
    save(output/'demonstration.pt',nm,ns,meta);rm,rs,_=load(output/'demonstration.pt')
    report['portable_exact']=fingerprint(nm,ns)==fingerprint(rm,rs)
    # Source-association ablation cannot recover stored provenance; restore does.
    ablated=copy.deepcopy(rs)
    for matrix in ablated.internal.provenance:matrix.zero_()
    report['source_ablation_unknown']=read_provenance(rm,ablated,12,72)['source'] is None
    report['restored_source']=read_provenance(rm,rs,12,72)['source']=='direct'
    protected();report['originals_unchanged']=True
    report['passed']=exact and old_exact and report['legacy_provenance_unknown'] and report['portable_exact'] and report['source_ablation_unknown'] and report['restored_source'] and answers==expected_answers and all(all(r[k] for k in ('reader_exact','recovery_exact','continuation_exact')) for r in report['crashes'])
    (output/'audit.json').write_text(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
