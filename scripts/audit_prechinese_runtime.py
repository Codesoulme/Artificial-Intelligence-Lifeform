"""Lineage, mixed transactions, consolidation, and byte-resume integration."""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spora.growing import load, save, grow, query
from spora.state_codec import flatten
from spora.durable import TensorStore
from spora.semantic_memory import SemanticMemory, consolidate_semantic_store, semantic_query
from spora.skills import FeedbackPolicy
from spora.internal_memory import observe_sourced, read_provenance
from spora.sleep_memory import consolidate_store
from spora.language_bridge import LanguageBridge
from spora.language_training import ByteTrainer, atomic_save, interface_state
from audit_durable import fingerprint, worker_line
from audit_internal_runtime import protected
from audit_time_runtime import history_equal


def protected_all():
    hashes = protected()
    report = json.loads((ROOT/'reports/P12-P13-results.json').read_text())
    for name, info in report['candidate']['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != info['sha256']:
            raise AssertionError('P12 parent changed: '+name)
        hashes[name] = info['sha256']
    return hashes


def extend():
    model, state, meta = load(ROOT/'checkpoints/individual-internal-v2-candidate.pt')
    torch.manual_seed(2510000)
    model.semantic_memory = SemanticMemory()
    data = torch.load(ROOT/'runs/p14-p15-audit-v2/skill-seed7.pt', weights_only=True)
    model.skill_policy = FeedbackPolicy(**data['config'])
    model.skill_policy.load_state_dict(data['state'])
    return model, state, dict(meta, stage='P14-P16 pre-Chinese candidate',
        parent_sha256=hashlib.sha256((ROOT/'checkpoints/individual-internal-v2-candidate.pt').read_bytes()).hexdigest(),
        skill_training=data['training'], semantic_status='inactive; original individual has no sourced concept curriculum',
        Chinese_training_started=False, default_read='exact')


@torch.no_grad()
def mixed_change(model, state):
    state, _ = observe_sourced(model, state, 12, 8, 'direct', 1.)
    state, _ = observe_sourced(model, state, 12, 2, 'simulation', 1.)
    model.semantic_memory.head.bias.add_(.01)
    model.skill_policy.head.bias.add_(.02)
    # Atomicity stress only: a deliberately distinguishable runtime tensor.
    state.runtime.tensors['observed'].add_(.001)
    return state


def child(path, stage):
    torch.set_num_threads(2)
    with TensorStore(path) as store:
        model, state, _, revision = store.read()
        state = mixed_change(model, state)
        def hook(at):
            if at == stage:
                print(json.dumps({'paused': at}), flush=True)
                sys.stdin.readline()
        store.commit(model, state, revision, hook=hook)


def byte_audit(model, state, output):
    before = fingerprint(model, state)
    torch.manual_seed(2541000)
    bridge = LanguageBridge(model)
    trainer = ByteTrainer(bridge, state, 'runtime-parent', {'data':'random-integer-bytes','batch':2}, 2541000)
    examples = torch.randint(256, (16, 32))
    for _ in range(3):trainer.step(examples, 2)
    atomic_save(trainer.snapshot(), output/'random-byte-step3.pt')
    losses = [trainer.step(examples, 2) for _ in range(3)]
    other_model, other_state = copy.deepcopy(model), copy.deepcopy(state)
    other = ByteTrainer(LanguageBridge(other_model), other_state, 'runtime-parent', {'data':'random-integer-bytes','batch':2}, 2541000)
    other.restore(torch.load(output/'random-byte-step3.pt', weights_only=True))
    continuation = [other.step(examples, 2) for _ in range(3)]
    exact = losses == continuation and all(torch.equal(v, interface_state(other.model)[k]) for k,v in interface_state(bridge).items())
    return {'random_byte_updates':6,'synthetic_token_presentations':trainer.tokens_seen,
        'resume_losses_and_tensors_exact':exact,'organism_all_tensors_exact':fingerprint(model,state)==before,
        'frozen_gradients_absent':all(p.grad is None for p in model.parameters()),
        'Chinese_training_started':False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('--child')
    parser.add_argument('--stage')
    args = parser.parse_args()
    if args.child:return child(args.child,args.stage)
    output = Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    hashes = protected_all()
    if not json.loads((ROOT/'runs/p14-p15-audit-v2/audit.json').read_text())['passed']:
        raise ValueError('Prior concept/skill gates failed')
    original, os, _ = load(ROOT/'checkpoints/individual-internal-v2-candidate.pt')
    model, state, meta = extend()
    _, old = flatten(original,os)
    _, new = flatten(model,state)
    report = {'originals':hashes, 'old_tensors_exact':all(torch.equal(t,new[k]) for k,t in old.items()),
        'old_history_logits_exact':history_equal(original,os,model,state),'crashes':[]}
    baseline = fingerprint(model,state)
    changed_model, changed_state = copy.deepcopy(model), copy.deepcopy(state)
    changed_state = mixed_change(changed_model,changed_state)
    changed = fingerprint(changed_model,changed_state)
    base = output/'base.sqlite'
    with TensorStore.create(base,model,state,meta) as store:store.checkpoint()
    for stage in ('before_begin','after_begin','after_first_tensor','after_tensors','after_manifest','before_commit','after_commit'):
        path = output/f'{stage}.sqlite'
        with TensorStore(base) as store:store.backup(path)
        process = subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--child',str(path),'--stage',stage],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            line = worker_line(process)
            if not line or json.loads(line).get('paused') != stage:
                raise RuntimeError('Worker failed: '+process.stderr.read())
            expected = changed if stage=='after_commit' else baseline
            with TensorStore(path) as store:
                rm,rs,_,_=store.read()
                concurrent = fingerprint(rm,rs)==expected
            process.kill();process.wait(timeout=10)
            with TensorStore(path) as store:
                rm,rs,_,rev=store.read()
                exact=fingerprint(rm,rs)==expected
                rm2,rs2=(copy.deepcopy(changed_model),copy.deepcopy(changed_state)) if stage=='after_commit' else (copy.deepcopy(model),copy.deepcopy(state))
                rs=mixed_change(rm,rs);rs2=mixed_change(rm2,rs2)
                continuation=fingerprint(rm,rs)==fingerprint(rm2,rs2)
                store.commit(rm,rs,rev)
            row={'stage':stage,'concurrent_exact':concurrent,'recovery_exact':exact,'continuation_exact':continuation}
            report['crashes'].append(row)
            print(json.dumps(row),flush=True)
        finally:
            if process.poll() is None:process.kill();process.wait(timeout=10)
            for stream in (process.stdin,process.stdout,process.stderr):stream.close()
    report['byte_training']=byte_audit(model,state,output)
    # Rehearse growth and legacy sleep on a separate descendant, preserving
    # both optional modules exactly. No edits to any protected individual.
    sandbox_model,sandbox_state=copy.deepcopy(model),grow(model,copy.deepcopy(state))
    # Parent page zero is already sealed. Complete the next active page using
    # explicitly simulated inputs, which cannot replace factual beliefs.
    while sandbox_state.cursor < 128:
        sandbox_state,_=observe_sourced(sandbox_model,sandbox_state,sandbox_state.cursor%64,
            sandbox_state.cursor%16,'simulation',1.)
    module_before={k:v.clone() for k,v in sandbox_model.state_dict().items()}
    with TensorStore.create(output/'sleep.sqlite',sandbox_model,sandbox_state,meta) as store:
        result=consolidate_store(store,sandbox_model,sandbox_state,store.revision,1,steps=600)
        sm,ss,_,_=store.read()
        module_exact=all(torch.equal(v,sm.state_dict()[k]) for k,v in module_before.items())
    report['sleep']={'accepted':result['accepted'],'modules_exact':module_exact,'gate':result['gate']}
    # Real gated semantic publication from newly supplied constant-rule facts;
    # old observations are retained. This synthetic descendant is not exported
    # as the user's original experiential state.
    for key in range(32):
        current=int(query(sm,ss,torch.tensor([key]),torch.tensor([0]),torch.tensor([0])).argmax(-1))
        if current==3:continue
        ss,_=observe_sourced(sm,ss,key,3,'direct',1.)
    with TensorStore.create(output/'semantic.sqlite',sm,ss) as store:
        result=consolidate_semantic_store(store,sm,ss,store.revision,seed=2542000)
        cm,cs,_,_=store.read()
        report['semantic_publication']={'accepted':result['accepted'],
            'active_after_restore':int(cm.semantic_memory.trained_cursor)==cs.cursor,
            'gate':result,'source_preserved':read_provenance(cm,cs,12,0)['source'] is None}
        save(output/'semantic-demonstration.pt',cm,cs)
    protected_all()
    report['passed']=report['old_tensors_exact'] and report['old_history_logits_exact'] and all(
        all(row[k] for k in ('concurrent_exact','recovery_exact','continuation_exact')) for row in report['crashes']) and all(
        report['byte_training'][k] for k in ('resume_losses_and_tensors_exact','organism_all_tensors_exact','frozen_gradients_absent')) and all(
        report['sleep'][k] for k in ('accepted','modules_exact')) and all(
        report['semantic_publication'][k] for k in ('accepted','active_after_restore','source_preserved'))
    (output/'audit.json').write_text(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
