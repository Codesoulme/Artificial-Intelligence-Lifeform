"""Held graphs, recurrence interventions, real-clock cost and crash recovery."""
import copy,json,sys,time,tempfile,subprocess,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from spora.awake import AwakeCore
from spora.awake_session import AwakeSession
from spora.growing import load,query
from spora.state_codec import flatten
from spora.durable import TensorStore
from scripts.train_awake_core import evaluate,sha,ROOT,PARENT,PARENT_SHA


def main():
    out=ROOT/'runs/p24-awake/acceptance.json'
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);groups=[]
    for seed in (7,11,23):
        payload=torch.load(ROOT/f'runs/p24-awake/core-seed{seed}.pt',weights_only=True)
        core=AwakeCore();core.load_state_dict(payload['state'])
        groups.append(evaluate(core,seed+190000,500))
    model,state,_=load(ROOT/'runs/p24-awake/individual-awake.pt');core=model.awake_core
    _,before=flatten(model,state)
    before={k:v.detach().clone() for k,v in before.items() if not k.startswith('model/awake_core.')}
    graph=torch.zeros(16,16)
    for i in range(15):graph[i,i+1]=1
    core.observe(graph,0,'synthetic');core.set_goal(15)
    normal=copy.deepcopy(core);ablated=copy.deepcopy(core)
    for _ in range(22):
        normal.tick(250_000_000)
        ablated.values.zero_();ablated.tick(250_000_000)
    recurrence=dict(normal=normal.status(),reset_each_tick=ablated.status(),
                    passed=normal.status()['proposed_node']==1 and ablated.status()['proposed_node'] is None)
    comparisons=[]
    for mode,policy in [('relax','adaptive'),('focus','adaptive'),('relax','fixed')]:
        c=copy.deepcopy(core);c.set_mode(mode)
        start=time.process_time()
        for i in range(100):
            status=c.tick(int(c.interval()*1e9))
            if status['proposed_node'] is not None:break
        comparisons.append(dict(mode=mode,policy=policy,ticks_to_plan=i+1,
            modeled_seconds_to_plan=(i+1)*c.interval(),steps=int(c.neural_steps),
            cpu_seconds=time.process_time()-start,action=status['proposed_node']))
    # Fix the fixed-budget arm explicitly; preserve same task and clock interval.
    c=copy.deepcopy(core);start=time.process_time()
    for i in range(100):
        status=c.tick(250_000_000,policy='fixed')
        if status['proposed_node'] is not None:break
    comparisons[-1]=dict(mode='relax',policy='fixed',ticks_to_plan=i+1,
        modeled_seconds_to_plan=(i+1)*.25,steps=int(c.neural_steps),cpu_seconds=time.process_time()-start,action=status['proposed_node'])
    blocked=graph.clone();blocked[7,8]=0
    normal.observe(blocked,0,'sensor',urgent=True)
    invalidated=normal.status()['proposed_node'] is None
    for _ in range(24):normal.tick(250_000_000)
    correction=dict(invalidated_immediately=invalidated,status=normal.status(),
                    passed=invalidated and normal.status()['outcome']=='unreachable_estimate')
    live=[]
    for mode in ('relax','focus','sleep'):
        core.set_mode(mode);core.observe(graph,0,'synthetic');core.set_goal(15)
        session=AwakeSession(model)  # Marks prior input stale on a new runtime.
        core.observe(graph,0,'synthetic')
        cpu=time.process_time();wall=time.perf_counter();ticks0=int(core.ticks);steps0=int(core.neural_steps)
        first=None
        while time.perf_counter()-wall<2.:
            time.sleep(core.interval());s=session.tick()
            if first is None and s['proposed_node'] is not None:first=time.perf_counter()-wall
        live.append(dict(mode=mode,wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu,
                         ticks=int(core.ticks)-ticks0,neural_steps=int(core.neural_steps)-steps0,
                         first_proposal_seconds=first))
    # Wake then reach a stable proposal for a save/restore continuity check.
    core.set_mode('focus');core.observe(graph,0,'synthetic')
    for _ in range(8):core.tick(50_000_000)
    crash=[]
    with tempfile.TemporaryDirectory() as temporary:
        for stage in ('after_first_tensor','before_commit','after_commit'):
            path=Path(temporary)/(stage+'.sqlite')
            with TensorStore.create(path,model,state) as store:pass
            baseline=int(core.ticks)
            code="""import os,sys
from spora.durable import TensorStore
with TensorStore(sys.argv[1]) as store:
 model,state,_,rev=store.read();model.awake_core.tick(50000000)
 def fail(stage):
  if stage==sys.argv[2]:os._exit(77)
 store.commit(model,state,rev,hook=fail)
"""
            result=subprocess.run([sys.executable,'-c',code,str(path),stage],cwd=ROOT,capture_output=True,text=True)
            with TensorStore(path) as store:
                restored,_,_,_=store.read()
                expected=baseline+(stage=='after_commit')
                crash.append(dict(stage=stage,returncode=result.returncode,ticks=int(restored.awake_core.ticks),expected_ticks=expected,
                                  passed=result.returncode==77 and int(restored.awake_core.ticks)==expected))
    _,after=flatten(model,state)
    old_exact=all(torch.equal(v,after[k]) for k,v in before.items())
    with torch.no_grad():answers=query(model,state,torch.tensor([12]*4),torch.tensor([0,1,1,1]),torch.tensor([0,0,1,2])).argmax(-1).tolist()
    result=dict(held_graphs=groups,recurrence_intervention=recurrence,budget_comparison=comparisons,
                changed_environment=correction,live_clock=live,crash_recovery=crash,
                old_tensors_exact=old_exact,memory_answers=answers,original_file_unchanged=sha(PARENT)==PARENT_SHA,
                scope='Finite16-node fully observed graph planning, three seeds; synthetic environment. Budget imitates explicit desired categories; not learned intrinsic goals or Chinese reasoning. Timing is CPU process time, not energy/power measurement.',
                sources={p:sha(ROOT/p) for p in ('spora/awake.py','spora/awake_session.py','spora/awake_cli.py','spora/state_codec.py','spora/growing.py','scripts/evaluate_awake_core.py')})
    result['passed']=all(g['accuracy']>=.98 for g in groups) and recurrence['passed'] and correction['passed'] and all(c['passed'] for c in crash) and old_exact and answers==[7,5,9,3] and result['original_file_unchanged'] and live[-1]['neural_steps']==0
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('held_graphs','sources')}),flush=True)
    print(json.dumps(dict(held_accuracies=[g['accuracy'] for g in groups])),flush=True)
    if not result['passed']:raise SystemExit('Acceptance failed')

if __name__=='__main__':main()
