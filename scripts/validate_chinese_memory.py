"""Frozen final wording tests and causal memory checks; no answer text input."""
import copy
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch

from spora.chinese_query import ChineseQuery, ChineseQueryConfig, answer_memory, attach_query
from spora.durable import TensorStore
from spora.growing import load, save, new_state
from spora.internal_memory import observe_sourced
from spora.state_codec import flatten
from scripts.chinese_query_task import examples
from scripts.train_chinese_query import evaluate, fingerprint

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'runs/p24-awake/individual-awake.pt'
PARENT_SHA='34d023e87bb14504864a2997a29f0842986b163f641f6a1f78232ae4a84e43e4'


def verify_development_binding(report,module_path):
    if not report['passed'] or not all(group['passed'] for group in report['development'].values()):
        raise ValueError('Development gates have not passed')
    if report['checkpoint_sha256']!=fingerprint(module_path):
        raise ValueError('Development result belongs to different question weights')
    if not report.get('source_sha256') or any(fingerprint(ROOT/name)!=digest for name,digest in report['source_sha256'].items()):
        raise ValueError('Development result belongs to different source code')


def module_at(seed):
    payload=torch.load(ROOT/f'runs/p26-query-b/seed{seed}/module.pt',map_location='cpu',weights_only=True)
    module=ChineseQuery(ChineseQueryConfig(**payload['config']))
    module.load_state_dict(payload['state'])
    return module.eval()


def check_memory(seed):
    model,state,_=load(PARENT)
    _,prior=flatten(model,state)
    prior={k:v.detach().clone() for k,v in prior.items()}
    attach_query(model,module_at(seed))
    rows=[]

    def check(label,question,expected_value=None,expected_source=None,status='known',current=state,individual=model):
        result=answer_memory(individual,current,question)
        passed=(result['status']==status and result.get('value')==expected_value
                and result.get('source')==expected_source)
        rows.append(dict(label=label,question=question,expected=dict(value=expected_value,source=expected_source,status=status),
                         actual=result,passed=passed))

    check('original-current','对象12现在的值是多少？',7)
    for event,value in enumerate((5,9,3)):
        check('original-history',f'对象12在事件{event}的值是多少？',value)
    check('old-source-unknown','对象12在事件0的来源是什么？',status='unknown')
    check('future-unknown','对象12在事件999的值是多少？',status='unknown')
    # One wording and one unchanged query network: only the saved neural history
    # and current matrices change. Expected facts are evaluator-owned labels.
    for value in range(16):
        changed,decision=observe_sourced(model,state,12,value,'direct',1.)
        check('counterfactual-current','对象12现在的值是多少？',value,current=changed)
        check('counterfactual-history',f'对象12在事件{state.cursor}的值是多少？',value,current=changed)
        check('counterfactual-source',f'对象12在事件{state.cursor}的来源是什么？',expected_source='direct',current=changed)
        check('correction-keeps-old-history','对象12在事件0的值是多少？',5,current=changed)
    changed,decision=observe_sourced(model,state,12,8,'direct',1.)
    simulated,sim_decision=observe_sourced(model,changed,12,2,'simulation',1.)
    check('simulation-not-current','对象12现在的值是多少？',8,current=simulated)
    check('simulation-in-history',f'对象12在事件{changed.cursor}的值是多少？',2,current=simulated)
    check('simulation-source',f'对象12在事件{changed.cursor}的来源是什么？',expected_source='simulation',current=simulated)
    # Read out unseen object-number combinations after a real neural correction.
    for key in (11,22,33,44,55,60):
        variant,_=observe_sourced(model,state,key,6,'direct',1.)
        check('held-entity-current',f'对象{key}现在的值是多少？',6,current=variant)
    erased=copy.deepcopy(state)
    with torch.no_grad():
        for matrix in erased.current:matrix.zero_()
    check('current-memory-erasure','对象12现在的值是多少？',status='unknown',current=erased)
    erased=copy.deepcopy(state)
    erased.pages[0]=new_state(model).pages[0]
    check('history-memory-erasure','对象12在事件0的值是多少？',status='unknown',current=erased)
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'roundtrip.pt';save(path,model,simulated)
        restored,recovered,_=load(path)
        check('portable-restart','对象12现在的值是多少？',8,current=recovered,individual=restored)
        check('portable-source',f'对象12在事件{changed.cursor}的来源是什么？',expected_source='simulation',current=recovered,individual=restored)
        with TensorStore.create(Path(directory)/'individual.sqlite',model,state) as store:
            store.commit(model,simulated,store.revision)
            recovered_model,recovered_state,_,_=store.read()
            check('transaction-restart','对象12现在的值是多少？',8,current=recovered_state,individual=recovered_model)
    _,after=flatten(model,state)
    original_exact=all(torch.equal(v,after[k]) for k,v in prior.items())
    return dict(seed=seed,cases=len(rows),passed_cases=sum(r['passed'] for r in rows),
        original_tensors_exact=original_exact,rows=rows,
        passed=all(r['passed'] for r in rows) and original_exact and not sim_decision['current_write'])


def main():
    torch.set_num_threads(2)
    out=ROOT/'runs/p26-memory-acceptance'
    out.mkdir(exist_ok=False)
    if fingerprint(PARENT)!=PARENT_SHA:raise ValueError('Parent changed')
    development={seed:json.loads((ROOT/f'runs/p26-query-b/seed{seed}/report.json').read_text('utf-8')) for seed in (7,11,23)}
    for seed,report in development.items():
        verify_development_binding(report,ROOT/f'runs/p26-query-b/seed{seed}/module.pt')
    sources=('spora/chinese_query.py','spora/state_codec.py','spora/growing.py','scripts/chinese_query_task.py',
             'scripts/train_chinese_query.py','scripts/validate_chinese_memory.py')
    frozen=dict(modules={seed:fingerprint(ROOT/f'runs/p26-query-b/seed{seed}/module.pt') for seed in development},
                sources={p:fingerprint(ROOT/p) for p in sources},parent_sha256=PARENT_SHA)
    (out/'frozen-before-final.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    final={};causal=[]
    for seed in (7,11,23):
        module=module_at(seed)
        groups={}
        for i,(keys,times) in enumerate(((False,False),(True,False),(False,True),(True,True))):
            groups[f'held_keys_{keys}_times_{times}']=evaluate(module,examples(26900+i,1024,'final',keys,times))
        groups['balanced_lengths_held_keys']=evaluate(module,examples(26910,1024,'final',True,False,True))
        final[seed]=groups
        result=check_memory(seed);causal.append(result)
        print(json.dumps(dict(seed=seed,accuracies={k:v['accept_accuracy'] for k,v in groups.items()},
                             causal_passed=result['passed_cases'],causal_cases=result['cases'])),flush=True)
    passed=all(g['passed'] for groups in final.values() for g in groups.values()) and all(c['passed'] for c in causal)
    report=dict(development=development,final_wording=final,causal=causal,passed=passed,
        frozen=frozen,original_file_unchanged=fingerprint(PARENT)==PARENT_SHA,
        scope='Finite numbered Chinese questions, neural address selection, existing neural values/provenance, deterministic response formatting. No open chat, new entity semantics, automatic experience writing or unified awake cognition claim.')
    report['passed']=passed and report['original_file_unchanged']
    if report['passed']:
        model,state,_=load(PARENT);attach_query(model,module_at(7))
        metadata=dict(stage='P26 finite Chinese memory query',parent_sha256=PARENT_SHA,
            training='Random-initialized neural address extractor; original facts unchanged.',scope=report['scope'])
        save(out/'individual-query.pt',model,state,metadata)
        with TensorStore.create(out/'individual-query.sqlite',model,state,metadata):pass
        report['candidate_sha256']=fingerprint(out/'individual-query.pt')
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],candidate_sha256=report.get('candidate_sha256'))),flush=True)
    if not report['passed']:raise SystemExit('Acceptance failed; no candidate published')


if __name__=='__main__':main()
