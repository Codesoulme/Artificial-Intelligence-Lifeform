"""Cold final P19 audit. Never select a checkpoint or modify trained tensors."""
import argparse
import hashlib
import json
import statistics
import time
from types import SimpleNamespace
import torch
from reasoning_task import ROOT, dataset, balanced, batch, audit, changed_query, reversed_events, prompt, OBJECTS
from chinese_curriculum_common import content_probes, memory_answers
from spora.growing import load
from spora.language_bridge import LanguageBridge, PAD, encode
from spora.language_training import load_interface, interface_state, frozen_digest
from spora.language_eval import evaluate
from spora.short_reasoning import restore_research_interface, generate_equal_prefixes
from spora.text_data import read_manifest, windows


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def compact(result):
    return {k:v for k,v in result.items() if k != 'details'}


def paired(original, changed, indices=None):
    if indices is None: indices = list(range(len(original['details'])))
    if not indices: raise ValueError('Nonempty paired group required')
    pairs = [(original['details'][i], changed['details'][i]) for i in indices]
    return {'count':len(pairs),
        'both_answers_correct':sum(a['answer_correct'] and b['answer_correct'] for a,b in pairs)/len(pairs),
        'output_changed':sum(a['output']['bytes_hex'] != b['output']['bytes_hex'] for a,b in pairs)/len(pairs)}


def protected_files(plan):
    previous = read('reports/P17-results.json')
    expected = dict(previous['verification']['protected_hashes'])
    expected[plan['source']] = plan['source_sha256']
    expected['runs/p16-language-ready-v2/untrained-interface.pt'] = previous['baseline']['interface_sha256']
    expected['runs/chinese-curriculum-v2/manifest.json'] = read('configs/chinese-curriculum-1000.json')['manifest_sha256']
    actual = {path:sha(path) for path in expected}
    if actual != expected: raise AssertionError('Protected file changed')
    return actual


def reconstruct(plan, seed, data, documents):
    """Independently rebuild paired data schedules and target counts."""
    focus = windows([d for d in documents if d['path'].startswith('focus-')], 'train', 128)
    basic = windows([d for d in documents if not d['path'].startswith('focus-')], 'train', 128)
    generator = torch.Generator().manual_seed(2810000+seed)
    schedule = hashlib.sha256()
    counts = {'answer':0, 'trace':0, 'recurrent_trace':0, 'replay':0, 'task_presentations':0}
    used_focus, used_basic = set(), set()
    for step in range(1,plan['steps_each']+1):
        if step % plan['replay_every'] == 0:
            n = plan['batch']//2
            fi = torch.randint(len(focus),(n,),generator=generator)
            bi = torch.randint(len(basic),(n,),generator=generator)
            used_focus.update(fi.tolist()); used_basic.update(bi.tolist())
            tokens = torch.cat([focus[fi],basic[bi]])
            schedule.update(b'replay'+tokens.numpy().tobytes())
            counts['replay'] += int((tokens[:,1:] != PAD).sum())
        else:
            indices = torch.randint(len(data['train']),(plan['batch'],),generator=generator).tolist()
            templates = torch.randint(2,(plan['batch'],),generator=generator).tolist()
            schedule.update(json.dumps([indices,templates]).encode())
            rows = [data['train'][i] for i in indices]
            for arm in ('answer','trace'):
                tokens, starts = batch(rows,arm,templates)
                counts[arm] += int(((torch.arange(tokens.shape[1])[None]>=starts[:,None])&(tokens!=PAD)).sum())
            counts['task_presentations'] += len(rows)
    counts['recurrent_trace'] = counts['trace']
    return dict(counts=counts,schedule_sha256=schedule.hexdigest(),sampler_rng=generator.get_state(),
        replay_coverage={'focus_seen':len(used_focus),'focus_total':len(focus),
                         'basic_seen':len(used_basic),'basic_total':len(basic)})


def retention(bridge, state, documents, probes):
    return {'development':evaluate(bridge,state,windows(documents,'development',128)),
        'focus_training':evaluate(bridge,state,windows([d for d in documents if d['path'].startswith('focus-')],'train',128)),
        'probes':content_probes(bridge,state,probes)}


def timed_task(bridge,state,rows,arm,template=0):
    torch.cuda.synchronize(); tick=time.perf_counter()
    result=audit(bridge,state,rows,arm,template)
    torch.cuda.synchronize()
    result['batch_audit_seconds']=time.perf_counter()-tick
    result['generated_bytes']=sum(len(d['output']['bytes_hex'])//2 for d in result['details'])
    return result


def latency(bridge,state,row):
    prefix=encode(prompt(row))[:-1]
    generate_equal_prefixes(bridge,state,[prefix])
    samples=[]
    for _ in range(7):
        torch.cuda.synchronize(); tick=time.perf_counter()
        output=generate_equal_prefixes(bridge,state,[prefix])[0]
        torch.cuda.synchronize(); samples.append(time.perf_counter()-tick)
    return {'samples_seconds':samples,'median_seconds':statistics.median(samples),
        'generated_bytes':len(output['bytes_hex'])//2,'ended_with_eos':output['ended_with_eos'],
        'scope':'one fixed first test prompt, warm process, raw greedy full-sequence recomputation; not general chat speed'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='runs/p19-final-audit');args=parser.parse_args()
    output=ROOT/args.output;output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    plan=read('configs/p19-short-reasoning.json');protected=protected_files(plan)
    data=dataset();test=balanced(data['test'],plan['test_examples'],plan['audit_seed'])
    seen=balanced(data['train'],plan['test_examples'],plan['audit_seed']+1)
    documents=read_manifest(ROOT/'runs/chinese-curriculum-v2/manifest.json')
    probes=read('data/chinese-pilot/values-probes.json')['probes']+read('data/chinese-pilot/values-probes-v2.json')['probes']
    schedules={seed:reconstruct(plan,seed,data,documents) for seed in plan['seeds']}
    report={'plan':plan,'plan_sha256':sha('configs/p19-short-reasoning.json'),
        'device':torch.cuda.get_device_name(),'torch':str(torch.__version__),'protected_hashes':protected,
        'data':{'sizes':{k:len(v) for k,v in data.items()},'object_order':OBJECTS,
            'pairs':{k:sorted({tuple(r['pair']) for r in v}) for k,v in data.items()},
            'test_sample_sha256':hashlib.sha256(json.dumps(test,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
            'clarification':'The plan word sorted was imprecise: code fixed before training uses itertools.combinations in declared OBJECTS order, not Unicode sorting. No partition was changed after results.',
            'seen_training_diagnostic':'96 fixed training-partition worlds, added for fit/generalization diagnosis before final test; not a checkpoint selection signal'},
        'baselines':{'balanced_random_answer':.25,'last_global_event':sum(r['events'][-1][1]==r['answer'] for r in test)/len(test)},
        'schedules':{str(k):{name:v for name,v in item.items() if name!='sampler_rng'} for k,item in schedules.items()},
        'trials':[]}
    organism,state,_=load(ROOT/plan['parent'],'cuda');frozen=frozen_digest(organism,state)
    bridge=LanguageBridge(organism).cuda().eval()
    load_interface(bridge,torch.load(ROOT/plan['source'],map_location='cpu',weights_only=True),plan['parent_sha256'])
    report['source_retention']=retention(bridge,state,documents,probes)
    del bridge,organism,state
    for seed in plan['seeds']:
        for arm in plan['arms']:
            path=f'runs/p19-{arm}-seed{seed}'
            training=read(path+'/training.json');payload=torch.load(ROOT/path/'checkpoint.pt',map_location='cpu',weights_only=True)
            if (payload['steps']!=plan['steps_each'] or payload['arm']!=arm or payload['seed']!=seed
                    or payload['passes']!=(plan['recurrent_passes'] if arm=='recurrent_trace' else 0)
                    or payload['plan_sha256']!=report['plan_sha256'] or payload['source_sha256']!=plan['source_sha256']
                    or sha(path+'/checkpoint.pt')!=training['checkpoint_sha256']):raise ValueError('Final checkpoint protocol mismatch')
            rebuilt=schedules[seed]
            if (not torch.equal(payload['sampler_rng'],rebuilt['sampler_rng'])
                    or payload['supervised_task_targets']!=rebuilt['counts'][arm]
                    or payload['replay_targets']!=rebuilt['counts']['replay']
                    or training['task_presentations']!=rebuilt['counts']['task_presentations']):raise AssertionError('Sampling reconstruction mismatch')
            organism,state,_=load(ROOT/plan['parent'],'cuda')
            bridge=restore_research_interface(organism,payload,plan['parent_sha256'],'cuda')
            restored=interface_state(bridge)
            exact=all(torch.equal(t,restored[k]) for k,t in payload['state'].items())
            if not exact or frozen_digest(organism,state)!=frozen:raise AssertionError('Cold restore differs')
            result={'arm':arm,'seed':seed,'training':training,'cold_restore_exact':exact,
                'sampling_reconstruction_exact':True,'checkpoint_bytes':(ROOT/path/'checkpoint.pt').stat().st_size,
                'interface_tensor_bytes':sum(t.numel()*t.element_size() for t in payload['state'].values())}
            result['standard']=timed_task(bridge,state,test,arm)
            result['new_wording']=timed_task(bridge,state,test,arm,2)
            result['changed_query']=timed_task(bridge,state,[changed_query(r) for r in test],arm)
            result['reversed_events']=timed_task(bridge,state,[reversed_events(r) for r in test],arm)
            result['seen_training']=timed_task(bridge,state,seen,arm)
            result['paired_query']=paired(result['standard'],result['changed_query'])
            result['paired_reversal_changed']=paired(result['standard'],result['reversed_events'],[i for i,r in enumerate(test) if len(r['indices'])==2])
            result['paired_reversal_invariant']=paired(result['standard'],result['reversed_events'],[i for i,r in enumerate(test) if len(r['indices'])==1])
            result['answer_correct_trace_wrong']=sum(d['answer_correct'] and not d['trace_correct'] for d in result['standard']['details']) if arm!='answer' else None
            result['latency']=latency(bridge,state,test[0])
            result['retention']=retention(bridge,state,documents,probes)
            if arm=='recurrent_trace':
                result['inference_ablations']={}
                for passes in (1,0):
                    bridge.passes=passes
                    result['inference_ablations'][str(passes)]=timed_task(bridge,state,test,arm)
                bridge.passes=plan['recurrent_passes']
            result['memory']=memory_answers(SimpleNamespace(model=bridge,state=state))
            result['old_tensors_exact']=frozen_digest(organism,state)==frozen==training['frozen_before']
            result['engineering_passed']=all((result['cold_restore_exact'],result['old_tensors_exact'],result['memory']['passed'],training['passed']))
            result['capability_passed']=all(v>=.95 for v in [result['standard']['answer_accuracy'],
                result['paired_query']['both_answers_correct'],result['paired_reversal_changed']['both_answers_correct'],
                result['paired_reversal_invariant']['both_answers_correct']]) and (arm=='answer' or result['standard']['trace_correct']>=.95)
            report['trials'].append(result)
            (output/f'{arm}-seed{seed}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({'arm':arm,'seed':seed,'standard':compact(result['standard']),
                'query':result['paired_query'],'reversal':result['paired_reversal_changed'],
                'values_correct':result['retention']['probes']['correct'],'integrity':result['engineering_passed']}),flush=True)
            del bridge,organism,state,payload,restored
    if protected_files(plan)!=protected:raise AssertionError('Audit wrote protected files')
    report['engineering_passed']=all(t['engineering_passed'] for t in report['trials'])
    report['capability_passed_by_arm']={arm:all(t['capability_passed'] for t in report['trials'] if t['arm']==arm) for arm in plan['arms']}
    report['default_chat_promoted']=False
    report['limitations']=['Shared P18 parent; seeds vary sampling and new layer initialization, not independent pretraining.',
        'Equal steps and paired data, unequal target tokens, parameter counts and FLOPs.',
        'B/C confounds additional trainable capacity with repeated computation; inference ablations are not training controls.',
        'Finite template task, text working context only; no new events written to autobiographical memory.',
        'Correct supervised traces would not establish faithful internal deliberation.',
        'Existing values probe match rule is a diagnostic, not semantic or behavioral validation.']
    (output/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if not report['engineering_passed']:raise SystemExit(1)


if __name__=='__main__':main()
