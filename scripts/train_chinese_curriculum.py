"""P18: at most 1000 additional steps with covering traversal and fixed checks."""
import argparse,json,time
import torch
from chinese_curriculum_common import ROOT,plan_data,new_trainer,monitor,memory_answers,sha
from spora.language_training import frozen_digest,atomic_save


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--start-language-training',action='store_true')
    args=parser.parse_args()
    if not args.start_language_training:parser.error('Explicit start flag required')
    plan,docs,examples,groups,labels=plan_data()
    output=ROOT/plan['output']
    if output.exists():raise FileExistsError(output)
    trainer=new_trainer(plan,groups)
    original=frozen_digest(trainer.model.organism,trainer.state)
    source=torch.load(ROOT/plan['source_checkpoint'],map_location='cpu',weights_only=True)
    trainer.begin(source,plan['source_sha256'])
    output.mkdir(parents=True)
    (output/'config.json').write_text(json.dumps({'plan':plan,'groups':{k:len(v) for k,v in groups.items()},
        'lineage':trainer.lineage,'original_tensor_digest':original},ensure_ascii=False,indent=2),encoding='utf-8')
    baseline=monitor(trainer,docs,plan)
    (output/'baseline.json').write_text(json.dumps(baseline,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'baseline_development_nll':baseline['development']['negative_log_likelihood'],
        'valid_utf8':baseline['valid_utf8'],'prompts':baseline['prompts']}),flush=True)
    started=time.perf_counter();learning_seconds=0.;status='complete';rows=[]
    torch.cuda.reset_peak_memory_stats()
    for _ in range(plan['new_steps']):
        torch.cuda.synchronize();tick=time.perf_counter()
        loss=trainer.step(examples,plan['batch'])
        torch.cuda.synchronize();learning_seconds+=time.perf_counter()-tick
        if trainer.steps % plan['checkpoint_every']==0:
            atomic_save(trainer.snapshot(),output/'checkpoint.pt')
        if trainer.steps % plan['check_every']==0:
            row=monitor(trainer,docs,plan)
            row.update(train_loss_last_batch=loss,targets_seen=trainer.tokens_seen,seconds=time.perf_counter()-started,
                frozen_organism_exact=frozen_digest(trainer.model.organism,trainer.state)==original)
            if not row['frozen_organism_exact']:raise AssertionError('Frozen individual changed')
            if sha(plan['parent'])!=plan['parent_sha256']:raise AssertionError('Parent changed')
            atomic_save(trainer.snapshot(),output/f'step-{trainer.steps:04d}.pt')
            rows.append(row)
            with (output/'metrics.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(json.dumps({'step':trainer.steps,'development_nll':row['development']['negative_log_likelihood'],
                'focus_nll':row['focus']['negative_log_likelihood'],'valid_utf8':row['valid_utf8'],
                'targets':trainer.tokens_seen,'old_tensors_exact':row['frozen_organism_exact']}),flush=True)
            if row['development']['negative_log_likelihood']>baseline['development']['negative_log_likelihood']*1.2:
                status='stopped_development_regression';break
    coverage={name:sum(trainer.sampler.counts[i] for i,label in enumerate(labels) if label==name) for name in set(labels)}
    focus={name:count for name,count in coverage.items() if name.startswith('focus-')}
    result={'status':status,'new_steps':trainer.steps,'prior_steps':trainer.lineage['prior_steps'],
        'total_steps':trainer.steps+trainer.lineage['prior_steps'],'new_targets':trainer.tokens_seen,
        'prior_targets':trainer.lineage['prior_targets'],'sampler_draws':trainer.sampler.draws,
        'coverage':dict(sorted(coverage.items())),'focus_min_presentations':min(focus.values()),
        'all_training_documents_covered':all(coverage.values()),'focus_presentations':sum(focus.values()),
        'memory':memory_answers(trainer),'old_tensors_exact':frozen_digest(trainer.model.organism,trainer.state)==original,
        'parent_file_exact':sha(plan['parent'])==plan['parent_sha256'],
        'source_checkpoint_exact':sha(plan['source_checkpoint'])==plan['source_sha256'],
        'optimizer_seconds':learning_seconds,'seconds_including_periodic_evaluation':time.perf_counter()-started,
        'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'memory_excludes_driver_context':True,'further_training_started':False}
    if not result['old_tensors_exact'] or not result['memory']['passed']:raise AssertionError('Memory protection failed')
    (output/'complete.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='coverage'}),flush=True)


if __name__=='__main__':main()
