"""Cold load and measure final P18 data once; do not select or update weights."""
import json
import torch
from chinese_curriculum_common import ROOT,read,sha,plan_data,new_trainer,content_probes,memory_answers
from spora.language_training import frozen_digest
from spora.language_eval import evaluate
from spora.text_data import windows


def main():
    plan,docs,examples,groups,labels=plan_data()
    destination=ROOT/plan['output']/'final-evaluation.json'
    if destination.exists():raise FileExistsError(destination)
    trainer=new_trainer(plan,groups)
    before=frozen_digest(trainer.model.organism,trainer.state)
    checkpoint=ROOT/plan['output']/'checkpoint.pt'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    trainer.restore(payload)
    complete=read(plan['output']+'/complete.json')
    if trainer.steps!=complete['new_steps'] or trainer.steps>plan['new_steps']:raise ValueError('Budget differs')
    old_probes=read('data/chinese-pilot/values-probes.json')['probes']
    new_probes=read('data/chinese-pilot/values-probes-v2.json')['probes']
    result={'new_steps':trainer.steps,'total_lineage_steps':trainer.lineage['prior_steps']+trainer.steps,
        'checkpoint_sha256':sha(checkpoint),'losses':{split:evaluate(trainer.model,trainer.state,windows(docs,split,plan['length']))
            for split in ('train','development','test')},
        'previous_probes':content_probes(trainer.model,trainer.state,old_probes),
        'new_probes':content_probes(trainer.model,trainer.state,new_probes),
        'memory':memory_answers(trainer),
        'old_tensors_exact':before==frozen_digest(trainer.model.organism,trainer.state),
        'sampler_restored':trainer.sampler.draws==trainer.steps*plan['batch'],
        'sealed_test_evaluations':1,'cold_process':True}
    result['integrity_passed']=result['memory']['passed'] and result['old_tensors_exact'] and result['sampler_restored']
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'integrity_passed':result['integrity_passed'],'losses':result['losses'],
        'previous_correct':result['previous_probes']['correct'],'new_correct':result['new_probes']['correct'],
        'previous_utf8':result['previous_probes']['valid_utf8'],'new_utf8':result['new_probes']['valid_utf8']}),flush=True)
    if not result['integrity_passed']:raise SystemExit(1)


if __name__=='__main__':main()
