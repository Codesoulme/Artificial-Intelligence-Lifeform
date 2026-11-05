"""Shared bounded evaluation/data assembly for the fixed P18 experiment."""
import hashlib,json,re,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spora.language_eval import generate,evaluate
from spora.text_data import read_manifest,windows
from spora.curriculum import CoveringSampler,CurriculumTrainer
from spora.growing import load,query
from spora.language_bridge import LanguageBridge


def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def plan_data():
    plan=read('configs/chinese-curriculum-1000.json')
    for path,key in (('parent','parent_sha256'),('source_checkpoint','source_sha256'),('manifest','manifest_sha256')):
        if sha(plan[path])!=plan[key]:raise ValueError('Protocol hash mismatch: '+path)
    docs=read_manifest(ROOT/plan['manifest'])
    examples=windows(docs,'train',plan['length'])
    groups={'focus':[],'basic':[]};labels=[]
    for doc in docs:
        if doc['split']!='train':continue
        group='focus' if doc['path'].startswith('focus-') else 'basic'
        count=len(windows([doc],'train',plan['length']))
        groups[group].extend(range(len(labels),len(labels)+count))
        labels.extend([doc['path']]*count)
    if len(labels)!=len(examples):raise AssertionError('Curriculum alignment differs')
    return plan,docs,examples,groups,labels


def new_trainer(plan,groups):
    torch.set_num_threads(2);torch.manual_seed(plan['seed'])
    organism,state,_=load(ROOT/plan['parent'],plan['device'])
    bridge=LanguageBridge(organism).to(plan['device'])
    provenance={k:plan[k] for k in ('manifest_sha256','batch','length','seed','device','schedule')}
    provenance.update(source_checkpoint_sha256=plan['source_sha256'],stage='P18')
    trainer=CurriculumTrainer(bridge,state,plan['parent_sha256'],provenance,plan['seed'],
        CoveringSampler(groups,plan['schedule'],plan['seed']))
    return trainer


@torch.no_grad()
def raw_generation(bridge,state,prompt,budget=96):
    result=generate(bridge,state,prompt,budget)
    raw=bytes(result.pop('generated_byte_values'))
    result.update(prompt=prompt,bytes_hex=raw.hex(),byte_count=len(raw))
    if not result['utf8_valid']:
        try:raw.decode('utf-8')
        except UnicodeDecodeError as error:
            result.update(error_start=error.start,error_reason=error.reason,
                only_incomplete_tail=error.reason=='unexpected end of data' and error.end==len(raw))
    return result


@torch.no_grad()
def monitor(trainer,docs,plan):
    development=evaluate(trainer.model,trainer.state,windows(docs,'development',plan['length']))
    focus=evaluate(trainer.model,trainer.state,windows([d for d in docs if d['path'].startswith('focus-')],'train',plan['length']))
    generated=[raw_generation(trainer.model,trainer.state,p,plan['generation']['max_new_bytes']) for p in plan['monitor_prompts']]
    return {'stage_steps':trainer.steps,'total_lineage_steps':trainer.lineage['prior_steps']+trainer.steps,
        'development':development,'focus':focus,'generation':generated,
        'valid_utf8':sum(g['utf8_valid'] for g in generated),'prompts':len(generated)}


@torch.no_grad()
def content_probes(bridge,state,rows):
    vocabulary=['富强','民主','文明','和谐','自由','平等','公正','法治','爱国','敬业','诚信','友善']
    results=[]
    for row in rows:
        generated=raw_generation(bridge,state,row['prompt'])
        candidates=['国家','社会','个人'] if row['kind']=='level' else vocabulary
        matches=re.findall('|'.join(candidates),generated['text'])
        results.append(dict(row,**{k:v for k,v in generated.items() if k!='prompt'},matched_terms=matches,
            correct=generated['utf8_valid'] and matches==row['expected']))
    return {'rows':results,'correct':sum(r['correct'] for r in results),'count':len(results),
        'valid_utf8':sum(r['utf8_valid'] for r in results)}


@torch.no_grad()
def memory_answers(trainer):
    device=next(trainer.model.parameters()).device
    answers=query(trainer.model.organism,trainer.state,torch.tensor([12]*4,device=device),
        torch.tensor([0,1,1,1],device=device),torch.tensor([0,0,1,2],device=device)).argmax(-1).tolist()
    return {'actual':answers,'expected':[7,5,9,3],'passed':answers==[7,5,9,3]}
