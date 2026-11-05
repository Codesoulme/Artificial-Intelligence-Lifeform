"""Fixed P17 measurements; evaluation never writes model weights."""
import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spora.growing import load,query
from spora.language_bridge import LanguageBridge,encode
from spora.language_training import load_interface,frozen_digest
from spora.language_eval import evaluate,generate
from spora.text_data import read_manifest,windows


def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def score_target(model,state,prompt,target):
    prefix=encode(prompt)[:-1]
    answer=list(target.encode('utf-8'))
    tokens=torch.tensor([prefix+answer],device=next(model.parameters()).device)
    logits=model(tokens,state)[0,len(prefix)-1:-1]
    return float(F.cross_entropy(logits,tokens[0,len(prefix):]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--interface',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--final',action='store_true')
    parser.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    plan=json.loads((ROOT/'configs/chinese-pilot-100-values.json').read_text())
    parent=ROOT/plan['parent'];manifest=ROOT/plan['manifest']
    if file_hash(parent)!=plan['parent_sha256'] or file_hash(manifest)!=plan['manifest_sha256']:
        raise ValueError('Protocol parent/data mismatch')
    torch.set_num_threads(2)
    organism,state,_=load(parent,args.device)
    before=frozen_digest(organism,state)
    bridge=LanguageBridge(organism).to(args.device).eval()
    payload=torch.load(args.interface,map_location='cpu',weights_only=True)
    load_interface(bridge,payload,plan['parent_sha256'])
    if args.final and (payload.get('steps')!=100 or payload['provenance']['manifest_sha256']!=plan['manifest_sha256']):
        raise ValueError('Final evaluation requires the fixed 100-step checkpoint')
    documents=read_manifest(manifest)
    probes=json.loads((ROOT/plan['probes']).read_text(encoding='utf-8'))
    started=time.perf_counter()
    report={'parent_sha256':file_hash(parent),'interface_sha256':file_hash(args.interface),
        'manifest_sha256':file_hash(manifest),'steps':payload.get('steps',0),
        'trained_targets':payload.get('trained_tokens',0),'losses':{},'probes':[],'general':[]}
    for split in ('train','development','test') if args.final else ('train','development'):
        report['losses'][split]=evaluate(bridge,state,windows(documents,split,128))
    report['losses']['focus']=evaluate(bridge,state,windows([d for d in documents if d['path'].startswith('focus-')],'train',128))
    vocabulary=['富强','民主','文明','和谐','自由','平等','公正','法治','爱国','敬业','诚信','友善']
    for probe in probes['probes']:
        result=generate(bridge,state,probe['prompt'],probes['max_new_bytes'])
        candidates=['国家','社会','个人'] if probe['kind']=='level' else vocabulary
        matches=re.findall('|'.join(candidates),result['text'])
        # No repairing UTF-8 or accepting a list containing every alternative.
        correct=result['utf8_valid'] and matches==probe['expected']
        row=dict(probe,**result,matched_terms=matches,correct=correct,
            expected_answer_nll=score_target(bridge,state,probe['prompt'],'、'.join(probe['expected'])))
        report['probes'].append(row)
    report['general']=[dict(prompt=p,**generate(bridge,state,p,96)) for p in probes['general_prompts']]
    report['diagnostics']={kind:{'count':sum(p['kind']==kind for p in report['probes']),
        'correct':sum(p['correct'] for p in report['probes'] if p['kind']==kind)} for kind in ('group','level','example')}
    generated=report['probes']+report['general']
    report['generation']={'prompts':len(generated),'valid_utf8':sum(p['utf8_valid'] for p in generated),
        'empty':sum(not p['generated_byte_values'] for p in generated)}
    device=args.device
    with torch.no_grad():
        answers=query(organism,state,torch.tensor([12]*4,device=device),
            torch.tensor([0,1,1,1],device=device),torch.tensor([0,0,1,2],device=device)).argmax(-1).tolist()
    report['memory']={'answers':answers,'expected':[7,5,9,3],
        'all_organism_tensors_unchanged':before==frozen_digest(organism,state),'parent_file_unchanged':file_hash(parent)==plan['parent_sha256']}
    report['seconds']=time.perf_counter()-started
    report['measurements_valid']=answers==[7,5,9,3] and all(report['memory'][k] for k in ('all_organism_tensors_unchanged','parent_file_unchanged')) and all(math.isfinite(row['negative_log_likelihood']) for row in report['losses'].values())
    report['limitations']='Finite shared-fact diagnostics. Loss reduction or term matches do not prove understanding or stable behavioral values.'
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ('probes','general')},ensure_ascii=False),flush=True)
    if not report['measurements_valid']:raise SystemExit(1)


if __name__=='__main__':main()
