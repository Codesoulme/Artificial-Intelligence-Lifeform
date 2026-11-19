"""Recompute transfer scores and verify the focus-only intervention artifacts."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from scripts.deepseek_focus_transfer_20261118 import OUT,read,write,sha,check,lexical,summarize


def main():
    target=OUT/'input-audit.json'
    if target.exists():raise FileExistsError(target)
    torch.set_num_threads(2)
    plan=check();result=read(OUT/'result.json')
    if not result['complete'] or result['plan_sha256']!=sha(OUT/'plan.json'):raise ValueError('Result incomplete')
    fixtures={f['name']:f for f in plan['fixtures']}
    if len(fixtures)!=16:raise ValueError('Wrong fixture count')
    worlds={};comparisons=[]
    for fixture in fixtures.values():
        payload=torch.load(OUT/(fixture['name']+'.pt'),map_location='cpu',weights_only=True)
        if payload['format']!='spora-growing-v2':raise ValueError('Wrong state format')
        world=fixture['world']
        if world in worlds:
            reference=worlds[world]
            if payload['schema']!=reference['schema'] or set(payload['tensors'])!=set(reference['tensors']):
                raise ValueError('Within-world state shape changed')
            differences=[n for n,t in payload['tensors'].items() if not torch.equal(t,reference['tensors'][n])]
            if differences!=['internal/event_focus']:raise ValueError('Non-focus state difference')
            comparisons.append(fixture['name'])
        else:worlds[world]=payload
        if payload['tensors']['internal/event_focus'].tolist()!=[fixture['key'],fixture['event']]:
            raise ValueError('Focus mismatch')
    rows=result['rows'];expected=set()
    for q in plan['questions']:
        expected.add((q['name'],'disabled',None))
        for f in fixtures:expected.add((q['name'],'reader_only',f))
        group=[r for r in rows if r['question_name']==q['name']]
        if len({r['prompt'] for r in group})!=1 or len({tuple(r['prompt_ids']) for r in group})!=1:
            raise ValueError('Question-dependent prompt changed with memory')
        if any(r['question']!=q['text'] or r['format']!=q['format'] for r in group):
            raise ValueError('Question changed')
    actual=[(r['question_name'],r['condition'],r['fixture']) for r in rows]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('Missing or duplicated cells')
    for r in rows:
        if r['condition']=='disabled':
            if r['expected'] is not None or r['lexical'] is not None:raise ValueError('Control mislabeled')
            continue
        f=fixtures[r['fixture']]
        if any(r[k]!=f[k] for k in ('world','key','version')) or r['expected']!=f['expected_word']:
            raise ValueError('Reference target mismatch')
        if lexical(r['response'],f['expected_word'])!=r['lexical']:raise ValueError('Lexical score mismatch')
    if summarize(rows,plan)!=result['summary']:raise ValueError('Summary mismatch')
    write(target,{'result_sha256':sha(OUT/'result.json'),'plan_sha256':sha(OUT/'plan.json'),
          'cells_verified':len(rows),'focus_only_comparisons':len(comparisons),
          'scope':'Input integrity and lexical counts; sentence semantics require separate review'})
    print(json.dumps({'verified':len(rows),'focus_only_comparisons':len(comparisons)}),flush=True)


if __name__=='__main__':main()
