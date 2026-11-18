"""Verify saved interventions and recompute narrow word scores without generation."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from scripts.deepseek_event_focus_generation_20261118 import OUT,read,write,sha,summarize


def main():
    target=OUT/'input-audit.json'
    if target.exists():raise FileExistsError(target)
    plan=read(OUT/'plan.json');result=read(OUT/'result.json')
    if result['plan_sha256']!=sha(OUT/'plan.json') or result['complete'] is not True:
        raise ValueError('Incomplete result or changed plan')
    reference=None;comparisons=[]
    for fixture in plan['fixtures']:
        path=OUT/(fixture['name']+'.pt')
        if sha(path)!=fixture['checkpoint_sha256']:raise ValueError('Changed fixture')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload['format']!='spora-growing-v2':raise ValueError('Unexpected fixture format')
        if reference is not None:
            if payload['schema']!=reference['schema'] or set(payload['tensors'])!=set(reference['tensors']):
                raise ValueError('Schema or tensor set differs')
            changed=[n for n,t in payload['tensors'].items() if not torch.equal(t,reference['tensors'][n])]
            if changed!=['internal/event_focus']:raise ValueError('Intervention changed more than focus: '+str(changed))
            comparisons.append({'fixture':fixture['name'],'changed_tensors':changed})
        else:reference=payload
    rows=result['rows']
    if len(rows)!=len(plan['questions'])*(len(plan['fixtures'])*2+1):raise ValueError('Incomplete generation rows')
    keys=[(r['question'],r['condition'],r['fixture']) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate generation cells')
    expected=set()
    for question in plan['questions']:
        group=[r for r in rows if r['question']==question]
        if len({r['prompt'] for r in group})!=1 or len({tuple(r['prompt_ids']) for r in group})!=1:
            raise ValueError('Prompt changed with memory condition')
        expected.add((question,'disabled',None))
        for condition in ('reader_only','reader_and_finite_output'):
            for fixture in plan['fixtures']:expected.add((question,condition,fixture['name']))
    if set(keys)!=expected:raise ValueError('Wrong generation cells')
    fixtures={f['name']:f for f in plan['fixtures']}
    recomputed=[]
    for row in rows:
        target_word=None if row['condition']=='disabled' else fixtures[row['fixture']]['expected_word']
        score=target_word is not None and row['response'].strip().rstrip('。.!！ \n')==target_word
        if target_word!=row['expected'] or score!=row['exact_word']:raise ValueError('Score mismatch')
        recomputed.append(dict(row,exact_word=score))
    if summarize(recomputed)!=result['summary']:raise ValueError('Summary mismatch')
    write(target,{'focus_only_comparisons':comparisons,'same_prompt_all_conditions':True,
                  'cells':len(rows),'word_scores_recomputed':True,'result_sha256':sha(OUT/'result.json'),
                  'scope':'Artifact and experimental intervention audit; no language capability claim'})
    print(json.dumps({'verified_cells':len(rows),'focus_only_comparisons':len(comparisons)}),flush=True)


if __name__=='__main__':main()
