"""Read-only locator to ordered-vector routing on previously reviewed dialogue."""
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.audit_p236_duconv import ARCHIVE, ARCHIVE_SHA
from scripts.deepseek_spora_duconv_dev_identity_audit import reviewed_sample,select_sample
from scripts.deepseek_spora_consensus_reviewed_dialogue_audit import rows_from_review,chosen,CONTEXT
from scripts.deepseek_spora_general_subject_locator import extract,pack
from scripts.deepseek_spora_interleaved_natural_transaction import HEADS,SOURCE,source_paths
from scripts.deepseek_spora_joint_span_probe import SpanSelector
from scripts.deepseek_spora_natural_write_span_diagnostic import selected_text
from spora.research_context_span import ContextSpanSelector
from spora.research_ordered_identity import select_ordered

OUT=ROOT/'runs/deepseek-natural-ordered-route-audit-20261117.json'
THRESHOLD_FILE=ROOT/'runs/ordered-identity-embedding-20261117.json'


def main():
    if OUT.exists():raise FileExistsError(OUT)
    started=time.perf_counter();torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
    rows=rows_from_review(reviewed_sample(select_sample()))
    paths={**source_paths(),'archive':ARCHIVE,'linear':CONTEXT/'linear-head.pt',
           'report':CONTEXT/'result.json','context':CONTEXT/'head.pt','threshold':THRESHOLD_FILE}
    hashes={name:runtime.sha(path) for name,path in paths.items()}
    context_data=torch.load(paths['context'],map_location='cpu',weights_only=True)
    linear_data=torch.load(paths['linear'],map_location='cpu',weights_only=True)
    report=json.loads(paths['report'].read_text(encoding='utf8'))
    threshold_info=json.loads(paths['threshold'].read_text(encoding='utf8'))
    if (hashes['base']!=runtime.WEIGHT_SHA or hashes['archive']!=ARCHIVE_SHA
            or hashes['context']!=report['head_sha256'] or threshold_info['base_sha256']!=runtime.WEIGHT_SHA
            or any(context_data['source_sha256'][key]!=hashes[key] for key in source_paths())
            or any(linear_data['source_sha256'][key]!=hashes[key] for key in source_paths())):
        raise ValueError('Protected source mismatch')
    runtime.CHECKPOINT=SOURCE
    with tempfile.TemporaryDirectory(prefix='spora-natural-ordered-route-') as folder:
        runtime.WORKING_CHECKPOINT=Path(folder)/'not-created.pt'
        chat=runtime.load_chat(natural_write_heads_path=HEADS)
        items=extract(chat,rows)
        if runtime.WORKING_CHECKPOINT.exists():raise ValueError('Unexpected individual file')
        width=items[0]['features'].shape[-1]
        linear=SpanSelector(width).eval();linear.load_state_dict(linear_data['state'],strict=True)
        context=ContextSpanSelector(width,rank=context_data['rank']).eval();context.load_state_dict(context_data['state'],strict=True)
        # Isolated synthetic enrollment: the reviewed exact mentions form the
        # record list, with duplicate names collapsed. No state is written.
        names=sorted({row['text'][row['left']:row['right']] for row in rows if row['group']=='single'})
        if len(names)<2:raise ValueError('Insufficient distinct names')
        tok=chat.tokenizer
        emb=chat.base.model.embed_tokens
        def representation(text):
            ids=tok.encode(text,add_special_tokens=False)
            if not ids:raise ValueError('Empty identity tokenization')
            with torch.inference_mode():return emb(torch.tensor(ids,device=emb.weight.device)).float().cpu()
        records=[representation(name) for name in names]
        decisions=[]
        for item in items:
            x,_,_,_=pack([item]);span_width=x.shape[1]
            a=chosen(linear,item,span_width);b=chosen(context,item,span_width)
            row=item['row'];accepted=bool(a.any()) and torch.equal(a,b)
            picked=selected_text(row['text'],a,item['offsets'],item['start']) if accepted else None
            selection=select_ordered(representation(picked),records,threshold_info['threshold']) if picked else None
            index=selection['index'] if selection else None
            expected=(row['text'][row['left']:row['right']] if row['group']=='single' else None)
            decisions.append({'digest':row['digest'],'group':row['group'],'locator_selected':picked is not None,
                'locator_exact':picked==expected if expected is not None else None,
                'routed':index is not None,
                'route_matches_reviewed_name':names[index]==expected if index is not None and expected is not None else None,
                'top_score':max(selection['scores']) if selection else None,
                'ambiguous':selection['ambiguous'] if selection else None})
        del chat
    if any(runtime.sha(path)!=hashes[name] for name,path in paths.items()):raise ValueError('Protected source changed')
    summary={}
    for group in ('single','none','multiple'):
        subset=[r for r in decisions if r['group']==group]
        summary[group]={'total':len(subset),'locator_selected':sum(r['locator_selected'] for r in subset),
                        'locator_exact':sum(r['locator_exact'] is True for r in subset),
                        'routed':sum(r['routed'] for r in subset),
                        'correct_reviewed_name':sum(r['route_matches_reviewed_name'] is True for r in subset),
                        'wrong_reviewed_name':sum(r['route_matches_reviewed_name'] is False for r in subset)}
    result={'summary':summary,'rows':decisions,'registered_distinct_names':len(names),'source_sha256':hashes,
            'threshold':threshold_info['threshold'],'seconds':time.perf_counter()-started,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Previously reviewed development data; single annotator, not an independent test.',
                           'Enrolled names are extracted from this same reviewed set; oracle enrollment.',
                           'No unknown-single-person or homonym gold labels; no accuracy claim for multi-person cases.',
                           'Candidate names are exact strings and are not actual stored neural identity states.',
                           'Routes only: no model response, actual memory read, persistent state, or natural write.']}
    with OUT.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
