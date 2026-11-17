"""Oracle-span frozen embedding baseline on saved identity episodes."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer
from scripts.deepseek_local_baseline import MODEL,WEIGHT_SHA,sha
from scripts.deepseek_identity_selector_probe_20261117 import TRAIN,HELD,TRAIN_Q,HELD_Q,RECORD
from scripts.deepseek_kernel_value_span_audit_20261117 import value_token_indices
from spora.research_identity_threshold import calibrate,score

SOURCE=ROOT/'runs/deepseek-identity-selector-near-name-probe-20261117/result.json'
OUT=ROOT/'runs/identity-embedding-similarity-20261117.json'


def main():
    if OUT.exists():raise FileExistsError(OUT)
    start=time.perf_counter();torch.set_num_threads(2)
    model_path=MODEL/'model.safetensors';source_sha=sha(SOURCE)
    if sha(model_path)!=WEIGHT_SHA:raise ValueError('Base hash mismatch')
    previous=json.loads(SOURCE.read_text(encoding='utf8'))
    if tuple(previous['train_names'])!=TRAIN or tuple(previous['held_names'])!=HELD:
        raise ValueError('Names differ from saved episode design')
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    vectors={};token_sequences={}
    with safe_open(str(model_path),framework='pt',device='cpu') as f:
        table=f.get_slice('model.embed_tokens.weight')
        def feature(text,name):
            encoded=tok(text,return_offsets_mapping=True,add_special_tokens=False)
            indexes=value_token_indices(text,name,encoded['offset_mapping'])
            ids=[encoded['input_ids'][i] for i in indexes]
            rows=torch.stack([table[token_id:token_id+1][0].float() for token_id in ids])
            return F.normalize(rows.mean(0),dim=0),ids
        for name in TRAIN+HELD:
            rec,rec_ids=feature(RECORD.format(name=name),name)
            query=[];same=[]
            for template in (TRAIN_Q if name in TRAIN else HELD_Q):
                q,ids=feature(template.format(name=name),name)
                query.append(q);same.append(ids==rec_ids)
            vectors[name]=(rec,query)
            token_sequences[name]={'record_ids':rec_ids,'query_matches_record_ids':same}
    def transform(row):
        name=row['query_name'];q=vectors[name][1][row['query_template']]
        scores=[float((q*vectors[c][0]).sum()) for c in row['candidate_names']]+[0.0]
        return {'known':row['known'],'target':row['target'],'scores':scores}
    train=[transform(row) for row in previous['train_rows']]
    held=[transform(row) for row in previous['rows']]
    near=[]
    for row in previous['near_name']:
        q=vectors[row['query_name']][1][row['template']]
        near.append({'known':False,'scores':[float((q*vectors[c][0]).sum()) for c in row['candidate_names']]+[0.0],
                     'query_name':row['query_name']})
    threshold=calibrate([r for r in train if not r['known']],2)
    if sha(model_path)!=WEIGHT_SHA or sha(SOURCE)!=source_sha:raise ValueError('Protected source changed')
    result={'base_sha256':WEIGHT_SHA,'source_sha256':source_sha,'threshold':threshold,
            'train':score(train,threshold),'held':score(held,threshold),'near':score(near,threshold),
            'same_token_sequence_counts':sum(sum(x['query_matches_record_ids']) for x in token_sequences.values()),
            'total_query_representations':2*len(token_sequences),'token_sequences':token_sequences,
            'near_details':[{'query_name':r['query_name'],'margin':max(r['scores'][:-1]),
                             'false_accept':max(r['scores'][:-1])>threshold} for r in near],
            'seconds':time.perf_counter()-start,
            'limitations':['Oracle supplies exact subject span and candidate identities.',
                           'Token embeddings ignore surrounding context, aliases, and homonyms.',
                           'Same saved, previously inspected synthetic development episodes.',
                           'Threshold chosen on training examples, not a safety guarantee.',
                           'No model inference, memory access, generation, or autonomous identity extraction.']}
    with OUT.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='token_sequences'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
