"""Offline ordered-token comparison of prior oracle identity episodes."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
from safetensors import safe_open
from scripts.deepseek_local_baseline import MODEL,WEIGHT_SHA,sha
from spora.research_ordered_identity import ordered_similarity
from spora.research_identity_threshold import calibrate,score

EPISODES=ROOT/'runs/deepseek-identity-selector-near-name-probe-20261117/result.json'
TOKENS=ROOT/'runs/identity-embedding-similarity-20261117.json'
OUT=ROOT/'runs/ordered-identity-embedding-20261117.json'


def main():
    if OUT.exists():raise FileExistsError(OUT)
    started=time.perf_counter();torch.set_num_threads(2)
    source_sha=sha(EPISODES);tokens_sha=sha(TOKENS);path=MODEL/'model.safetensors'
    if sha(path)!=WEIGHT_SHA:raise ValueError('Base hash mismatch')
    episodes=json.loads(EPISODES.read_text(encoding='utf8'))
    prior=json.loads(TOKENS.read_text(encoding='utf8'))
    if prior['base_sha256']!=WEIGHT_SHA or prior['source_sha256']!=source_sha:
        raise ValueError('Prior token provenance mismatch')
    sequences=prior['token_sequences'];vectors={}
    with safe_open(str(path),framework='pt',device='cpu') as f:
        table=f.get_slice('model.embed_tokens.weight')
        for name,item in sequences.items():
            if not all(item['query_matches_record_ids']):raise ValueError('Saved query token ids unavailable')
            vectors[name]=torch.stack([table[i:i+1][0].float() for i in item['record_ids']])
    def convert(row):
        q=vectors[row['query_name']]
        return {'known':row['known'],'target':row['target'],
                'scores':[ordered_similarity(q,vectors[n]) for n in row['candidate_names']]+[0.]}
    train=[convert(r) for r in episodes['train_rows']];held=[convert(r) for r in episodes['rows']]
    near=[{'known':False,'query_name':r['query_name'],
           'scores':[ordered_similarity(vectors[r['query_name']],vectors[n]) for n in r['candidate_names']]+[0.]}
          for r in episodes['near_name']]
    threshold=calibrate([r for r in train if not r['known']],2)
    if sha(path)!=WEIGHT_SHA or sha(EPISODES)!=source_sha or sha(TOKENS)!=tokens_sha:
        raise ValueError('Protected input changed')
    result={'base_sha256':WEIGHT_SHA,'episodes_sha256':source_sha,'token_data_sha256':tokens_sha,
            'threshold':threshold,'train':score(train,threshold),'held':score(held,threshold),'near':score(near,threshold),
            'near_details':[{'query_name':r['query_name'],'best_score':max(r['scores'][:-1]),
                             'false_accept':max(r['scores'][:-1])>threshold} for r in near],
            'seconds':time.perf_counter()-started,
            'limitations':['Oracle name spans, cached identical query/record token sequences.',
                           'Length mismatch is rejected; aliases and tokenization differences may be falsely rejected.',
                           'Identical names cannot be distinguished; no real identity resolution.',
                           'The offline scoring helper does not reject equal-score ties.',
                           'Previously observed small synthetic dev episodes, threshold calibrated on training cases.',
                           'No model generation, autonomous extraction, memory integration or production deployment.']}
    with OUT.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
