"""Calibrate on recorded training cases; audit held/near cases only."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spora.research_identity_threshold import calibrate,score

OUT=ROOT/'runs/identity-threshold-calibration-20261117.json'
SOURCES={'last':ROOT/'runs/deepseek-identity-selector-near-name-probe-20261117/result.json',
         'oracle_span':ROOT/'runs/deepseek-identity-selector-oracle-span-20261117/result.json'}


def main():
    if OUT.exists():raise FileExistsError(OUT)
    result={}
    for label,path in SOURCES.items():
        data=json.loads(path.read_text(encoding='utf8'))
        train=data['train_rows'];held=data['rows']
        neg=[r for r in train if not r['known']]
        threshold=calibrate(neg,2)
        near=[{'known':False,'scores':row['scores']} for row in data['near_name']]
        ordinary=[{'known':False,'scores':row['scores']} for row in data['ordinary']]
        result[label]={'train_negative_total':len(neg),'training_false_accept_cap':2,'threshold':threshold,
                       'default_train':score(train,0.0),'calibrated_train':score(train,threshold),
                       'default_held':score(held,0.0),'calibrated_held':score(held,threshold),
                       'default_near':score(near,0.0),'calibrated_near':score(near,threshold),
                       'default_ordinary':score(ordinary,0.0),'calibrated_ordinary':score(ordinary,threshold),
                       'candidate_sha256':data['candidate_sha256']}
    with OUT.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
