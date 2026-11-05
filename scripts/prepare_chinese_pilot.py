"""Materialize and validate an original tiny corpus; never runs an optimizer."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spora.text_data import read_manifest,overlap_audit,windows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    source=ROOT/'data/chinese-pilot/source.json'
    data=json.loads(source.read_text(encoding='utf-8'))
    rows=[]
    for split in ('train','development','test'):
        for number,text in enumerate(data[split]):
            path=output/f'{split}-{number:03d}.txt'
            path.write_text(text+'\n',encoding='utf-8')
            rows.append({'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'split':split,'source':data['source'],'license':data['license']})
    manifest=output/'manifest.json'
    manifest.write_text(json.dumps({'schema':'spora-text-v1','documents':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    documents=read_manifest(manifest)
    report={'passed':True,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'splits':{split:{'documents':sum(d['split']==split for d in documents),
            'utf8_bytes':sum(len(d['text'].encode('utf-8')) for d in documents if d['split']==split),
            'windows128':len(windows(documents,split,128))} for split in ('train','development','test')},
        'overlap':overlap_audit(documents),'Chinese_training_started':False,'optimizer_steps':0,
        'limitation':'Tiny synthetic corpus for pipeline checks. Not evidence of broad Chinese comprehension; repeated short training texts risk overfitting.'}
    (output/'readiness.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':main()
