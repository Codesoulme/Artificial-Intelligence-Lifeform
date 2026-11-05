"""Build P18 data using training-only characters and independently written splits."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spora.text_data import read_manifest,windows,overlap_audit


def main():
    plan=json.loads((ROOT/'configs/chinese-curriculum-1000.json').read_text(encoding='utf-8'))
    output=(ROOT/plan['manifest']).parent
    output.mkdir(parents=True,exist_ok=False)
    source_paths=['data/chinese-pilot/source.json','data/chinese-pilot/values-focus.json','data/chinese-pilot/basic-curriculum-v2.json']
    base,focus,basic=[json.loads((ROOT/p).read_text(encoding='utf-8')) for p in source_paths]
    rows=[]
    def write(name,text,split,origin):
        file=output/(name+'.txt')
        file.write_text(text+'\n',encoding='utf-8')
        rows.append({'path':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
            'split':split,'source':origin['source'],'license':origin['license']})
    for i,text in enumerate(base['train']):write(f'basic-original-{i:03d}',text,'train',base)
    for i,text in enumerate(focus['train']):write(f'focus-{i:03d}',text,'train',focus)
    for i,text in enumerate(basic['short_train']):write(f'basic-short-{i:03d}',text,'train',basic)
    for i,word in enumerate(basic['words']):write(f'basic-word-{i:03d}','读词：'+word+'。','train',basic)
    for i,subject in enumerate(basic['subjects']):
        for j,action in enumerate(basic['actions']):write(f'basic-combination-{i:02d}-{j:02d}',subject+action+'。','train',basic)
    # No characters are extracted from development/test documents or probes.
    teaching=''.join(base['train']+focus['train']+basic['short_train']+basic['words']+basic['subjects']+basic['actions'])
    chars=sorted({c for c in teaching if '\u4e00'<=c<='\u9fff'})
    for start in range(0,len(chars),16):
        write(f'basic-characters-{start//16:03d}','识字：'+'、'.join(chars[start:start+16])+'。','train',basic)
    for split in ('development','test'):
        for i,text in enumerate(basic[split]):write(f'{split}-{i:03d}',text,split,basic)
    manifest=output/'manifest.json'
    manifest.write_text(json.dumps({'schema':'spora-text-v1','documents':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    docs=read_manifest(manifest)
    counts={split:{'documents':sum(d['split']==split for d in docs),
        'bytes':sum(len(d['text'].encode('utf-8')) for d in docs if d['split']==split),
        'windows':len(windows(docs,split,128))} for split in ('train','development','test')}
    report={'passed':True,'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths},
        'splits':counts,'training_characters':len(chars),'focus_windows':len(windows([d for d in docs if d['path'].startswith('focus-')],'train',128)),
        'overlap':overlap_audit(docs),'note':'New held-out documents; original P17 held-outs not added to training. Templates and isolated words are disclosed priors.'}
    (output/'readiness.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
