"""Freeze character-disjoint research splits without inspecting QA answers."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'runs/research/perltqa-source-20261127'
OUT=SOURCE/'character-split.json'
SALT='spora-perltqa-split-20261127'

def main():
    if OUT.exists():raise FileExistsError(OUT)
    manifest=json.loads((SOURCE/'manifest.json').read_text(encoding='utf8'))
    for name,info in manifest['files'].items():
        if hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()!=info['sha256']:raise ValueError('source changed')
    base=SOURCE/'Dataset/zh'
    memory=json.loads((base/'perltmem.json').read_text(encoding='utf8'))
    data=json.loads((base/'perltqa.json').read_text(encoding='utf8'))
    qa={k:v for row in data for k,v in row.items()}
    mem={row['profile']['Protagonist']:row for row in memory}
    if len(qa)!=32 or len(mem)!=len(memory) or not set(qa)<=set(mem):raise ValueError('unexpected character set')
    names=sorted(qa,key=lambda n:hashlib.sha256((SALT+n).encode('utf8')).digest())
    def entries(names):
        result=[]
        for name in names:
            for key,rows in qa[name]['dialogues'].items():
                contents=mem[name]['dialogues'][key]['contents']
                if not contents:continue
                result.append({'character':name,'dialogue_key':key,'qa_count':len(rows)})
        return result
    splits={'development':entries(names[:4]),'sealed_holdout':entries(names[4:8]),'reserve':entries(names[8:])}
    result={'source_commit':manifest['commit'],'source_sha256':{k:v['sha256'] for k,v in manifest['files'].items()},
        'salt':SALT,'character_names':{'development':names[:4],'sealed_holdout':names[4:8],'reserve':names[8:]},'splits':splits}
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:{'characters':len(result['character_names'][k]),'dialogue_groups':len(v),
                         'questions':sum(x['qa_count'] for x in v)} for k,v in splits.items()},ensure_ascii=False))

if __name__=='__main__':main()
