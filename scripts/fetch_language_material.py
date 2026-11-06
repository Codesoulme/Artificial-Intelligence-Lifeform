"""Fetch pinned public text datasets; never downloads pretrained model weights."""
import hashlib,json,time,urllib.request,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCES={
 'pretrain':dict(path='pretrain_t2t_mini.jsonl',bytes=1241043656,sha256='6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c'),
 'dialogue':dict(path='sft_t2t_mini.jsonl',bytes=1739201170,sha256='abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518')}

def main():
    key=sys.argv[1];entry=SOURCES[key];out=ROOT/'runs/language-material';out.mkdir(exist_ok=True)
    tree=json.loads((ROOT/'runs/p23-source-discovery/46bfd1b7e183.txt').read_text('utf-8'))
    item=next(x for x in tree['Data']['Files'] if x['Path']==entry['path'])
    if item['Sha256']!=entry['sha256']:raise ValueError('Source metadata differs')
    revision=item['Revision'];url=f"https://modelscope.cn/api/v1/datasets/gongjy/minimind_dataset/repo?Revision={revision}&FilePath={entry['path']}"
    dest=out/entry['path'];part=dest.with_suffix('.part')
    if dest.exists():
        if hashlib.file_digest(dest.open('rb'),'sha256').hexdigest()!=entry['sha256']:raise ValueError('Existing file differs')
        print('Already verified: '+str(dest));return
    start=time.monotonic()
    # Partial attempts are retained, but a retry starts its own full temporary
    # file because not all redirects correctly honor byte-range requests.
    for attempt in range(3):
        attempt_path=part.with_suffix(f'.attempt{attempt}')
        if attempt_path.exists():continue
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'SPORA-research/1.0'})
            digest=hashlib.sha256();count=0;last=time.monotonic()
            with urllib.request.urlopen(request,timeout=45) as response,attempt_path.open('xb') as stream:
                if response.status!=200:raise ValueError('Unexpected HTTP response')
                while True:
                    chunk=response.read(1024*1024)
                    if not chunk:break
                    stream.write(chunk);digest.update(chunk);count+=len(chunk)
                    if count>entry['bytes']:raise ValueError('Unexpected excess data')
                    if time.monotonic()-last>20:
                        print(json.dumps(dict(dataset=key,bytes=count,total=entry['bytes'],seconds=time.monotonic()-start)),flush=True);last=time.monotonic()
            if count!=entry['bytes'] or digest.hexdigest()!=entry['sha256']:raise ValueError('Downloaded bytes/hash differ from repository metadata')
            attempt_path.rename(dest)
            card=(ROOT/'runs/p23-source-discovery/f015c6adf560.txt').read_text('utf-8')
            (out/'dataset-card.md').write_text(card,encoding='utf-8')
            provenance=dict(**entry,url=url,revision=revision,repository='gongjy/minimind_dataset',
                upstream='https://www.modelscope.cn/datasets/gongjy/minimind_dataset',license_declared='CC-BY-NC-4.0',
                use='Local noncommercial research; attribution retained. Dataset card also describes upstream mixed licenses. Text only; no weights.',seconds=time.monotonic()-start)
            (out/f'{key}-provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
            print(json.dumps(provenance),flush=True);return
        except Exception as error:
            print(json.dumps(dict(dataset=key,attempt=attempt,error=str(error))),flush=True)
    raise RuntimeError('No verified dataset downloaded')

if __name__=='__main__':main()
