"""Acquire a fixed, attributed prose candidate; no training or weights."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPOSITORY='0xDing/wikipedia-cn-20230720-filtered'
REVISION='4cef256a3f426ae1d3f6930c8cd59a32d785d99d'
FILENAME='wikipedia-cn-20230720-filtered.json'
DIGEST='3a4dcb0e1b37436358d6cc5de28300071d55367395381958eba9fd38b3d78e32'
SIZE=524489333


def main():
    out=ROOT/'runs/chinese-prose-material';out.mkdir(exist_ok=True)
    final=out/FILENAME;partial=out/(FILENAME+'.partial')
    if final.exists() or partial.exists():raise FileExistsError('Retain existing download and inspect before retry')
    metadata=ROOT/'runs/p23-source-candidates/wiki-filtered-metadata.json'
    card=json.loads(metadata.read_text('utf-8'))
    if card['id']!=REPOSITORY or card['sha']!=REVISION or card['cardData']['license']!='cc-by-sa-3.0':
        raise ValueError('Unexpected repository metadata')
    url=f'https://hf-mirror.com/datasets/{REPOSITORY}/resolve/{REVISION}/{FILENAME}?download=true'
    digest=hashlib.sha256();size=0;start=time.perf_counter();next_notice=50_000_000
    with urllib.request.urlopen(url,timeout=60) as response,partial.open('xb') as dest:
        if response.status!=200:raise ValueError('Full file response required')
        while block:=response.read(1<<20):
            dest.write(block);digest.update(block);size+=len(block)
            if size>SIZE:raise ValueError('Unexpected source size')
            if size>=next_notice:
                print(json.dumps(dict(bytes=size,expected=SIZE)),flush=True);next_notice+=50_000_000
    if size!=SIZE or digest.hexdigest()!=DIGEST:raise ValueError('Source integrity mismatch; partial retained')
    partial.rename(final)
    provenance=dict(repository=REPOSITORY,revision=REVISION,path=FILENAME,url=url,
        bytes=size,sha256=digest.hexdigest(),repository_metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
        license_declared='CC-BY-SA-3.0',source='Chinese Wikipedia 2023-07-20 dump, filtered and script-converted by dataset provider; not an independent truth audit.',
        scope='Candidate text only, not selected by any training plan. Preserve attribution and original source notice; no model weights downloaded.',
        seconds=time.perf_counter()-start)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    (out/'repository-metadata.json').write_bytes(metadata.read_bytes())
    print(json.dumps(provenance),flush=True)


if __name__=='__main__':main()
