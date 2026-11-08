"""Fetch a small pinned train-only TextGrid audit sample, not training data."""
import concurrent.futures
import hashlib
import json
import random
import subprocess
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT/'runs/research/aishell4'
REV = 'aada72727856313b19d4a030383c426364931dbf'
OUT = RESEARCH/'sample-v1'


def fetch(record):
    path = record['path']
    local = OUT/(hashlib.sha256(path.encode('utf8')).hexdigest()+'.TextGrid')
    url = 'https://huggingface.co/datasets/AISHELL/AISHELL-4/resolve/'+REV+'/'+quote(path, safe='/')
    subprocess.run(['curl.exe','--ssl-no-revoke','-L','--fail','--max-time','90','--silent','--show-error',
                    '--output',str(local),url],check=True)
    value = local.read_bytes()
    digest = hashlib.sha1(b'blob '+str(len(value)).encode('ascii')+b'\0'+value).hexdigest()
    if len(value) != record['size'] or digest != record['oid']:
        raise ValueError('Git blob verification failed: '+path)
    return dict(source_path=path, git_blob_sha1=digest, sha256=hashlib.sha256(value).hexdigest(),
                bytes=len(value), local_file=local.name)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    tree = json.loads((RESEARCH/'tree.json').read_text(encoding='utf8'))
    rows = sorted((r for r in tree if r.get('type') == 'file' and r['path'].endswith('.TextGrid')
                   and r['path'].split('/')[0] in ('train_L','train_M','train_S')), key=lambda r:r['path'])
    seed, count = 801109, 12
    if len(rows) < count:
        raise ValueError('Insufficient train files')
    selection = random.Random(seed).sample(rows,count)
    OUT.mkdir(parents=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        files = list(pool.map(fetch, selection))
    result = dict(status='sample_for_review_not_approved_for_training', repository='AISHELL/AISHELL-4',
                  revision=REV, seed=seed, train_textgrid_count=len(rows), sample_size=count, files=files,
                  limitations=['Source license labels differ; review before use.',
                               'Sample includes natural speech and may contain personal information.',
                               'TLS certificate chain verified; revocation check unavailable.',
                               'Training files only; no test files downloaded.'])
    (OUT/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k != 'files'},ensure_ascii=False))


if __name__ == '__main__':
    main()
