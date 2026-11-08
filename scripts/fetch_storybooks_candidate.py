"""Fetch a pinned story corpus for local review, without approving training."""
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT/'runs/research/storybooks-canada'
REV = 'be28dab38ac55e8eb2185edb6631b11615d02185'
OUT = RESEARCH/'raw-v1'


def download(row):
    key = row['path'].split('/')[-1][:4]
    if not (key.isascii() and key.isdigit()):
        raise ValueError('Unexpected story filename')
    out = OUT/(key+'.md')
    url = 'https://raw.githubusercontent.com/global-asp/sbc-source/'+REV+'/'+quote(row['path'],safe='/')
    subprocess.run(['curl.exe','--ssl-no-revoke','-L','--fail','--max-time','60','--silent','--show-error',
                    '--output',str(out),url],check=True)
    data = out.read_bytes()
    digest = hashlib.sha1(b'blob '+str(len(data)).encode('ascii')+b'\0'+data).hexdigest()
    if digest != row['sha'] or len(data) != row['size']:
        raise ValueError('Story hash mismatch')
    return dict(story=key, source_path=row['path'], git_blob_sha1=digest, bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(), file=out.name)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    metadata = json.loads((RESEARCH/'metadata.json').read_text(encoding='utf8'))
    if metadata.get('truncated') is not False or metadata.get('sha') != REV:
        raise ValueError('GitHub tree is incomplete or not the pinned revision')
    rows = sorted((r for r in metadata['tree'] if r['path'].startswith('zh/') and r['path'].endswith('.md')),
                  key=lambda r:r['path'])
    if len(rows) != 40 or len({r['path'].split('/')[-1][:4] for r in rows}) != 40:
        raise ValueError('Unexpected source files')
    OUT.mkdir(parents=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        files = list(pool.map(download,rows))
    report = dict(status='candidate_for_review_not_approved_for_training',repository='global-asp/sbc-source',
                  revision=REV,files=files,scope='All Chinese text files for license and content review, not a training set.',
                  limitations=['Per-story license and attribution must be verified against the story page.',
                               'Some stories are noncommercial; do not assume one blanket license.',
                               'TLS chain checked but revocation check unavailable.'])
    (OUT/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'status':report['status'],'files':len(files),'bytes':sum(x['bytes'] for x in files)}))


if __name__=='__main__':
    main()
