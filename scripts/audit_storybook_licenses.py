"""Compare pinned story labels to current per-story license pages."""
import concurrent.futures
import hashlib
import json
import re
import subprocess
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'runs/research/storybooks-canada/raw-v1'
OUT=ROOT/'runs/research/storybooks-canada/license-audit-v1'


def check(row):
    data=(DATA/row['file']).read_bytes()
    if hashlib.sha256(data).hexdigest()!=row['sha256']:
        raise ValueError('Source file changed')
    text=data.decode('utf8')
    matches=re.findall(r'^\*\s*License:\s*\[([^\]]+)\]\s*$',text,re.M)
    if len(matches)!=1:
        raise ValueError('Missing or ambiguous story label')
    label=matches[0].strip().upper()
    result=dict(story=row['story'],source_sha256=row['sha256'],source_label=label,
                source_path=row['source_path'],source_git_sha1=row['git_blob_sha1'])
    if label!='CC-BY':
        result['status']='excluded_non_cc_by_label'
        return result
    page=OUT/(row['story']+'.html')
    url='https://storybookscanada.ca/stories/zh/'+row['story']+'/'
    subprocess.run(['curl.exe','--ssl-no-revoke','-L','--fail','--max-time','60','--silent','--show-error',
                    '--output',str(page),url],check=True)
    html=page.read_text(encoding='utf8')
    licenses=sorted(set(re.findall(r'<a\s+rel=["\']license["\']\s+href=["\']([^"\']+)["\']',html,re.I)))
    result.update(page_url=url,page_sha256=hashlib.sha256(page.read_bytes()).hexdigest(),license_urls=licenses)
    result['status']='license_page_review_needed'
    return result


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf8'))
    OUT.mkdir(parents=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results=list(pool.map(check,manifest['files']))
    labels={}
    for x in results:labels[x['source_label']]=labels.get(x['source_label'],0)+1
    report=dict(status='license_review_pending_no_training',reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
                repository=manifest['repository'],revision=manifest['revision'],labels=labels,stories=results,
                limitations=['Site pages are mutable; captured SHA-256 does not prove correspondence to pinned source revision.',
                             'Per-story licensing, upstream rights and contributor attribution need human interpretation.',
                             'Non-CC-BY labels excluded from page requests, not automatically rejected as unlawful.',
                             'No source data is approved for training.'])
    (OUT/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'status':report['status'],'labels':labels,'pages':sum('page_url' in r for r in results)}))


if __name__=='__main__':
    main()
