"""Fetch pinned KdConv training files for a local, non-promoted quality audit."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/kdconv-material'
REPOSITORY = 'thu-coai/kdconv'
REVISION = '460c94a39c1498241b3c7e94a22be25e1489601e'
FILES = ('film/train.json', 'music/train.json', 'travel/train.json')


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    records = []
    for name in FILES:
        url = f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{name}'
        request = urllib.request.Request(url, headers={'User-Agent': 'SPORA-research/1.0'})
        destination = OUT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        sha = hashlib.sha256()
        total = 0
        with urllib.request.urlopen(request, timeout=60) as source, destination.open('xb') as target:
            while block := source.read(1024 * 1024):
                total += len(block)
                if total > 20 * 1024 * 1024:
                    raise ValueError('Unexpectedly large train file: '+name)
                sha.update(block)
                target.write(block)
        with destination.open(encoding='utf-8') as stream:
            rows = json.load(stream)
        if not isinstance(rows, list) or not rows:
            raise ValueError('Expected a nonempty list: '+name)
        records.append(dict(path=name, url=url, bytes=total, sha256=sha.hexdigest(), dialogues=len(rows)))
        print(json.dumps(records[-1]), flush=True)
    provenance = dict(repository=REPOSITORY, revision=REVISION, license_declared='apache-2.0',
                      license_scope='Dataset card label; verify upstream rights for knowledge content before distribution.',
                      split='train', files=records, ready_for_training=False,
                      scope='Downloaded only train files for inspection; facts and identities are not SPORA experiences.')
    (OUT/'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
