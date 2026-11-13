"""Check irrelevant beverage-memory reads on real dialogue in isolated state."""
import json
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

from scripts import deepseek_spora_consensus_identity_guard_audit as audit
from scripts.deepseek_spora_duconv_span_audit import ARCHIVE, ARCHIVE_SHA, PRIOR_SAMPLE, select_candidates
from scripts.deepseek_spora_runtime_chat import sha

OUTPUT = ROOT / 'runs/deepseek-spora-duconv-irrelevant-memory-audit-20261113.json'


def main():
    if sha(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError('Archive digest mismatch')
    excluded = {json.loads(line)['conversation_sha256']
                for line in PRIOR_SAMPLE.read_text(encoding='utf-8').splitlines()}
    with ZipFile(ARCHIVE) as archive, archive.open('DuConv/train.txt') as stream:
        cases = select_candidates(stream, excluded, count=80)[:20]
    names = tuple(row['name'] for row in cases[:4])
    if len(set(names)) != 4 or any('饮' in row['text'] or '喝' in row['text'] for row in cases):
        raise ValueError('Unexpected selection for irrelevant-beverage probe')
    previous = audit.audit.NAMES
    try:
        audit.audit.NAMES = names
        audit.main(cases=[(row['text'], None) for row in cases], output=OUTPUT)
    finally:
        audit.audit.NAMES = previous
    if sha(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError('Archive changed during audit')


if __name__ == '__main__':
    main()
