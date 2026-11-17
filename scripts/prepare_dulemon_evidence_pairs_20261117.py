"""Build local evidence-before/after pairs without publishing source dialogue.

Selected records are manually reviewed development examples, not an
independent benchmark or proof of cross-session persistent memory.
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts.deepseek_dulemon_persona_audit import normalize_turn
from scripts.deepseek_dulemon_routing_audit import ARCHIVE, ARCHIVE_SHA
from scripts.open_base_baseline import sha

MANIFEST = ROOT / 'configs/dulemon_evidence_pairs_20261117.json'
OUT = ROOT / 'runs/dulemon-evidence-pairs-development-20261117.json'


def build_pairs(dialogues, items):
    if not isinstance(items, list) or not items:
        raise ValueError('Nonempty annotations required')
    output, seen = [], set()
    for item in items:
        session, turn = item['session'], item['evidence_turn']
        if (type(session) is not int or type(turn) is not int or session < 0
                or session >= len(dialogues) or session in seen or turn < 1
                or item['split'] != 'development' or not item['question'].strip()):
            raise ValueError('Invalid or duplicate session')
        seen.add(session)
        conversation = dialogues[session]['conversation']
        if turn >= len(conversation):
            raise ValueError('Evidence outside conversation')
        normalized = [normalize_turn(x) for x in conversation[:turn+1]]
        if any(x is None or not x[1] for x in normalized):
            raise ValueError('Invalid conversation turn')
        role, text, _ = normalized[-1]
        bounds = item['span']
        if (role != item['speaker'] or not isinstance(bounds, list) or len(bounds) != 2
                or any(type(x) is not int for x in bounds) or not 0 <= bounds[0] < bounds[1] <= len(text)):
            raise ValueError('Invalid evidence span or speaker')
        answer = text[bounds[0]:bounds[1]]
        earlier = normalized[:-1]
        if any(answer in row[1] for row in earlier):
            raise ValueError('Answer already occurs in prior context')
        # Prevent synthetic question from leaking the evidence answer.
        if answer in item['question']:
            raise ValueError('Answer leaked in question')
        previous = [{'speaker': x[0], 'text': x[1]} for x in earlier]
        evidence = {'speaker': role, 'text': text}
        source_digest = hashlib.sha256(json.dumps(normalized, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()
        for condition, context, expected in (('before', previous, None),
                                              ('after', previous + [evidence], answer)):
            output.append({'session': session, 'evidence_turn': turn, 'split': item['split'],
                           'condition': condition, 'question': item['question'], 'context': context,
                           'answer': expected, 'evidence_speaker': role, 'source_sha256': source_digest})
    return output


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    archive_sha, manifest_sha = sha(ARCHIVE), sha(MANIFEST)
    if archive_sha != ARCHIVE_SHA:
        raise ValueError('Archive digest mismatch')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    with zipfile.ZipFile(ARCHIVE) as archive, archive.open(manifest['source_member']) as stream:
        dialogues = [json.loads(line) for line in stream if line.strip()]
    rows = build_pairs(dialogues, manifest['items'])
    if sha(ARCHIVE) != archive_sha or sha(MANIFEST) != manifest_sha:
        raise ValueError('Source changed')
    result = {'status': 'manually_selected_single_annotator_development_only',
              'source_archive_sha256': archive_sha, 'manifest_sha256': manifest_sha,
              'source_member': manifest['source_member'], 'rows': rows,
              'limitations': ['Selected and annotated by one model; no independent human review.',
                              'Before evidence answer text absent does not prove absence of synonymous evidence.',
                              'Synthetic questions; dialogue context is not an internal memory state.',
                              'All records are development data, not an independent final evaluation.',
                              'Source data rights not individually confirmed; keep output local.']}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'sessions': len(manifest['items']), 'pairs': len(rows)//2,
                      'conditions': len(rows), 'source_sha256': archive_sha}))


if __name__ == '__main__':
    main()
