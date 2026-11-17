"""Build a local-only, deterministic review queue; not training approval."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/oasst2-chinese-audit-v1'
OUT = ROOT / 'runs/oasst-evidence-review-20261118'
EXCLUDE = [ROOT / 'runs/oasst1-chinese-candidate-v5/dialogue.jsonl',
           ROOT / 'runs/oasst2-chinese-candidate-v1/train.jsonl',
           ROOT / 'runs/oasst2-chinese-candidate-v1/development.jsonl']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eligible(row):
    labels = row.get('labels') or {}
    return (row.get('lang') == 'zh' and row.get('review_result') is True
            and row.get('deleted') is False and isinstance(row.get('text'), str)
            and all(labels.get(k, {}).get('value') == 0 for k in ('pii', 'spam', 'hate_speech', 'sexual_content')))


def chains(rows, excluded):
    index = {r['message_id']: r for r in rows}
    candidates = []
    for row in rows:
        if row.get('role') != 'prompter' or not eligible(row):
            continue
        parent = index.get(row.get('parent_id'), {})
        first = index.get(parent.get('parent_id'), {})
        if (parent.get('role') != 'assistant' or first.get('role') != 'prompter'
                or first.get('parent_id') is not None or not eligible(parent) or not eligible(first)):
            continue
        group = row['message_tree_id']
        if group in excluded or any(x.get('message_tree_id') != group for x in (parent, first)):
            continue
        if not 2 <= len(row['text']) <= 300 or len(first['text']) + len(parent['text']) > 1400:
            continue
        candidates.append({'group': group, 'message_ids': [x['message_id'] for x in (first, parent, row)],
                           'messages': [{'role': role, 'content': x['text']} for x, role in
                                        zip((first, parent, row), ('user', 'assistant', 'user'))]})
    candidates.sort(key=lambda r: hashlib.sha256(('|'.join(r['message_ids'])).encode()).hexdigest())
    result, groups = [], set()
    for row in candidates:
        if row['group'] not in groups:
            result.append(row)
            groups.add(row['group'])
    return result, len(candidates)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    provenance = json.loads((SOURCE / 'provenance.json').read_text(encoding='utf8'))
    path = SOURCE / provenance['file']
    if digest(path) != provenance['sha256']:
        raise ValueError('Source mismatch')
    excluded = {r['group'] for p in EXCLUDE for r in
                (json.loads(line) for line in p.read_text(encoding='utf8').splitlines())}
    with gzip.open(path, 'rt', encoding='utf8') as f:
        rows = [json.loads(line) for line in f]
    candidates, count = chains(rows, excluded)
    chosen = candidates[:36]
    for i, row in enumerate(chosen):
        row['case'] = i
    OUT.mkdir()
    with (OUT / 'queue.json').open('x', encoding='utf8') as f:
        json.dump(chosen, f, ensure_ascii=False, indent=2)
    manifest = {'source': provenance, 'exclusion_sha256': {str(p): digest(p) for p in EXCLUDE},
                'candidate_paths': count, 'eligible_trees': len(candidates), 'chosen': len(chosen),
                'queue_sha256': digest(OUT / 'queue.json'), 'status': 'awaiting_agent_annotation_not_training',
                'limits': ['Review labels do not establish truth or complete privacy checks.',
                           'Earlier candidate trees excluded, not all previous corpora or pretraining.',
                           'No future assistant answer is included. Text remains local.']}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k: manifest[k] for k in ('candidate_paths', 'eligible_trees', 'chosen')}))


if __name__ == '__main__':
    main()
