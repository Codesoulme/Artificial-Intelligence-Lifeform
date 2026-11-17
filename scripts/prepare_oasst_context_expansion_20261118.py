"""Split unseen conversation trees before label inspection; raw text stays local."""
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_oasst_evidence_review_20261118 import SOURCE, EXCLUDE, chains, digest

OUT = ROOT/'runs/oasst-context-expansion-20261118'
PREVIOUS = ROOT/'runs/oasst-evidence-review-20261118/queue.json'
RULES = '''Label the current request, not the factual correctness of the answer.
context_need: 1 when correctly fulfilling the whole request requires a referent,
specific prior artifact, or active constraint omitted from the current turn;
0 when the current turn specifies a standalone request or acknowledgment;
null when intent or reference remains genuinely ambiguous even with history.
Generic replies or asking clarification do not prove context_need=0.
history_resolves: true/false/null for whether the original first two messages
resolve that missing referent/constraint, not whether they prove factual truth.
If context_need=0, use history_resolves=null.
quality_flag: short separate note on unreliable prior facts or other scope concerns.
Language is a separate flag, not a context-need label. Do not train on quoted
claims as factual truth. Provide a short rationale. Do not read model scores or
another reviewer's annotation. Exclude null/disagreement from training pending
adjudication. Wrong-history labels must be reviewed separately later.
'''


def split_candidates(candidates, excluded, sizes=(32, 16, 48)):
    ordered = sorted((r for r in candidates if r['group'] not in excluded),
                     key=lambda r: hashlib.sha256(('spora-context-v2:' + r['group']).encode()).hexdigest())
    if len({r['group'] for r in ordered}) != len(ordered):
        raise ValueError('Duplicate tree')
    # Prevent exact current-turn duplication from crossing partitions. This
    # does not remove semantic paraphrases or other shared template patterns.
    unique, texts = [], set()
    for row in ordered:
        text = row['messages'][-1]['content'].strip()
        if text not in texts:
            texts.add(text)
            unique.append(row)
    if len(unique) < sum(sizes):
        raise ValueError('Insufficient distinct candidates')
    result, offset = {}, 0
    for name, size in zip(('train_review', 'calibration_review', 'reserved_holdout'), sizes):
        result[name] = [dict(r, case=offset+i, partition=name) for i, r in enumerate(unique[offset:offset+size])]
        offset += size
    return result


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    provenance = json.loads((SOURCE/'provenance.json').read_text(encoding='utf8'))
    path = SOURCE/provenance['file']
    if digest(path) != provenance['sha256']:
        raise ValueError('Source mismatch')
    excluded = {r['group'] for p in EXCLUDE for r in
                (json.loads(line) for line in p.read_text(encoding='utf8').splitlines())}
    excluded |= {r['group'] for r in json.loads(PREVIOUS.read_text(encoding='utf8'))}
    with gzip.open(path, 'rt', encoding='utf8') as f:
        raw = [json.loads(line) for line in f]
    candidates, count = chains(raw, excluded)
    splits = split_candidates(candidates, excluded)
    OUT.mkdir()
    files = {}
    for name, rows in splits.items():
        p = OUT/(name+'.json')
        p.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf8')
        files[p.name] = digest(p)
    (OUT/'annotation-rules.txt').write_text(RULES, encoding='utf8')
    manifest = {'source': provenance, 'source_script_sha256': digest(Path(__file__)),
                'exclusion_sha256': {str(p): digest(p) for p in EXCLUDE+[PREVIOUS]},
                'candidate_paths': count, 'candidate_trees': len(candidates),
                'files': files, 'rules_sha256': digest(OUT/'annotation-rules.txt'),
                'counts': {k: len(v) for k, v in splits.items()},
                'status': 'split_before_review_not_training_ready',
                'limits': ['Holdout content not inspected by reviewers in this preparation.',
                           'Prior exposures and semantic duplicates not exhaustively ruled out.',
                           'Borrowed histories must stay inside their partition.']}
    (OUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(manifest['counts']))


if __name__ == '__main__':
    main()
