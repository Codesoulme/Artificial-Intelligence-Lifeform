"""Measure exact full-paragraph containment in an ancestor pretraining pool."""
import collections
import hashlib
import json
from pathlib import Path

import ahocorasick  # Optional local research dependency: pyahocorasick 2.3.1.

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from scripts.prepare_wikimedia_intro_candidate import norm

CANDIDATE = ROOT/'runs/wikimedia-intro-candidate-v2'
ANCESTOR = ROOT/'runs/conversation-data-v5'
OUTPUT = CANDIDATE/'ancestor-overlap.json'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    cm = json.loads((CANDIDATE/'manifest.json').read_text(encoding='utf8'))
    am = json.loads((ANCESTOR/'manifest.json').read_text(encoding='utf8'))
    ancestor_path = ANCESTOR/'pretrain-train.jsonl'
    if sha(ancestor_path) != am['files'][ancestor_path.name]:
        raise ValueError('Ancestor source differs')
    candidates, patterns = {}, collections.defaultdict(list)
    for split in ('train', 'development'):
        path = CANDIDATE/(split+'.jsonl')
        if sha(path) != cm['files'][path.name]:
            raise ValueError('Candidate differs: '+split)
        with path.open(encoding='utf8') as f:
            for line in f:
                row = json.loads(line)
                value = norm(row['text'])
                if len(value) < 24:
                    raise ValueError('Unexpectedly short paragraph')
                candidates[row['group']] = (split, value)
                patterns[value[:24]].append(row['group'])
    automaton = ahocorasick.Automaton()
    for prefix, groups in patterns.items():
        automaton.add_word(prefix, groups)
    automaton.make_automaton()
    matched, documents = set(), 0
    with ancestor_path.open(encoding='utf8') as f:
        for line in f:
            documents += 1
            text = norm(json.loads(line)['text'])
            for _, possible in automaton.iter(text):
                for group in possible:
                    if group not in matched and candidates[group][1] in text:
                        matched.add(group)
    matched_counts = collections.Counter(candidates[group][0] for group in matched)
    report = dict(status='analysis_only_not_approved_for_training', ancestor_file_sha256=sha(ancestor_path),
                  ancestor_manifest_sha256=sha(ANCESTOR/'manifest.json'), ancestor_documents=documents,
                  candidate_manifest_sha256=sha(CANDIDATE/'manifest.json'), evaluator_sha256=sha(Path(__file__)),
                  normalized_full_containment_by_split=matched_counts,
                  candidate_rows_by_split={s: sum(v[0] == s for v in candidates.values()) for s in ('train','development')},
                  scope='Exact full paragraph containment after NFKC, casefold and nonword removal in ancestor source pool. Does not prove the model saw each matched record, nor find paraphrases or semantically similar text.')
    with OUTPUT.open('x', encoding='utf8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
