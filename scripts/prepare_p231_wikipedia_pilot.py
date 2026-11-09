"""Prepare a restricted local Wikipedia diagnostic candidate, without training."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.p115_encode_stories import identities
from tokenizers import Tokenizer, __version__ as tokenizers_version

CONTEXT = Path('runs/p231-wikipedia-context-20261109')
AUDIT = Path('runs/p231-wikipedia-ancestor-overlap-v1-20261109.json')
TOKENIZER = Path('runs/conversation-data-v5/tokenizer.json')
OUTPUT = Path('runs/p231-wikipedia-pilot-candidate-20261109')
SAMPLES = [Path('runs/p230-wikipedia-body-sample-20261109/sample.json'),
           Path('runs/p230-wikipedia-body-refined-v1-20261109/sample.json'),
           CONTEXT / 'sample.json']
PINS = {
    str(CONTEXT / 'manifest.json'): '38a4acf2ca6bbe95c65eafb3447b2c21a0d9580ae0d232f55d4806993192cdd8',
    str(CONTEXT / 'candidates.jsonl'): 'a551a761f511fe3b8f8d04168e7778ffea0b722e766b6a6f54ab317290832784',
    str(AUDIT): 'a5b6103fd7fa36b555aa3cc3b405d6245b40d24aebfc230932cf0f073a05aa8a',
    str(TOKENIZER): 'b3629001784896437c4aa9f279f6b4cdc5a534be191b537009e7293f971fda20',
    str(SAMPLES[0]): '66f7d4f32130f83607a54c49fc3b3d2e3df60a8c979515aec8d15a58908e60fb',
    str(SAMPLES[1]): 'c0c9868d5dd153636a7cf33240089781f22d147d36259e27742cedd3b6034fee',
    str(SAMPLES[2]): 'a9e4b43bfaeacebfa29f36371261f05c1bd531dfe4dcf4840fc047d84f57b312',
}
SEED = 'p231-wikipedia-pilot-20261109-v1'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_verified(root, path, pins=PINS):
    raw = (root / path).read_bytes()
    if digest(raw) != pins[str(path)]:
        raise ValueError('Digest mismatch: ' + str(path))
    return raw


def sample_pages(samples):
    rounds = []
    for sample in samples:
        if not isinstance(sample, list) or len(sample) != 40:
            raise ValueError('Expected 40 records per sample')
        pages = [r['page_id'] for r in sample]
        if any(not isinstance(p, str) or not p.isdecimal() for p in pages) or len(set(pages)) != 40:
            raise ValueError('Invalid sample pages')
        rounds.append(set(pages))
    if len(set.union(*rounds)) != 120:
        raise ValueError('Sample pages overlap')
    return set.union(*rounds)


def audit_pages(audit, manifest):
    if (audit['candidate_manifest_sha256'] != manifest['input_sha256']['refined_manifest']
            or audit['candidate_sha256'] != manifest['refined_candidate_sha256']
            or len(audit['candidates']) != audit['candidate_count']):
        raise ValueError('Audit provenance or count mismatch')
    seen, matched = set(), set()
    for row in audit['candidates']:
        key = row['candidate_id']
        parts = key.split(':')
        if len(parts) != 3 or not all(p.isdecimal() for p in parts) or key in seen:
            raise ValueError('Invalid audit candidate identity')
        seen.add(key)
        counts = row['sources']
        if set(counts) != set(audit['sources']) or any(type(n) is not int or n < 0 for n in counts.values()):
            raise ValueError('Invalid audit source counts')
        if any(counts.values()):
            matched.add(key)
    if len(matched) != audit['matched_any_source']:
        raise ValueError('Audit match count mismatch')
    return {key.split(':')[0] for key in matched}


def select(groups, excluded, tokenizer):
    stats = Counter()
    selected = []
    seen_pages, seen_spans = set(), set()
    for group in groups:
        pid, rows = group['page_id'], group['candidates']
        if not isinstance(pid, str) or not pid.isdecimal() or pid in seen_pages or not rows:
            raise ValueError('Invalid page group')
        seen_pages.add(pid)
        eligible = []
        for row in rows:
            stats['input_passages'] += 1
            start, end = row['char_start'], row['char_end']
            key = (pid, start, end)
            if (row['page_id'] != pid or row['status'] != 'pending_review_not_approved_for_training'
                    or type(start) is not int or type(end) is not int or not 0 <= start < end
                    or key in seen_spans or len(row['text']) != end - start
                    or digest(row['text'].encode('utf-8')) != row['text_sha256']):
                raise ValueError('Invalid candidate identity or body')
            seen_spans.add(key)
            section = row['section']
            labels = [row['title']]
            if section and section['text'] != row['title']:
                labels.append(section['text'])
            prefix = '\n'.join(labels) + '\n\n'
            if row['model_text'] != prefix + row['text']:
                raise ValueError('Model text differs from complete body')
            counts = [len(tokenizer.encode(text, add_special_tokens=False).ids)
                      for text in (row['text'], prefix, row['model_text'])]
            if counts != [row['body_bpe_tokens'], row['extra_heading_bpe_tokens'], row['model_bpe_tokens']]:
                raise ValueError('Token count mismatch')
            if row['model_minus_body_bpe_tokens'] != counts[2] - counts[0]:
                raise ValueError('Token difference mismatch')
            if counts[0] >= 64 and counts[2] <= 510:
                rank = digest(f'{SEED}\0span\0{pid}\0{start}\0{end}'.encode())
                eligible.append((rank, start, end, row))
        if pid in excluded:
            stats['excluded_input_pages'] += 1
        elif eligible:
            selected.append((digest(f'{SEED}\0page\0{pid}'.encode()), pid, min(eligible)[3]))
        else:
            stats['pages_without_eligible_passage'] += 1
    kept, full_seen, group_seen = [], set(), set()
    for _, pid, row in sorted(selected):
        full, group = identities(row['model_text'])
        if full in full_seen or group in group_seen:
            stats['identity_duplicates_removed'] += 1
            continue
        full_seen.add(full)
        group_seen.add(group)
        kept.append(dict(row, text=row['model_text'], body_text=row['text'], sha256=full, group=group,
                         source_original_sha256=digest(row['model_text'].encode('utf-8')),
                         source_row_1based=len(kept) + 1,
                         status='local_restricted_diagnostic_candidate'))
    stats['input_pages'] = len(seen_pages)
    stats['output_pages'] = len(kept)
    return kept, dict(stats)


def build(root=ROOT):
    root = Path(root)
    output = root / OUTPUT
    if output.exists():
        raise FileExistsError(output)
    raw = {path: read_verified(root, path) for path in [CONTEXT / 'manifest.json', CONTEXT / 'candidates.jsonl',
                                                        AUDIT, TOKENIZER, *SAMPLES]}
    manifest, audit = json.loads(raw[CONTEXT / 'manifest.json']), json.loads(raw[AUDIT])
    if (manifest['files']['candidates.jsonl'] != PINS[str(CONTEXT / 'candidates.jsonl')]
            or manifest['files']['sample.json'] != PINS[str(SAMPLES[2])]
            or manifest['review_sample_sha256'] != [PINS[str(p)] for p in SAMPLES[:2]]
            or manifest['input_sha256']['tokenizer'] != PINS[str(TOKENIZER)]):
        raise ValueError('Context manifest provenance mismatch')
    samples = [json.loads(raw[p]) for p in SAMPLES]
    excluded_samples = sample_pages(samples)
    excluded_audit = audit_pages(audit, manifest)
    tokenizer = Tokenizer.from_str(raw[TOKENIZER].decode('utf-8'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    groups = [json.loads(line) for line in raw[CONTEXT / 'candidates.jsonl'].splitlines()]
    rows, stats = select(groups, excluded_samples | excluded_audit, tokenizer)
    if (stats['input_pages'] != manifest['counts']['candidate_pages']
            or stats['input_passages'] != manifest['counts']['candidate_passages']):
        raise ValueError('Context count mismatch')
    candidate = b''.join((json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8') for r in rows)
    result = {
        'status': 'local_restricted_diagnostic_candidate', 'trained': False,
        'input_sha256': PINS, 'script_sha256': digest(Path(__file__).read_bytes()),
        'source_repository': manifest['source_repository'], 'revision': manifest['revision'],
        'license': manifest['license'], 'offsets': manifest['offsets'],
        'sampling': {'seed': SEED, 'method': 'one lowest hash eligible complete passage per page; pages hash ordered; deduplicate normalized full and prefix identities',
                     'body_bpe_min': 64, 'model_bpe_max': 510, 'special_tokens_included': False},
        'tokenizer_version': tokenizers_version,
        'exclusions': {'sample_pages': len(excluded_samples), 'audit_matched_pages': len(excluded_audit),
                       'audit_and_sample_overlap': len(excluded_samples & excluded_audit)},
        'counts': stats, 'model_bpe_tokens': sum(r['model_bpe_tokens'] for r in rows),
        'files': {'candidate.jsonl': digest(candidate)},
        'review': {'sample_pages': 40, 'major': 5, 'minor': 22, 'model_reviewers': 2,
                   'pages_per_reviewer': 20, 'note': 'Model judgments on different pages, not independent double review or human ground truth.'},
        'limitations': ['Restricted local diagnostic candidate; no quality, rights or general training clearance is claimed.',
                        'Human factual and rights review and historical approximate-overlap checks are incomplete.',
                        'Audit covers normalized full containment against two specified historical sources only; partial and semantic overlap may remain.',
                        'Sample is page weighted and is not a passage weighted estimate; reviewed pages are excluded.',
                        'Article attribution and applicable license obligations require separate review.']}
    output.mkdir(parents=False, exist_ok=False)
    with (output / 'candidate.jsonl').open('xb') as f:
        f.write(candidate)
    with (output / 'manifest.json').open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    return result


if __name__ == '__main__':
    print(json.dumps(build(), ensure_ascii=False, indent=2))
