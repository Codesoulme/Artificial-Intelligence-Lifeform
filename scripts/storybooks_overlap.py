"""Local, read-only comparison; requires pyahocorasick in the SPORA venv."""
import collections
import hashlib
import json
import re
import tempfile
import unicodedata
from pathlib import Path

import ahocorasick

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / 'runs'
RESEARCH = ROOT / 'runs/research/storybooks-canada'
SOURCES = {
    'ancestor_v5': ROOT / 'runs/conversation-data-v5/pretrain-train.jsonl',
    'p28_story_train': ROOT / 'runs/p28-stories-data/candidate-v3/pretrain-train.jsonl',
}
WINDOWS = (12, 20)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalize(text):
    return re.sub(r'[\W_]+', '', unicodedata.normalize('NFKC', text).casefold())


def body(raw, paragraphs):
    sections = re.split(r'(?m)^##\s*$', raw)
    if len(sections) < paragraphs + 2:
        raise ValueError('Not enough body sections plus attribution')
    return ''.join(sections[1:1 + paragraphs])


def ranges(positions, width):
    result = []
    for start in sorted(positions):
        end = start + width
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def build(texts):
    patterns = collections.defaultdict(list)
    for index, text in enumerate(texts):
        if not text:
            raise ValueError('Empty candidate')
        patterns[text].append((index, 'full', 0))
        for width in WINDOWS:
            for pos in range(len(text) - width + 1):
                patterns[text[pos:pos + width]].append((index, str(width), pos))
    machine = ahocorasick.Automaton()
    for pattern, refs in patterns.items():
        machine.add_word(pattern, (len(pattern), refs))
    machine.make_automaton()
    return machine


def blank():
    return {'documents_with_full_containment': 0, 'documents_equal_full': 0,
            'windows': {str(w): {'occurrences': 0, 'documents': 0, 'positions': set()} for w in WINDOWS}}


def scan_record(machine, text, results, skip=None):
    seen_full, seen_equal = set(), set()
    seen_windows = {str(w): set() for w in WINDOWS}
    for _, (pattern_length, refs) in machine.iter(text):
        for index, kind, pos in refs:
            if index == skip:
                continue
            if kind == 'full':
                seen_full.add(index)
                if pattern_length == len(text):
                    seen_equal.add(index)
            else:
                item = results[index]['windows'][kind]
                item['occurrences'] += 1
                item['positions'].add(pos)
                seen_windows[kind].add(index)
    for index in seen_full:
        results[index]['documents_with_full_containment'] += 1
    for index in seen_equal:
        results[index]['documents_equal_full'] += 1
    for kind, indexes in seen_windows.items():
        for index in indexes:
            results[index]['windows'][kind]['documents'] += 1


def finish(results):
    for result in results:
        for width, window in result['windows'].items():
            positions = window.pop('positions')
            window['candidate_window_positions'] = len(positions)
            window['candidate_ranges'] = ranges(positions, int(width))
    return results


def create_output_dir(runs=RUNS):
    """Reserve a new directory without touching any existing run."""
    runs.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='storybooks-overlap-', dir=runs))


def write_results(out, output):
    """Never replace an existing result, even if the directory is reused."""
    with (out / 'results.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(output, ensure_ascii=False, indent=2) + '\n')


def main():
    input_files = [RESEARCH / n for n in ('page-text-verification-v2.json', 'quality-review-a.json', 'quality-review-b.json')]
    verification, review_a, review_b = [json.loads(p.read_text(encoding='utf-8')) for p in input_files]
    candidates = [s for s in verification['stories'] if s['status'] == 'matched_pending_quality_and_rights_review']
    reviews = review_a['stories'] + review_b['reviews']
    by_story = {r['story']: r for r in reviews}
    if len(candidates) != 29 or len(reviews) != 29 or len(by_story) != 29 or set(by_story) != {s['story'] for s in candidates}:
        raise ValueError('Candidate/review coverage mismatch')
    texts, entries = [], []
    for item in candidates:
        review = by_story[item['story']]
        path = RESEARCH / 'raw-v1' / (item['story'] + '.md')
        raw = path.read_bytes()
        digest = sha(raw)
        if digest != item['source_sha256'] or digest != review['source_sha256'] or not item['text_matches_page'] or not item['allowed_page_license']:
            raise ValueError('Candidate integrity or verification mismatch')
        text = normalize(body(raw.decode('utf-8-sig'), item['paragraphs']))
        texts.append(text)
        entries.append({'source_sha256': digest, 'normalized_body_sha256': sha(text.encode('utf-8')),
                        'normalized_body_length': len(text), 'quality_mark': review.get('decision', review.get('verdict')),
                        'training_approved': False, 'comparisons': {}})
    machine = build(texts)
    pair_results = [blank() for _ in texts]
    for index, text in enumerate(texts):
        scan_record(machine, text, pair_results, skip=index)
    for entry, result in zip(entries, finish(pair_results)):
        entry['comparisons']['other_candidates'] = result
    source_metadata = {}
    for label, path in SOURCES.items():
        results = [blank() for _ in texts]
        h = hashlib.sha256()
        rows = 0
        with path.open('rb') as stream:
            for line in stream:
                h.update(line)
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record.get('text'), str):
                    raise ValueError('Training record missing text')
                scan_record(machine, normalize(record['text']), results)
                rows += 1
        for entry, result in zip(entries, finish(results)):
            entry['comparisons'][label] = result
        source_metadata[label] = {'path': path.relative_to(ROOT).as_posix(), 'sha256': h.hexdigest(), 'records': rows}
    output = {'verification_inputs': {p.name: sha(p.read_bytes()) for p in input_files},
              'scope': source_metadata, 'candidates': entries}
    out = create_output_dir()
    write_results(out, output)
    print('Output:', out)
    print('Candidates:', len(entries))
    for label in ('other_candidates', *SOURCES):
        print(label, 'full:', sum(e['comparisons'][label]['documents_with_full_containment'] > 0 for e in entries),
              '12-char:', sum(e['comparisons'][label]['windows']['12']['documents'] > 0 for e in entries),
              '20-char:', sum(e['comparisons'][label]['windows']['20']['documents'] > 0 for e in entries))


if __name__ == '__main__':
    main()
