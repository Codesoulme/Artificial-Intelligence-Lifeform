"""Prepare P104 candidates for local quality review only; no model training."""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from p104_screen import assess, norm
from prepare_conversation_v2 import encode_dialogue

SOURCE = ROOT / 'runs/conversation-data-v5/dialogue-train.jsonl'
INDEX = ROOT / 'reports/P104-multiturn-screen.json'
SUMMARY = ROOT / 'runs/p117-dialogue-link-review-v2/summary.json'
FAMILIES = ROOT / 'runs/p117-dialogue-link-review-v2/proposed-families.json'
TOKENIZER = ROOT / 'runs/conversation-data-v5/tokenizer.json'
OUT = ROOT / 'runs/p231-dialogue-candidates'
EXPECTED = {
    'source': '2629f0c8668ef6736c2a0431f0681c24944299e5d60c8ae8ac78d688b565973e',
    'index': 'dbcb85e3c39de838026faa536cbadfacbd2d1c74605f3d72613f72af316b6508',
    'summary': '5ebde92368c38d5f893cfc912df3d45c5e91eff26d16c25e5de819415a1ce791',
    'families': '4b055582fc307142db1f052ee91e271c36f4e8688506b6553bc4a5ea59308802',
    'tokenizer': 'b3629001784896437c4aa9f279f6b4cdc5a534be191b537009e7293f971fda20',
}
SEED = 'p231-dialogue-review-20261109-v1'
STATUS = 'pending_quality_review_not_approved_for_training'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def validate_index(index, summary, families):
    if index['source_sha256'] != EXPECTED['source'] or summary['input_sha256']['training'] != EXPECTED['source'] or summary['input_sha256']['p104_index'] != EXPECTED['index']:
        raise ValueError('input lineage mismatch')
    by_group = {}
    family_states = defaultdict(set)
    changes = Counter()
    for entry in families:
        group, family = entry['group_sha256'], entry['family_sha256']
        original, proposed = entry['original'], entry['proposed']
        if group in by_group or original not in {'candidate', 'development', 'quarantine'}:
            raise ValueError('invalid family mapping')
        if proposed != original and (original, proposed) != ('candidate', 'quarantine'):
            raise ValueError('invalid state transition')
        by_group[group] = entry
        family_states[family].add(proposed)
        if original != proposed:
            changes[original + '->' + proposed] += 1
    if dict(changes) != summary['proposed_changes'] or dict(changes) != {'candidate->quarantine': 4} or len(family_states) != summary['family_count']:
        raise ValueError('family summary mismatch')
    if any('candidate' in states and len(states) > 1 for states in family_states.values()):
        raise ValueError('candidate family overlaps held or quarantine')
    selected = {}
    for row in index['rows']:
        line, group = row['line'], row['group_sha256']
        if type(line) is not int or line < 1 or line in selected or group not in by_group:
            raise ValueError('invalid source index')
        entry = by_group[group]
        if row['partition'] != entry['original'] or row['partition'] not in {'candidate', 'development'}:
            raise ValueError('index partition mismatch')
        selected[line] = entry
    for part in ('candidate', 'development'):
        rows = [r for r in index['rows'] if r['partition'] == part]
        if len(rows) != index['counts'][part + '_rows'] or len({r['group_sha256'] for r in rows}) != index['counts'][part + '_groups']:
            raise ValueError('index count mismatch')
    return selected


def dialogue_hash(messages):
    # P117 normalizes each turn before comparing complete sequences.
    seq = [(m['role'], norm(m['content'])) for m in messages]
    return digest(json.dumps(seq, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))


def prepare_rows(rows, tokenizer):
    counts = Counter()
    unique = {}
    for line, entry, row in rows:
        counts['candidate_rows_examined'] += 1
        flags = assess(row)
        if flags or len(row.get('messages', [])) < 4:
            counts['risk_or_shape_excluded'] += 1
            counts.update('risk_' + x for x in (flags or ['single_turn']))
            continue
        messages = row['messages']
        key = (entry['family_sha256'], dialogue_hash(messages))
        origin = {'line': line, 'p104_group_sha256': entry['group_sha256'],
                  'source_group': row['group'], 'source_row_sha256': row.get('sha256'),
                  'source_line': row.get('source_line')}
        if key in unique:
            unique[key]['origins'].append(origin)
            counts['same_family_duplicate_rows'] += 1
            continue
        unique[key] = {'status': STATUS, 'family_sha256': key[0], 'dialogue_sha256': key[1],
                       'messages': messages, 'origins': [origin], 'risk_flags': flags}
    result = []
    for item in unique.values():
        encoded = encode_dialogue(tokenizer, item['messages'], limit=512)
        if encoded is None:
            counts['unique_over_512_excluded'] += 1
            counts['rows_in_over_512_dialogues'] += len(item['origins'])
            continue
        tokens, mask = encoded
        if len(tokens) > 512 or len(tokens) != len(mask) or not any(mask):
            raise ValueError('invalid encoding')
        item.update(tokens=tokens, mask=mask, token_count=len(tokens), supervised_tokens=sum(mask),
                    supervision='assistant content and end-of-message tokens')
        result.append(item)
    counts['unique_dialogues_before_length_filter'] = len(unique)
    counts['retained_dialogues'] = len(result)
    counts['retained_source_rows'] = sum(len(x['origins']) for x in result)
    counts['retained_families'] = len({x['family_sha256'] for x in result})
    return result, counts


def select_review(rows, count=40):
    families = defaultdict(list)
    for row in rows:
        families[row['family_sha256']].append(row)
    if len(families) < count:
        raise ValueError('not enough retained families for review')
    ranked = sorted(families, key=lambda f: digest((SEED + '\0family\0' + f).encode()))
    return [min(families[f], key=lambda r: digest((SEED + '\0dialogue\0' + r['dialogue_sha256']).encode())) for f in ranked[:count]]


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = {'source': SOURCE, 'index': INDEX, 'summary': SUMMARY, 'families': FAMILIES, 'tokenizer': TOKENIZER}
    for name, path in paths.items():
        if file_hash(path) != EXPECTED[name]:
            raise ValueError(name + ' digest mismatch')
    index = json.loads(INDEX.read_bytes())
    summary = json.loads(SUMMARY.read_bytes())
    families = json.loads(FAMILIES.read_bytes())
    selected = validate_index(index, summary, families)
    candidates = {n: e for n, e in selected.items() if e['proposed'] == 'candidate'}
    rows = []
    hasher = hashlib.sha256()
    line_count = 0
    with SOURCE.open('rb') as stream:
        for line_count, raw in enumerate(stream, 1):
            hasher.update(raw)
            if line_count in candidates:
                rows.append((line_count, candidates[line_count], json.loads(raw)))
    if hasher.hexdigest() != EXPECTED['source'] or line_count != index['source_rows'] or len(rows) != len(candidates):
        raise ValueError('source changed or index out of bounds')
    result, counts = prepare_rows(rows, Tokenizer.from_file(str(TOKENIZER)))
    review = select_review(result)
    counts.update(p104_candidate_rows=index['counts']['candidate_rows'],
                  p104_candidate_groups=index['counts']['candidate_groups'],
                  p104_development_rows=index['counts']['development_rows'],
                  candidate_to_quarantine_groups=4,
                  candidate_to_quarantine_rows=sum(e['original'] == 'candidate' and e['proposed'] == 'quarantine' for e in selected.values()),
                  review_dialogues=len(review))
    OUT.mkdir(parents=True, exist_ok=False)
    for name, data in [('candidates.jsonl', result), ('review-40.jsonl', review)]:
        with (OUT / name).open('x', encoding='utf-8', newline='\n') as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    manifest = {'status': STATUS, 'input_sha256': EXPECTED, 'script_sha256': file_hash(Path(__file__)),
                'encoder_sha256': file_hash(ROOT / 'scripts/prepare_conversation_v2.py'),
                'risk_checker_sha256': file_hash(ROOT / 'scripts/p104_screen.py'),
                'counts': dict(counts), 'review_seed': SEED,
                'review_sampling': 'One retained complete dialogue per family, hash-ranked families and dialogue; prior reviews may have covered some samples.',
                'deduplication': 'Within family, complete role/content sequence normalized with NFKC, casefold and whitespace removal; all source origins retained.',
                'limitations': ['Quality review only; not approved for training.', 'Risk checks are heuristic, not proof of safety, accuracy, privacy or rights.', 'Family isolation relies on P117-v2; semantic overlap may remain.', 'Source file bytes are hashed in full; only selected candidate lines are decoded and parsed.'],
                'output_sha256': {name: file_hash(OUT / name) for name in ('candidates.jsonl', 'review-40.jsonl')}}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'counts': dict(counts)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
