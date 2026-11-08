"""Reproducible, local-only P110 candidate sample from the fixed training JSONL."""
import collections
import hashlib
import json
import random
import re
from pathlib import Path

def signals(text):
    """P110 predicates, with an inexpensive guard for the email pattern."""
    s = text.strip()
    question = bool(re.search(r'[？?]', s))
    dialogue = bool(re.search(r'(?:^|\n)\s*(?:问|答|用户|助手|提问|回答|Q|A)\s*[:：]', s, re.I))
    defect = (len(s) < 50 or '\ufffd' in s or bool(re.search(r'(.)\1{9,}', s))
              or not bool(re.search(r'[。！？!?…」』”"）)]\s*$', s)))
    privacy = bool(re.search(r'(?<!\d)1[3-9]\d{9}(?!\d)|(?<!\d)\d{17}[\dXx](?!\d)', s))
    if not privacy and '@' in s:
        privacy = bool(re.search(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', s))
    return {'privacy_cues': privacy, 'defect_cues': defect,
            'prose_candidate': not question and not dialogue and len(s) >= 100
                               and len(re.findall(r'[。！？]', s)) >= 3}


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'runs/conversation-data-v5/pretrain-train.jsonl'
OUTPUT = ROOT / 'runs/p111-prose-sample'
EXPECTED = '4ec0b001db65b7ccabfaa064e56ce6dff3bbf0bb0a3890c116f4b99999bae9a1'
SEED = 11120261109
SIZE = 160


def quotas(counts, size):
    total = sum(counts.values())
    if size <= 0 or total < size:
        raise ValueError('insufficient eligible groups or invalid sample size')
    result = {b: size * n // total for b, n in counts.items()}
    order = sorted(counts, key=lambda b: (-(size * counts[b] % total), b))
    for b in order[:size - sum(result.values())]:
        result[b] += 1
    return result


def run(input_path, output_dir, expected, size=SIZE, seed=SEED):
    """Write only after validating all input and hash; never overwrite output."""
    input_path, output_dir = Path(input_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError('output directory already exists')
    digest = hashlib.sha256()
    # First qualifying record is the representative; all records can quarantine its group.
    candidates = {}
    bad_groups = set()
    totals = collections.Counter()
    rejected = collections.Counter()
    by_bucket = collections.defaultdict(collections.Counter)
    with input_path.open('rb') as source:
        while True:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            digest.update(line)
            try:
                row = json.loads(line)
                text, group, bucket = row['text'], row['group'], row['source_byte_bucket']
                if not isinstance(text, str) or not isinstance(group, str) or not group or type(bucket) is not int or bucket not in range(10):
                    raise ValueError('invalid fields')
            except (ValueError, TypeError, KeyError, UnicodeError) as exc:
                raise ValueError('invalid input record') from exc
            flags = signals(text)
            totals['records'] += 1
            by_bucket[bucket]['records'] += 1
            for reason in ('privacy_cues', 'defect_cues'):
                if flags[reason]:
                    rejected[reason] += 1
                    by_bucket[bucket][reason] += 1
            if flags['privacy_cues'] or flags['defect_cues']:
                bad_groups.add(group)
            if not flags['prose_candidate']:
                rejected['not_prose_candidate'] += 1
                by_bucket[bucket]['not_prose_candidate'] += 1
            elif not flags['privacy_cues'] and not flags['defect_cues']:
                totals['rule_candidates'] += 1
                by_bucket[bucket]['rule_candidates'] += 1
                if group not in candidates:
                    candidates[group] = (bucket, offset)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError('training JSONL digest mismatch')
        for group in bad_groups:
            candidates.pop(group, None)
        strata = collections.defaultdict(list)
        for group, (bucket, offset) in candidates.items():
            strata[bucket].append((group, offset))
        counts = {b: len(strata[b]) for b in range(10)}
        allocation = quotas(counts, size)
        rng = random.Random(seed)
        chosen = []
        for bucket in range(10):
            entries = sorted(strata[bucket])
            for group, offset in rng.sample(entries, allocation[bucket]):
                chosen.append((bucket, group, offset))
            by_bucket[bucket]['eligible_groups'] = counts[bucket]
            by_bucket[bucket]['sampled_groups'] = allocation[bucket]
        # Recheck the same open file before creating outputs, catching changes during selection.
        source.seek(0)
        check = hashlib.sha256()
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            check.update(chunk)
        if check.hexdigest() != expected:
            raise ValueError('training JSONL changed during sampling')
        manifest = {'version': 1, 'input_sha256': actual, 'seed': seed, 'sample_size': size,
                    'records': totals['records'], 'rule_candidates': totals['rule_candidates'],
                    'quarantined_groups': len(bad_groups), 'eligible_groups': sum(counts.values()),
                    'rejections_overlapping': dict(sorted(rejected.items())),
                    'buckets': {str(b): dict(sorted(by_bucket[b].items())) for b in range(10)}}
        output_dir.mkdir(parents=False, exist_ok=False)
        try:
            with (output_dir / 'sample.jsonl').open('xb') as out:
                for _, _, offset in chosen:
                    source.seek(offset)
                    line = source.readline()
                    out.write(line if line.endswith(b'\n') else line + b'\n')
            with (output_dir / 'manifest.json').open('x', encoding='utf-8', newline='\n') as out:
                json.dump(manifest, out, ensure_ascii=True, sort_keys=True, indent=2)
                out.write('\n')
        except Exception:
            for name in ('sample.jsonl', 'manifest.json'):
                (output_dir / name).unlink(missing_ok=True)
            output_dir.rmdir()
            raise
    return manifest


def main():
    manifest = run(INPUT, OUTPUT, EXPECTED)
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()

