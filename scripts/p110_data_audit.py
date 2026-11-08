"""Aggregate-only audit of the fixed old training partition; never emit records."""
import collections
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'runs/conversation-data-v5'
SEED = 11020261109
SIZE = 4096
EXPECTED = {'pretrain-train.jsonl': '4ec0b001db65b7ccabfaa064e56ce6dff3bbf0bb0a3890c116f4b99999bae9a1',
            'pretrain-train.bin': '781088fbcc6c81650d1e01b0e49a7e2a7393034e88399e585f6487073f10acdf'}


def signals(text):
    """Conservative observable cues, not semantic or rights classifications."""
    s = text.strip()
    question = bool(re.search(r'[？?]', s))
    dialogue = bool(re.search(r'(?:^|\n)\s*(?:问|答|用户|助手|提问|回答|Q|A)\s*[:：]', s, re.I))
    template = bool(re.search(r'(?:首先|其次|最后|第一[，、：]|第二[，、：]|综上所述)', s))
    defect = (len(s) < 50 or '\ufffd' in s or bool(re.search(r'(.)\1{9,}', s))
              or not bool(re.search(r'[。！？!?…」』”"）)]\s*$', s)))
    privacy = bool(re.search(r'(?<!\d)1[3-9]\d{9}(?!\d)|(?<!\d)\d{17}[\dXx](?!\d)|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', s))
    prose = not question and not dialogue and len(s) >= 100 and len(re.findall(r'[。！？]', s)) >= 3
    return {'question_mark': question, 'role_marker': dialogue, 'template_cues': template,
            'defect_cues': defect, 'privacy_cues': privacy, 'prose_candidate': prose}


def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def audit(path, size=SIZE, seed=SEED):
    rng = random.Random(seed)
    sample = []
    digests = collections.Counter()
    buckets = collections.Counter()
    group_counts = collections.Counter()
    raw_hash = hashlib.sha256()
    with path.open('rb') as f:
        for n, line in enumerate(f, 1):
            raw_hash.update(line)
            row = json.loads(line)
            digests[row['sha256']] += 1
            group_counts[row['group']] += 1
            buckets[row['source_byte_bucket']] += 1
            if n <= size:
                sample.append(row)
            else:
                j = rng.randrange(n)
                if j < size:
                    sample[j] = row
    stats = collections.Counter()
    sample_buckets = collections.Counter()
    char_weights = collections.Counter()
    strata = collections.Counter()
    lengths = []
    for row in sample:
        s = row['text']
        lengths.append(len(s))
        flags = signals(s)
        stats.update(k for k, v in flags.items() if v)
        char_weights.update({k: len(s) for k, v in flags.items() if v})
        sample_buckets[row['source_byte_bucket']] += 1
        if flags['question_mark'] or flags['role_marker']:
            strata['question_or_role_cue'] += 1
        elif flags['prose_candidate']:
            strata['prose_candidate_without_question_or_role'] += 1
        else:
            strata['other_or_unresolved'] += 1
        stats['duplicate_digest_in_population'] += digests[row['sha256']] > 1
        stats['shared_group_in_population'] += group_counts[row['group']] > 1
        # Match the producer's normalized digest, rather than trusting a field.
        normalized = re.sub(r'\s+', '', unicodedata.normalize('NFKC', s)).lower()
        stats['digest_mismatch'] += hashlib.sha256(normalized.encode()).hexdigest() != row['sha256']
    lengths.sort()
    return {'file_sha256': raw_hash.hexdigest(), 'population': sum(buckets.values()),
            'sample_size': len(sample), 'seed': seed, 'signals': dict(stats),
            'exclusive_cue_groups': dict(strata), 'sample_buckets': dict(sorted(sample_buckets.items())),
            'sample_characters': sum(lengths), 'signal_characters': dict(char_weights),
            'bucket_population': dict(sorted(buckets.items())),
            'unique_declared_digests': len(digests), 'duplicate_excess': sum(v-1 for v in digests.values()),
            'unique_groups': len(group_counts),
            'length_percentiles': {str(p): lengths[(len(lengths)-1)*p//100] for p in (0, 25, 50, 75, 95, 100)}}


def main():
    result = audit(DATA / 'pretrain-train.jsonl')
    if result['file_sha256'] != EXPECTED['pretrain-train.jsonl']:
        raise ValueError('training JSONL digest mismatch')
    binary_hash = digest_file(DATA / 'pretrain-train.bin')
    if binary_hash != EXPECTED['pretrain-train.bin']:
        raise ValueError('training binary digest mismatch')
    result['binary_sha256'] = binary_hash
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
