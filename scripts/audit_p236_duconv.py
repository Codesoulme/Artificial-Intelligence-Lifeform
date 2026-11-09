"""Inspect only the training member of a pinned DuConv archive; no training."""
import hashlib
import heapq
import json
import unicodedata
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from scripts.p104_screen import PATTERNS

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT/'runs/p236-duconv-source-20261109/DuConv.zip'
OUTPUT = ROOT/'runs/p236-duconv-audit-20261109'
ARCHIVE_SHA = 'f4f60d9363668e52eb3fd9db3d5c991d7be48f73ca7f15526629212427ae2b01'
SEED = 'p236-duconv-training-review-v1'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def conversation_key(conversation):
    encoded = json.dumps([unicodedata.normalize('NFKC', text).casefold().strip() for text in conversation],
                         ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def audit(archive, sample_size=30):
    counts, lengths, keys, sizes, risk = Counter(), Counter(), Counter(), [], Counter()
    seen, sample, train_hash = set(), [], hashlib.sha256()
    with ZipFile(archive) as zipped:
        matches = [info for info in zipped.infolist() if info.filename == 'DuConv/train.txt']
        if len(matches) != 1 or matches[0].file_size > 100_000_000 or matches[0].is_dir():
            raise ValueError('Invalid training member')
        with zipped.open(matches[0]) as stream:
            for line_no, raw in enumerate(stream, 1):
                train_hash.update(raw)
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError('Invalid record')
                counts['rows'] += 1
                keys.update(row.keys())
                conversation = row.get('conversation')
                if (not isinstance(conversation, list) or not conversation
                        or any(not isinstance(turn, str) or not turn.strip() for turn in conversation)):
                    counts['invalid_conversation'] += 1
                    continue
                counts['valid_conversations'] += 1
                lengths[len(conversation)] += 1
                text = '\n'.join(conversation)
                sizes.append(len(text))
                for name, pattern in PATTERNS.items():
                    risk[name] += bool(pattern.search(unicodedata.normalize('NFKC', text)))
                identity = conversation_key(conversation)
                if identity in seen:
                    counts['exact_duplicate_conversations'] += 1
                    continue
                seen.add(identity)
                rank = int(hashlib.sha256((SEED+'\0'+identity).encode()).hexdigest(), 16)
                entry = (-rank, identity, line_no, row)
                if len(sample) < sample_size:
                    heapq.heappush(sample, entry)
                elif entry > sample[0]:
                    heapq.heapreplace(sample, entry)
    counts['unique_conversations'] = len(seen)
    if len(sample) != sample_size or counts['rows'] != counts['valid_conversations']+counts['invalid_conversation']:
        raise ValueError('Incomplete sample or inconsistent counts')
    sample.sort(reverse=True)
    output_rows = [{'index': index, 'source_line': line_no, 'conversation_sha256': identity,
                    'conversation': row['conversation'], 'goal': row.get('goal'), 'knowledge': row.get('knowledge')}
                   for index, (_, identity, line_no, row) in enumerate(sample)]
    return {'train_sha256': train_hash.hexdigest(), 'counts': dict(counts), 'keys': dict(keys),
            'conversation_turns': dict(sorted(lengths.items())), 'conversation_characters': {
                'min': min(sizes), 'median': sorted(sizes)[len(sizes)//2], 'max': max(sizes)},
            'heuristic_flagged_conversations': dict(risk)}, output_rows


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError('Archive digest mismatch')
    result, sample = audit(ARCHIVE)
    if sha(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError('Archive changed during audit')
    payload = b''.join((json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n').encode('utf-8') for row in sample)
    result.update(status='review_only_not_training', archive_sha256=ARCHIVE_SHA,
                  source_url='https://bj.bcebos.com/paddlenlp/datasets/DuConv.zip',
                  seed=SEED, sample_size=len(sample), sample_sha256=hashlib.sha256(payload).hexdigest(),
                  script_sha256=sha(Path(__file__)),
                  limitations=['Only archive member DuConv/train.txt was decoded.',
                      'Exact normalized conversation identity does not detect partial or semantic duplicates.',
                      'Heuristics flag review candidates; they do not establish privacy, safety or quality.',
                      'Crowd participants are not automatically reliable assistant demonstrations.',
                      'Source rights and overlap with ancestors are not established.'])
    OUTPUT.mkdir(parents=False, exist_ok=False)
    (OUTPUT/'sample.jsonl').write_bytes(payload)
    (OUTPUT/'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
