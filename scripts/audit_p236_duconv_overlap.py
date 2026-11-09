"""Compare pinned DuConv train conversations with one pinned historical train source."""
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from scripts.audit_p236_duconv import ARCHIVE, ARCHIVE_SHA, sha
from scripts.prepare_p231_dialogue_candidates import EXPECTED

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT/'runs/conversation-data-v5/dialogue-train.jsonl'
OUTPUT = ROOT/'runs/p236-duconv-overlap-20261109.json'


def norm(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text).casefold())


def identity(turns):
    return hashlib.sha256(json.dumps([norm(t) for t in turns], ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def compare(history_rows, candidate_rows):
    old_conversations, old_first, old_turns = set(), set(), set()
    totals = Counter()
    for row in history_rows:
        messages = row.get('messages')
        if (not isinstance(messages, list) or not messages or any(not isinstance(m, dict)
                or not isinstance(m.get('content'), str) for m in messages)):
            raise ValueError('Invalid historical training row')
        turns = [m['content'] for m in messages]
        old_conversations.add(identity(turns))
        old_first.add(norm(turns[0]))
        old_turns.update(norm(t) for t in turns if len(norm(t)) >= 20)
        totals['historical_rows'] += 1
    for row in candidate_rows:
        turns = row.get('conversation')
        if not isinstance(turns, list) or not turns or any(not isinstance(t, str) for t in turns):
            raise ValueError('Invalid candidate training row')
        totals['candidate_rows'] += 1
        totals['candidate_exact_conversations'] += identity(turns) in old_conversations
        totals['candidate_exact_first_turn'] += norm(turns[0]) in old_first
        totals['candidate_with_exact_turn_at_least_20_chars'] += any(norm(t) in old_turns for t in turns if len(norm(t)) >= 20)
    return dict(totals)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(ARCHIVE) != ARCHIVE_SHA or sha(HISTORY) != EXPECTED['source']:
        raise ValueError('Input digest mismatch')
    with HISTORY.open(encoding='utf-8') as old, ZipFile(ARCHIVE) as zipped:
        names = [x for x in zipped.infolist() if x.filename == 'DuConv/train.txt']
        if len(names) != 1:
            raise ValueError('Invalid training member')
        with zipped.open(names[0]) as candidate:
            result = compare((json.loads(line) for line in old), (json.loads(line) for line in candidate))
    if sha(ARCHIVE) != ARCHIVE_SHA or sha(HISTORY) != EXPECTED['source']:
        raise ValueError('Input changed')
    result = {'status': 'training_side_overlap_diagnostic', 'counts': result,
              'inputs': {'archive_sha256': ARCHIVE_SHA, 'history_sha256': EXPECTED['source']},
              'script_sha256': sha(Path(__file__)),
              'limitations': ['Only one historical dialogue source and one archive training member were compared.',
                              'Exact normalized full text or turn identity does not detect substring or semantic overlap.',
                              'A source in the training archive does not establish exposure by any checkpoint.',
                              'No development or final-test text was compared.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
