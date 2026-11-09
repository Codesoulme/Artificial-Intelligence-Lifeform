"""Recover complete assistant turns from long training-side candidates for review only."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

from scripts.prepare_conversation_v2 import encode_dialogue
from scripts import prepare_p231_dialogue_candidates as previous


def digest(data):
    return hashlib.sha256(data).hexdigest()


def split_dialogue(tokenizer, messages, limit=512):
    """Emit one supervised full answer per row, discarding only oldest full turns."""
    if (not isinstance(messages, list) or len(messages) < 2 or len(messages) % 2
            or any(not isinstance(m, dict) or m.get('role') != ('user' if i % 2 == 0 else 'assistant')
                   or not isinstance(m.get('content'), str) or not m['content'].strip()
                   for i, m in enumerate(messages))):
        raise ValueError('Invalid dialogue')
    rows, skipped = [], []
    for answer in range(1, len(messages), 2):
        start = 0
        encoded = None
        while start < answer:
            encoded = encode_dialogue(tokenizer, messages[start:answer + 1], limit=limit)
            if encoded is not None:
                break
            start += 2
        if encoded is None:
            skipped.append(answer)
            continue
        tokens, mask = encoded
        final_only = encode_dialogue(tokenizer, messages[answer-1:answer+1], limit=limit)
        if final_only is None:
            raise AssertionError('Final turn unexpectedly exceeds limit')
        target_count = sum(final_only[1])
        # Earlier assistant messages are context, not repeated supervision.
        positions = [i for i, enabled in enumerate(mask) if enabled]
        if target_count < 1 or len(positions) < target_count:
            raise ValueError('Invalid answer mask')
        keep = set(positions[-target_count:])
        final_mask = [i in keep for i in range(len(mask))]
        if [t for t, enabled in zip(tokens, final_mask) if enabled] != [
                t for t, enabled in zip(*final_only) if enabled]:
            raise ValueError('Target differs after retaining history')
        rows.append({'answer_message_index': answer, 'context_start_message_index': start,
                     'tokens': tokens, 'mask': final_mask, 'target_tokens': target_count})
    return rows, skipped


def build(root=previous.ROOT):
    root = Path(root)
    output = root / 'runs/p235-long-dialogue-candidates-20261109'
    if output.exists():
        raise FileExistsError(output)
    paths = {'source': root/'runs/conversation-data-v5/dialogue-train.jsonl',
             'index': root/'reports/P104-multiturn-screen.json',
             'summary': root/'runs/p117-dialogue-link-review-v2/summary.json',
             'families': root/'runs/p117-dialogue-link-review-v2/proposed-families.json',
             'tokenizer': root/'runs/conversation-data-v5/tokenizer.json'}
    if any(previous.file_hash(path) != previous.EXPECTED[key] for key, path in paths.items()):
        raise ValueError('Input digest mismatch')
    index, summary, families = (json.loads(paths[k].read_bytes()) for k in ('index', 'summary', 'families'))
    selected = previous.validate_index(index, summary, families)
    candidates = {n: e for n, e in selected.items() if e['proposed'] == 'candidate'}
    tokenizer = Tokenizer.from_file(str(paths['tokenizer']))
    rows, counts = [], Counter()
    seen, observed = set(), hashlib.sha256()
    with paths['source'].open('rb') as source:
        for line_no, raw in enumerate(source, 1):
            observed.update(raw)
            if line_no not in candidates:
                continue
            row = json.loads(raw)
            if previous.assess(row) or len(row.get('messages', [])) < 4:
                continue
            messages = row['messages']
            identity = (candidates[line_no]['family_sha256'], previous.dialogue_hash(messages))
            if identity in seen:
                continue
            seen.add(identity)
            if encode_dialogue(tokenizer, messages, limit=512) is not None:
                continue
            counts['long_dialogues'] += 1
            pieces, skipped = split_dialogue(tokenizer, messages)
            counts['answer_turns'] += len(messages)//2
            counts['skipped_oversized_turns'] += len(skipped)
            for part in pieces:
                part.update(family_sha256=identity[0], dialogue_sha256=identity[1], source_line=line_no,
                            status='pending_review_not_approved_for_training')
                counts['retained_turns'] += 1
                counts['target_tokens'] += part['target_tokens']
                counts['turns_without_earlier_history'] += part['context_start_message_index'] == part['answer_message_index']-1
                rows.append(part)
    if observed.hexdigest() != previous.EXPECTED['source'] or line_no != index['source_rows']:
        raise ValueError('Source changed')
    if counts['long_dialogues'] != 68 or counts['answer_turns'] != counts['retained_turns'] + counts['skipped_oversized_turns']:
        raise ValueError('Long dialogue lineage or counts differ')
    candidate = b''.join((json.dumps(r, ensure_ascii=False, separators=(',', ':'))+'\n').encode('utf-8') for r in rows)
    result = {'status': 'pending_review_not_approved_for_training', 'trained': False,
              'input_sha256': previous.EXPECTED, 'script_sha256': previous.file_hash(Path(__file__)),
              'encoder_sha256': previous.file_hash(root/'scripts/prepare_conversation_v2.py'),
              'counts': dict(counts), 'candidate_sha256': digest(candidate),
              'limitations': ['Quality, privacy, factual correctness and rights are not certified.',
                  'Recent complete turns may omit required earlier context; each segment requires full-context review.',
                  'Retained targets may overlap other sources or prior training; family partition relies on P117-v2.',
                  'Records from the same dialogue are correlated and not independent examples.']}
    output.mkdir(parents=False, exist_ok=False)
    (output/'candidates.jsonl').write_bytes(candidate)
    (output/'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    print(json.dumps(build(), ensure_ascii=False, indent=2))
