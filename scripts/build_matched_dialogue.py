"""Build paired, last-answer-only supervision from training-side dialogue JSONL."""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer


SPECIAL = ('<|bos|>', '<|eos|>', '<|user|>', '<|assistant|>')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_path(path):
    path = Path(path).resolve()
    if any(re.search(r'(^|[-_.])(test|testing|final|heldout|held-out)([-_.]|$)', p, re.I)
           for p in path.parts):
        raise ValueError(f'Protected-looking path: {path}')
    return path


def encode(tokenizer, messages, ids, target_index):
    tokens, mask = [ids['<|bos|>']], [False]
    for i, msg in enumerate(messages):
        content = tokenizer.encode(msg['content'], add_special_tokens=False).ids
        tokens.extend([ids[f"<|{msg['role']}|>"]] + content + [ids['<|eos|>']])
        mask.extend([False] + [i == target_index] * (len(content) + 1))
    return tokens, mask


def build(source, tokenizer_path, output, limit):
    source, tokenizer_path, output = map(safe_path, (source, tokenizer_path, output))
    if limit < 1:
        raise ValueError('limit must be positive')
    if output == source or output == tokenizer_path or output in source.parents or output in tokenizer_path.parents:
        raise ValueError('Output must not contain an input')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    ids = {s: tokenizer.token_to_id(s) for s in SPECIAL}
    if any(v is None for v in ids.values()) or len(set(ids.values())) != len(ids):
        raise ValueError('Tokenizer lacks distinct dialogue role tokens')
    input_hash, tokenizer_hash = digest(source), digest(tokenizer_path)
    output.mkdir(parents=True, exist_ok=False)
    counts, seen = Counter(), {}
    with source.open(encoding='utf-8') as src, (output / 'context.jsonl').open('x', encoding='utf-8') as full, \
            (output / 'last-user.jsonl').open('x', encoding='utf-8') as single, \
            (output / 'lineage.jsonl').open('x', encoding='utf-8') as lineage:
        for line_no, line in enumerate(src, 1):
            row = json.loads(line)
            messages, group = row.get('messages'), row.get('group')
            if not isinstance(messages, list) or not isinstance(group, str) or not group:
                raise ValueError(f'Line {line_no}: messages and group required')
            for n, msg in enumerate(messages):
                if (not isinstance(msg, dict) or msg.get('role') != ('user' if n % 2 == 0 else 'assistant')
                        or not isinstance(msg.get('content'), str) or not msg['content'].strip()
                        or '<|' in msg['content'] or msg.get('tool_calls') or msg.get('tools')):
                    raise ValueError(f'Line {line_no}: invalid message {n}')
            if len(messages) % 2:
                raise ValueError(f'Line {line_no}: incomplete dialogue')
            offset = row.get('source_turn_offset', 0)
            if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0 or offset % 2:
                raise ValueError(f'Line {line_no}: invalid source_turn_offset')
            # Windowed KdConv records identify a dialogue by group and absolute offset;
            # WildChat uses identity to distinguish conversations in a similarity group.
            identity = row.get('identity') or (group if 'source_turn_offset' in row else row.get('sha256'))
            if not isinstance(identity, str) or not identity:
                identity = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            for index in range(3, len(messages), 2):
                counts['considered'] += 1
                key = json.dumps([group, identity, offset + index], ensure_ascii=False, separators=(',', ':'))
                pair_id = hashlib.sha256(key.encode()).hexdigest()
                last = messages[index - 1:index + 1]
                short_tokens, short_mask = encode(tokenizer, last, ids, 1)
                if len(short_tokens) > limit:
                    counts['last_pair_too_long'] += 1
                    continue
                start = 0
                while True:
                    context = messages[start:index + 1]
                    tokens, mask = encode(tokenizer, context, ids, len(context) - 1)
                    if len(tokens) <= limit:
                        break
                    start += 2
                if start == index - 1:
                    counts['no_history_after_trim'] += 1
                    continue
                target = [t for t, m in zip(tokens, mask) if m]
                assert target == [t for t, m in zip(short_tokens, short_mask) if m]
                signature = hashlib.sha256(json.dumps(last, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                if pair_id in seen:
                    if seen[pair_id] != signature:
                        raise ValueError(f'Conflicting overlapping target at line {line_no}')
                    counts['duplicate_target'] += 1
                    continue
                seen[pair_id] = signature
                record = dict(pair_id=pair_id, group=group)
                full.write(json.dumps(dict(record, tokens=tokens, mask=mask)) + '\n')
                single.write(json.dumps(dict(record, tokens=short_tokens, mask=short_mask)) + '\n')
                meta = dict(record, source_line=line_no, source_identity=identity,
                            source_turn_offset=offset, target_message_index=offset + index,
                            context_start_message_index=offset + start, removed_messages=start,
                            target_tokens=len(target), context_tokens=len(tokens), last_user_tokens=len(short_tokens),
                            source_metadata={k: v for k, v in row.items() if k not in ('messages', 'text', 'tokens', 'mask')})
                lineage.write(json.dumps(meta, ensure_ascii=False) + '\n')
                counts['pairs'] += 1
                counts['target_tokens'] += len(target)
                counts['trimmed_pairs'] += int(start > 0)
    if digest(source) != input_hash or digest(tokenizer_path) != tokenizer_hash:
        raise ValueError('Input changed during build; output is incomplete')
    manifest = dict(format='matched-dialogue-v1', input_path=str(source), input_sha256=input_hash,
                    tokenizer_path=str(tokenizer_path), tokenizer_sha256=tokenizer_hash,
                    builder_sha256=digest(Path(__file__)), max_tokens=limit, counts=counts,
                    files={name: digest(output / name) for name in ('context.jsonl', 'last-user.jsonl', 'lineage.jsonl')},
                    limitations=['Training-side input must be selected and authorized by the operator.',
                                 'No semantic check that a last-user-only answer is appropriate.',
                                 'No split isolation, near-duplicate, rights or quality verification.',
                                 'Rows without a stable dialogue identity cannot deduplicate across differently windowed records.'])
    with (output / 'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='training-side JSONL with messages and group')
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', required=True, help='new directory')
    parser.add_argument('--max-tokens', type=int, default=512, help='maximum tokens per complete record')
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.tokenizer, args.output, args.max_tokens)['counts']))


if __name__ == '__main__':
    main()
