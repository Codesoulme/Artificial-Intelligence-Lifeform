"""Encode caller-supplied training-side candidates; no training or screening."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
TOKENIZER = RUNS / 'conversation-data-v5' / 'tokenizer.json'


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def identities(text):
    key = re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).lower()
    return digest(key), digest(key[:96])


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')


def run(input_path, output_path, training_side, tokenizer_path=None):
    if training_side is not True:
        raise ValueError('caller must confirm training-side input')
    source = Path(input_path).resolve(strict=True)
    token_path = Path(tokenizer_path or TOKENIZER).resolve(strict=True)
    root = RUNS.resolve(strict=True)
    out = Path(output_path).resolve()
    if out == root or not out.is_relative_to(root) or source.is_relative_to(out) or token_path.is_relative_to(out):
        raise ValueError('output must be a new directory below runs, excluding inputs')
    out.mkdir(parents=False, exist_ok=False)
    stage = 'initializing'
    try:
        source_hash, token_hash = file_hash(source), file_hash(token_path)
        # Load exactly the bytes whose identity is recorded, without downloading weights.
        token_bytes = token_path.read_bytes()
        if hashlib.sha256(token_bytes).hexdigest() != token_hash:
            raise ValueError('tokenizer changed')
        from tokenizers import Tokenizer, __version__ as tokenizers_version
        tokenizer = Tokenizer.from_str(token_bytes.decode('utf-8'))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        bos, eos = (tokenizer.token_to_id(s) for s in ('<|bos|>', '<|eos|>'))
        vocab = tokenizer.get_vocab(with_added_tokens=True)
        if bos is None or eos is None or bos == eos or any(type(v) is not int or not 0 <= v <= 65535 for v in vocab.values()):
            raise ValueError('tokenizer vocabulary is incompatible with uint16 or BOS/EOS')
        (out / 'tokenizer.json').write_bytes(token_bytes)
        seen_rows, seen_hashes, seen_originals, seen_groups = set(), set(), set(), set()
        lengths, rejected = Counter(), Counter()
        observed = hashlib.sha256()
        offset = documents = targets = total = 0
        stage = 'encoding'
        with source.open('rb') as inp, (out/'stories.bin').open('xb') as binary, \
             (out/'stories.jsonl').open('x', encoding='utf-8', newline='\n') as records, \
             (out/'rejected.jsonl').open('x', encoding='utf-8', newline='\n') as rejects:
            for line_number, raw in enumerate(inp, 1):
                observed.update(raw)
                row = json.loads(raw)
                if not isinstance(row, dict) or not isinstance(row.get('text'), str):
                    raise ValueError(f'line {line_number}: invalid record or text')
                number = row.get('source_row_1based')
                if type(number) is not int or number < 1:
                    raise ValueError(f'line {line_number}: invalid source row')
                full, group = identities(row['text'])
                if (row.get('sha256'), row.get('group'), row.get('source_original_sha256')) != (full, group, digest(row['text'])):
                    raise ValueError(f'line {line_number}: identity mismatch')
                if number in seen_rows or full in seen_hashes:
                    raise ValueError(f'line {line_number}: duplicate source row or normalized digest')
                seen_rows.add(number); seen_hashes.add(full); seen_originals.add(row['source_original_sha256']); seen_groups.add(group)
                total += 1
                ids = tokenizer.encode(row['text'], add_special_tokens=False).ids
                if any(type(i) is not int or not 0 <= i <= 65535 for i in ids):
                    raise ValueError(f'line {line_number}: token outside uint16')
                lengths[len(ids)] += 1
                reason = ('empty_content' if not ids else 'over_510_content_tokens' if len(ids) > 510
                          else 'tokenizer_roundtrip' if tokenizer.decode(ids, skip_special_tokens=False) != row['text'] else None)
                if reason:
                    rejected[reason] += 1
                    rejects.write(json.dumps(dict(input_line_1based=line_number, source_row_1based=number,
                        sha256=full, source_original_sha256=row['source_original_sha256'], group=group,
                        content_tokens=len(ids), reason=reason, input_sha256=source_hash), ensure_ascii=False) + '\n')
                    continue
                seq = [bos, *ids, eos]
                binary.write(struct.pack('<' + 'H'*len(seq), *seq))
                record = dict(row, input_line_1based=line_number, input_sha256=source_hash,
                              token_offset=offset, token_count=len(seq), content_tokens=len(ids))
                records.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
                offset += len(seq); documents += 1; targets += len(seq)-1
        stage = 'input_verification'
        if observed.hexdigest() != source_hash or file_hash(source) != source_hash or file_hash(token_path) != token_hash:
            raise ValueError('input or tokenizer changed during conversion')
        summary = dict(status='candidate_not_trained', trained=False,
            input=dict(path=str(source), sha256=source_hash, records=total, training_side='caller_asserted'),
            tokenizer=dict(path=str(token_path), sha256=token_hash, library_version=tokenizers_version, bos=bos, eos=eos),
            script_sha256=file_hash(Path(__file__)),
            counts=dict(encoded_documents=documents, rejected=sum(rejected.values()), rejection_reasons=dict(rejected),
                        unique_source_rows=len(seen_rows), unique_normalized_sha256=len(seen_hashes),
                        unique_original_sha256=len(seen_originals), unique_groups=len(seen_groups),
                        tokens=offset, within_document_next_token_targets=targets),
            content_token_length_histogram={str(k): v for k, v in sorted(lengths.items())},
            binary_format='little-endian uint16; BOS + complete content + EOS; offsets and counts in tokens',
            files={name: file_hash(out/name) for name in ('stories.bin', 'stories.jsonl', 'rejected.jsonl', 'tokenizer.json')},
            limitations=['Caller supplies and designates training-side input; no partition verification is performed.',
                         'No claim of quality, authorization, or semantic isolation from previous corpora.',
                         'Training window boundaries remain the responsibility of a future consumer.'])
        write_json(out/'summary.json', summary)
        return summary
    except Exception as exc:
        write_json(out/'failure.json', dict(status='failed_do_not_use_partial_outputs', stage=stage,
                                            error_type=type(exc).__name__, error=str(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--input-is-training-side', action='store_true', required=True,
                        help='assert the caller has supplied training-side candidates')
    args = parser.parse_args()
    try:
        result = run(args.input, args.output, args.input_is_training_side)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f'conversion failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
