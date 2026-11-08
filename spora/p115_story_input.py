"""Validate a P115 candidate stream for P111; does not grant approval."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import re
import struct
import unicodedata

from tokenizers import Tokenizer

from spora.source_review import file_sha256


def _hash_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def validate_candidate(directory, expected_summary_sha256, expected_tokenizer_sha256, window_length, model_vocabulary):
    directory = Path(directory)
    summary_path = directory / 'summary.json'
    if (type(window_length) is not int or window_length < 1
            or type(model_vocabulary) is not int or model_vocabulary < 1):
        raise ValueError('Invalid story window or model vocabulary')
    if file_sha256(summary_path) != expected_summary_sha256:
        raise ValueError('Candidate summary changed')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if (summary.get('status') != 'candidate_not_trained' or summary.get('trained') is not False
            or summary.get('input', {}).get('training_side') != 'caller_asserted'):
        raise ValueError('Invalid candidate status or input designation')
    names = ('stories.bin', 'stories.jsonl', 'rejected.jsonl', 'tokenizer.json')
    files = summary.get('files', {})
    for name in names:
        if not isinstance(files.get(name), str) or file_sha256(directory/name) != files[name]:
            raise ValueError('Candidate artifact changed: ' + name)
    if files['tokenizer.json'] != expected_tokenizer_sha256 or summary.get('tokenizer', {}).get('sha256') != expected_tokenizer_sha256:
        raise ValueError('Candidate tokenizer differs from training tokenizer')
    tokenizer = Tokenizer.from_file(str(directory/'tokenizer.json'))
    tokenizer.no_padding(); tokenizer.no_truncation()
    bos, eos, pad = (tokenizer.token_to_id(x) for x in ('<|bos|>', '<|eos|>', '<|pad|>'))
    if any(type(x) is not int or not 0 <= x <= 65535 or x >= model_vocabulary for x in (bos, eos, pad)) or len({bos, eos, pad}) != 3:
        raise ValueError('Incompatible special tokens')
    if (summary['tokenizer'].get('bos'), summary['tokenizer'].get('eos')) != (bos, eos):
        raise ValueError('Candidate special token metadata differs')
    source_hash = summary['input'].get('sha256')
    if not isinstance(source_hash, str) or not re.fullmatch('[0-9a-f]{64}', source_hash):
        raise ValueError('Invalid input digest')
    counts = summary['counts']
    rows = offset = targets = 0
    source_rows, hashes, lines, originals, groups = set(), set(), set(), set(), set()
    lengths, reasons = Counter(), Counter()
    with (directory/'stories.jsonl').open(encoding='utf-8') as records, (directory/'stories.bin').open('rb') as binary:
        for raw in records:
            row = json.loads(raw)
            text = row['text']
            if not isinstance(text, str):
                raise ValueError('Invalid story text')
            normalized = re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).lower()
            identity = (_hash_text(normalized), _hash_text(normalized[:96]), _hash_text(text))
            if (row.get('sha256'), row.get('group'), row.get('source_original_sha256')) != identity:
                raise ValueError('Story identity mismatch')
            source_row, line = row.get('source_row_1based'), row.get('input_line_1based')
            if (type(source_row) is not int or source_row < 1 or source_row in source_rows
                    or type(line) is not int or line < 1 or line in lines or identity[0] in hashes
                    or row.get('input_sha256') != source_hash):
                raise ValueError('Story provenance mismatch')
            source_rows.add(source_row); lines.add(line); hashes.add(identity[0])
            originals.add(identity[2]); groups.add(identity[1])
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if (not 1 <= len(ids) <= 510 or len(ids)+1 > window_length
                    or any(type(i) is not int or not 0 <= i <= 65535 or i >= model_vocabulary or i in (bos, eos, pad) for i in ids)
                    or tokenizer.decode(ids, skip_special_tokens=False) != text):
                raise ValueError('Story incompatible with P111 document sampling')
            seq = [bos, *ids, eos]
            if (type(row.get('token_offset')) is not int or row['token_offset'] != offset
                    or type(row.get('token_count')) is not int or row['token_count'] != len(seq)
                    or type(row.get('content_tokens')) is not int or row['content_tokens'] != len(ids)
                    or binary.read(2*len(seq)) != struct.pack('<'+'H'*len(seq), *seq)):
                raise ValueError('Story binary or offsets mismatch')
            rows += 1; offset += len(seq); targets += len(seq)-1; lengths[len(ids)] += 1
        if binary.read(1):
            raise ValueError('Unexpected trailing story bytes')
    rejected = 0
    with (directory/'rejected.jsonl').open(encoding='utf-8') as records:
        for raw in records:
            row = json.loads(raw)
            source_row, line = row.get('source_row_1based'), row.get('input_line_1based')
            full, original, group = (row.get(k) for k in ('sha256', 'source_original_sha256', 'group'))
            length, reason = row.get('content_tokens'), row.get('reason')
            if (type(source_row) is not int or source_row < 1 or source_row in source_rows
                    or type(line) is not int or line < 1 or line in lines
                    or any(not isinstance(h, str) or not re.fullmatch('[0-9a-f]{64}', h) for h in (full, original, group))
                    or full in hashes or row.get('input_sha256') != source_hash
                    or type(length) is not int or length < 0
                    or reason not in ('empty_content', 'over_510_content_tokens', 'tokenizer_roundtrip')
                    or (reason == 'empty_content' and length != 0)
                    or (reason == 'over_510_content_tokens' and length <= 510)
                    or (reason == 'tokenizer_roundtrip' and not 1 <= length <= 510)):
                raise ValueError('Rejected record metadata mismatch')
            source_rows.add(source_row); lines.add(line); hashes.add(full); originals.add(original); groups.add(group)
            lengths[length] += 1; reasons[reason] += 1; rejected += 1
    if (not rows or counts.get('encoded_documents') != rows or counts.get('tokens') != offset
            or counts.get('within_document_next_token_targets') != targets
            or counts.get('rejected') != rejected or counts.get('rejection_reasons') != dict(reasons)
            or summary['input'].get('records') != rows+rejected or lines != set(range(1, rows+rejected+1))
            or counts.get('unique_source_rows') != len(source_rows) or counts.get('unique_normalized_sha256') != len(hashes)
            or counts.get('unique_original_sha256') != len(originals) or counts.get('unique_groups') != len(groups)
            or summary.get('content_token_length_histogram') != {str(k): v for k, v in sorted(lengths.items())}):
        raise ValueError('Candidate count mismatch or empty stream')
    # Check again after parsing so edits during validation fail closed.
    if file_sha256(summary_path) != expected_summary_sha256 or any(file_sha256(directory/n) != files[n] for n in names):
        raise ValueError('Candidate changed during validation')
    return {'summary_sha256': expected_summary_sha256, 'story_sha256': files['stories.bin'],
            'tokenizer_sha256': expected_tokenizer_sha256, 'documents': rows,
            'tokens': offset, 'within_document_next_token_targets': targets,
            'files': {name: files[name] for name in names}}
