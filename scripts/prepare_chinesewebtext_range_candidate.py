"""Prepare a bounded ChineseWebText2.0 candidate from a fixed byte range.

The source range is train-side only. This script performs mechanical screening
and deterministic domain balancing; it does not approve or train the data.
"""
import argparse
import hashlib
import json
import re
import sys
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = '9b647306646e5ddd85cc239870fc39bfb60e9531'
SOURCE_REPOSITORY = 'CASIA-LM/ChineseWebText2.0'
SOURCE_PATH = 'ChineseWebText2.0/part-0001.jsonl.gz'
EXPECTED_RANGE = 'bytes 0-16777215/11448548844'
EXPECTED_ETAG = '"e7291141348e8430affbdd7f1ea43e0119cfcf5b8f5b99980c8fbe3fb88e7953"'
DEFAULT_RANGE = ROOT / 'runs/p171-chinesewebtext-range-20261109/part-0001.range16m.jsonl.gz'
DEFAULT_HEADERS = ROOT / 'runs/p171-chinesewebtext-range-20261109/headers.txt'


def sha256_path(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def header_value(headers, name):
    prefix = name.lower() + ':'
    for line in headers.read_text(encoding='ascii').splitlines():
        if line.lower().startswith(prefix):
            return line.split(':', 1)[1].strip()
    return None


def parse_rows(path):
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    data = decompressor.decompress(path.read_bytes())
    rows = []
    incomplete = 0
    for line_number, line in enumerate(data.splitlines(), 1):
        try:
            row = json.loads(line.decode('utf8'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            incomplete += 1
            continue
        if not isinstance(row, dict):
            raise ValueError('source row is not an object')
        rows.append((line_number, row))
    if incomplete != 1:
        raise ValueError(f'expected one truncated final row, got {incomplete}')
    return rows, len(data), incomplete


def cjk_ratio(text):
    return sum('\u3400' <= char <= '\u9fff' for char in text) / max(1, len(text))


def filter_reasons(row):
    text = row.get('text')
    reasons = []
    if not isinstance(text, str) or not text.strip():
        return ['missing_text']
    if not isinstance(row.get('quality_score'), (int, float)) or row['quality_score'] < 0.9:
        reasons.append('quality_below_0.9')
    toxicity = row.get('toxicity') or {}
    if toxicity.get('label') != 0 or float(toxicity.get('score', 1)) > 0.9:
        reasons.append('toxicity_gate')
    if len(text) < 200:
        reasons.append('too_short')
    if len(text) > 4096:
        reasons.append('too_long_for_bounded_candidate')
    if cjk_ratio(text) < 0.5:
        reasons.append('low_cjk_ratio')
    if '\ufffd' in text or any(ord(char) < 32 and char not in '\n\r\t' for char in text):
        reasons.append('replacement_or_control')
    if re.search(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', text):
        reasons.append('email_pattern')
    if re.search(r'(?<!\d)1[3-9]\d{9}(?!\d)', text):
        reasons.append('mobile_pattern')
    domain = row.get('domain') or {}
    if not isinstance(domain.get('single_label'), str) or not domain['single_label']:
        reasons.append('missing_domain')
    return reasons


def stable_rank(line_number, text):
    return hashlib.sha256(f'cw2-p171:{line_number}:{text}'.encode('utf8')).hexdigest()


def select_rows(rows, limit=500):
    eligible = []
    rejected = Counter()
    for line_number, row in rows:
        reasons = filter_reasons(row)
        if reasons:
            rejected.update(reasons)
            continue
        domain = row['domain']['single_label']
        eligible.append({'source_line_number': line_number, 'row': row, 'domain': domain,
                         'rank': stable_rank(line_number, row['text'])})
    if limit != 500:
        raise ValueError('this bounded candidate contract requires 500 records')
    domains = sorted({entry['domain'] for entry in eligible})
    if len(domains) < 10:
        raise ValueError('expected at least 10 eligible domains')
    quota = limit // len(domains)
    selected = []
    selected_lines = set()
    for domain in domains:
        bucket = sorted((entry for entry in eligible if entry['domain'] == domain), key=lambda x: x['rank'])
        chosen = bucket[:min(quota, len(bucket))]
        selected.extend(chosen)
        selected_lines.update(entry['source_line_number'] for entry in chosen)
    if len(selected) < limit:
        residual = sorted((entry for entry in eligible if entry['source_line_number'] not in selected_lines), key=lambda x: x['rank'])
        counts = Counter(entry['domain'] for entry in selected)
        cap = quota * 2
        for entry in residual:
            if len(selected) >= limit:
                break
            if counts[entry['domain']] >= cap:
                continue
            selected.append(entry)
            counts[entry['domain']] += 1
    if len(selected) != limit:
        raise ValueError('not enough balanced eligible rows')
    selected.sort(key=lambda x: x['source_line_number'])
    return selected, rejected, Counter(entry['domain'] for entry in eligible)


def prepare(range_path, headers_path, output_dir):
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if header_value(headers_path, 'Content-Range') != EXPECTED_RANGE:
        raise ValueError('content range mismatch')
    if header_value(headers_path, 'ETag') != EXPECTED_ETAG:
        raise ValueError('etag mismatch')
    rows, decompressed_bytes, incomplete = parse_rows(range_path)
    selected, rejected, eligible_domains = select_rows(rows)
    output_dir.mkdir(parents=False)
    candidate_path = output_dir / 'candidate.jsonl'
    with candidate_path.open('x', encoding='utf8', newline='\n') as stream:
        for entry in selected:
            row = entry['row']
            stream.write(json.dumps({
                'record_id': f'cw2_part0001_range16m_line_{entry["source_line_number"]}',
                'source_repository': SOURCE_REPOSITORY,
                'source_revision': SOURCE_REVISION,
                'source_path': SOURCE_PATH,
                'source_range': EXPECTED_RANGE,
                'source_range_sha256': sha256_path(range_path),
                'source_line_number': entry['source_line_number'],
                'domain': row['domain'],
                'quality_score': row['quality_score'],
                'toxicity': row['toxicity'],
                'text': row['text'],
            }, ensure_ascii=False, separators=(',', ':')) + '\n')
    manifest = {
        'status': 'candidate_not_approved_for_training',
        'training_approval': False,
        'training_started': False,
        'repository': SOURCE_REPOSITORY,
        'revision': SOURCE_REVISION,
        'source_path': SOURCE_PATH,
        'source_range': EXPECTED_RANGE,
        'source_range_sha256': sha256_path(range_path),
        'source_total_bytes': 11448548844,
        'source_etag': EXPECTED_ETAG,
        'headers_sha256': sha256_path(headers_path),
        'range_file_bytes': range_path.stat().st_size,
        'decompressed_bytes_before_truncated_line': decompressed_bytes,
        'complete_rows_decoded': len(rows),
        'incomplete_lines_ignored': incomplete,
        'mechanical_rejection_counts': dict(sorted(rejected.items())),
        'eligible_domain_counts': dict(sorted(eligible_domains.items())),
        'selection_rule': 'quality_score >= 0.9, toxicity label 0 and score <= 0.9, 200-4096 chars, CJK ratio >= 0.5, no replacement/control/email/mobile patterns; select up to 50 deterministic SHA256-ranked rows per eligible domain, then fill to 500 from residuals with a 100-row per-domain cap',
        'records': len(selected),
        'candidate_sha256': sha256_path(candidate_path),
        'limitations': [
            'The range is only the first 16 MiB of one train-side shard and is not representative of the full source.',
            'quality_score, domain and toxicity fields are publisher-model outputs, not independent human labels.',
            'Mechanical filters do not prove factual accuracy, rights, privacy absence, naturalness or semantic deduplication.',
            'No training approval, final evaluation use or model training is authorized by this artifact.',
        ],
    }
    with (output_dir / 'manifest.json').open('x', encoding='utf8', newline='\n') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--range-file', type=Path, default=DEFAULT_RANGE)
    parser.add_argument('--headers', type=Path, default=DEFAULT_HEADERS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.range_file, args.headers, args.output)
    print(json.dumps({
        'output': str(args.output),
        'complete_rows_decoded': result['complete_rows_decoded'],
        'records': result['records'],
        'eligible_domain_counts': result['eligible_domain_counts'],
        'training_approval': result['training_approval'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
