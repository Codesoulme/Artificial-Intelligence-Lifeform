"""Clean and classify the bounded P189 paper sample; never trains."""
import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / 'runs/p189-open-chinese-paper-sample-20261109'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cjk_ratio(text):
    return sum('\u3400' <= c <= '\u9fff' for c in text) / max(1, len(text))


def emails(text):
    return re.findall(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', text)


def extract_hans(path):
    raw = path.read_text(encoding='utf8', errors='replace')
    root = ET.fromstring(raw)
    title = ' '.join((node.text or '').strip() for node in root.findall('.//article-title'))
    body = '\n'.join((node.text or '').strip() for node in root.findall('.//body//p') if (node.text or '').strip())
    return title, body, {'format': 'hanspub_xml', 'raw_bytes': path.stat().st_size}


def extract_china(path):
    raw = path.read_text(encoding='utf8', errors='replace')
    lines = [line.strip() for line in raw.splitlines() if line.strip() and not line.startswith('![')]
    title = lines[0] if lines else ''
    return title, '\n'.join(lines), {'format': 'chinaxiv_markdown', 'raw_bytes': path.stat().st_size}


def classify(path):
    try:
        title, body, meta = extract_hans(path) if path.name.startswith('hans-') else extract_china(path)
        reasons = []
    except Exception as exc:
        return {'file': path.name, 'status': 'reject', 'reasons': ['parse_error'], 'error': type(exc).__name__, 'source_sha256': sha(path)}
    if '\ufffd' in title + body:
        reasons.append('replacement_character')
    if len(body) < 1000:
        reasons.append('short_body')
    if cjk_ratio(body) < 0.35:
        reasons.append('low_cjk_ratio')
    if emails(body):
        reasons.append('email_in_body')
    return {'file': path.name, 'status': 'reject' if reasons else 'candidate', 'reasons': reasons,
            'title': title[:200], 'body': body, 'body_chars': len(body), 'cjk_ratio': cjk_ratio(body),
            'email_count': len(emails(body)), 'source_sha256': sha(path), **meta}


def run(input_dir, output_dir):
    if output_dir.exists():
        raise FileExistsError(output_dir)
    paths = sorted([*input_dir.glob('hans-*.xml'), *input_dir.glob('china-*.md')])
    if len(paths) != 24:
        raise ValueError('expected exactly 24 P189 sample files')
    results = [classify(path) for path in paths]
    output_dir.mkdir(parents=False)
    with (output_dir / 'classification.jsonl').open('x', encoding='utf8', newline='\n') as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    candidates = [row for row in results if row['status'] == 'candidate']
    with (output_dir / 'candidate.jsonl').open('x', encoding='utf8', newline='\n') as stream:
        for row in candidates:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    manifest = {'status': 'candidate_not_approved_for_training', 'training_approval': False, 'training_started': False,
                'input_directory': str(input_dir.relative_to(ROOT)), 'records': len(results), 'candidate_records': len(candidates),
                'candidate_sha256': sha(output_dir / 'candidate.jsonl'),
                'counts': {'candidate': len(candidates), 'reject': len(results) - len(candidates)},
                'reject_reasons': {reason: sum(reason in row.get('reasons', []) for row in results) for reason in sorted({reason for row in results for reason in row.get('reasons', [])})},
                'limitations': ['Only 24 files from two sources were sampled.', 'Rule-based cleaning is not quality, factual, rights or privacy proof.', 'No training authorization.']}
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
