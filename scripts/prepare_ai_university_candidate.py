"""Prepare a review-only candidate from fixed AI University lesson JSON.

The repository's content license is CC0, but every lesson declares AI-generated
and not human reviewed. This script preserves that limitation and never trains.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / 'runs/research/ai-university-6777837d'
EXPECTED_COMMIT = '6777837dbd46ac8ee6476ea77ea6385dbc8d59f6'
EXCLUDE_KEYS = {'latex', 'numericAnswer', 'correctIndex', 'difficulty', 'id', 'sources', 'provenance', 'preset', 'english', 'englishTitle', 'route', 'status', 'version', 'kind'}
TEXT_KEYS = {'title', 'subtitle', 'concept', 'intuition', 'text', 'instruction', 'observation', 'question', 'answer', 'solution', 'hint', 'explanation', 'claim', 'correction', 'description', 'tagline', 'scope', 'audience', 'reflection', 'body', 'derivation', 'workedExamples', 'commonMistakes', 'practice', 'learningCheck', 'teaching', 'objectives', 'problem'}


def sha256_path(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def collect_text(value, key=None):
    if isinstance(value, str):
        return [value] if key in TEXT_KEYS or key is None else []
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(collect_text(item, key))
        return out
    if isinstance(value, dict):
        out = []
        for child_key, child in value.items():
            if child_key in EXCLUDE_KEYS:
                continue
            out.extend(collect_text(child, child_key))
        return out
    return []


def lesson_files(source):
    paths = []
    for path in sorted((source / 'packages/content').rglob('lessons/*.json')):
        if '/translations/' in path.as_posix() or '\\translations\\' in str(path):
            continue
        paths.append(path)
    if not paths:
        raise ValueError('no lesson files found')
    return paths


def prepare(source, output):
    if output.exists():
        raise FileExistsError(output)
    actual = __import__('subprocess').check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != EXPECTED_COMMIT:
        raise ValueError('source commit mismatch')
    files = lesson_files(source)
    rows = []
    file_manifest = []
    for path in files:
        payload = json.loads(path.read_text(encoding='utf8'))
        prose = [text.strip() for text in collect_text(payload) if text.strip()]
        text = '\n'.join(prose)
        if len(text) < 80:
            continue
        relative = path.relative_to(source).as_posix()
        file_manifest.append({'path': relative, 'sha256': sha256_path(path), 'bytes': path.stat().st_size, 'characters': len(text)})
        rows.append({'record_id': relative, 'source_path': relative, 'text': text, 'characters': len(text), 'cjk_characters': sum('\u3400' <= c <= '\u9fff' for c in text), 'review_status': 'pending_independent_review', 'declared_authoring': 'AI-generated', 'declared_human_reviewed': False})
    output.mkdir(parents=False)
    candidate = output / 'candidate.jsonl'
    with candidate.open('x', encoding='utf8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    manifest = {
        'status': 'candidate_not_approved_for_training',
        'training_approval': False,
        'training_started': False,
        'repository': 'b1n9ui/ai-university',
        'revision': EXPECTED_COMMIT,
        'content_license': 'CC0 1.0 for rights held by contributors; software MIT; third-party material retains its own license',
        'source_files': file_manifest,
        'records': len(rows),
        'candidate_sha256': sha256_path(candidate),
        'limitations': [
            'Lessons declare AI-generated and humanReviewed=false.',
            'Content is concentrated in mathematics, Python and AI-assisted development.',
            'Lessons cite external sources; source facts and any copied material need separate review.',
            'Structured exercises, answers and formula-heavy text may cause format overfitting.',
            'Candidate is not approved for training and is not a general Chinese corpus.',
        ],
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source, args.output)
    print(json.dumps({'output': str(args.output), 'records': result['records'], 'training_approval': result['training_approval']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
