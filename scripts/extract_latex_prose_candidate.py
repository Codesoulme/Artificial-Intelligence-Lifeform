"""Extract a conservative Chinese prose candidate from fixed LaTeX chapters.

The output is a review artifact only. It never trains and never grants source
approval. Formula content is replaced by an explicit placeholder so that a
later reviewer can distinguish it from deleted text.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = (
    ROOT / 'runs/research/aljabr-1-1e2d0bf4',
    ROOT / 'runs/research/aljabr-2-d56e07eb',
)
EXPECTED_COMMITS = {
    'aljabr-1-1e2d0bf4': '1e2d0bf4e5b71194b85958ca6b16a6f396ad25be',
    'aljabr-2-d56e07eb': 'd56e07ebea5d29dceeaf9d227ae81c00eb720e70',
}
MATH_ENVIRONMENTS = r'(?:equation\*?|align\*?|gather\*?|multline\*?|displaymath|math|tikzpicture|tikzcd)'
FORMULA = '〈公式〉'


def sha256_path(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def strip_comments(text):
    lines = []
    for line in text.splitlines():
        cut = None
        for index, char in enumerate(line):
            if char == '%' and (index == 0 or line[index - 1] != '\\'):
                cut = index
                break
        lines.append(line if cut is None else line[:cut])
    return '\n'.join(lines)


def clean_latex(text):
    text = strip_comments(text)
    text = re.sub(r'\\begin\{' + MATH_ENVIRONMENTS + r'\}.*?\\end\{' + MATH_ENVIRONMENTS + r'\}', '\n' + FORMULA + '\n', text, flags=re.S)
    text = re.sub(r'(?s)\\\[.*?\\\]|\\\(.*?\\\)|\$\$.*?\$\$|(?<!\\)\$(?:\\.|[^$])*?\$', FORMULA, text)
    text = re.sub(r'\\(?:cite|parencite|textcite|ref|eqref|pageref|label|url|href|index)\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}', '〈引用〉', text)
    text = re.sub(r'\\(?:chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?\s*\{([^{}]*)\}', r'\n标题：\1\n', text)
    text = re.sub(r'\\begin\{(?:theorem|lemma|proposition|corollary|definition|remark|example)\}(?:\s*\[[^\]]*\])?', '\n定理环境：\n', text)
    text = re.sub(r'\\begin\{proof\}', '\n证明：\n', text)
    text = re.sub(r'\\end\{(?:theorem|lemma|proposition|corollary|definition|remark|example|proof)\}', '\n', text)
    text = re.sub(r'\\(?:begin|end)\{[^{}]*\}(?:\[[^\]]*\])?', '', text)
    text = re.sub(r'\\(?:text|textrm|textbf|textit|emph|mbox|underline)\s*\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'\\[A-Za-z@]+\*?(?:\s*\[[^\]]*\])?', '', text)
    text = text.replace('\\%', '%').replace('\\&', '&').replace('\\_', '_').replace('\\#', '#')
    text = text.replace('~', ' ')
    text = text.replace('{', '').replace('}', '')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n[ \t]+', '\n', text)
    return text.strip()


def cjk_count(text):
    return sum('\u3400' <= char <= '\u9fff' for char in text)


def extract_paragraphs(raw, min_cjk=40, min_chars=80):
    cleaned = clean_latex(raw)
    paragraphs = []
    for index, block in enumerate(re.split(r'\n\s*\n+', cleaned)):
        text = re.sub(r'\s+', ' ', block).strip()
        if len(text) < min_chars or cjk_count(text) < min_cjk:
            continue
        paragraphs.append({'block_index': index, 'text': text, 'chars': len(text), 'cjk': cjk_count(text), 'formula_placeholders': text.count(FORMULA)})
    return paragraphs


def extract_sources(sources, output_dir):
    if output_dir.exists():
        raise FileExistsError(output_dir)
    rows = []
    input_files = []
    for source in sources:
        if source.name not in EXPECTED_COMMITS:
            raise ValueError('unexpected source directory')
        if not (source / '.git').exists():
            raise FileNotFoundError(source)
        actual = __import__('subprocess').check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != EXPECTED_COMMITS[source.name]:
            raise ValueError(f'commit mismatch for {source.name}')
        for path in sorted(source.glob('chapter*.tex')):
            raw = path.read_text(encoding='utf8')
            digest = sha256_path(path)
            input_files.append({'source': source.name, 'path': path.relative_to(source).as_posix(), 'sha256': digest, 'bytes': path.stat().st_size})
            for paragraph in extract_paragraphs(raw):
                rows.append({'record_id': f'{source.name}:{path.stem}:{paragraph["block_index"]}', 'source': source.name, 'path': path.relative_to(source).as_posix(), 'source_sha256': digest, **paragraph, 'review_status': 'pending_independent_review'})
    output_dir.mkdir(parents=False)
    candidate = output_dir / 'candidate.jsonl'
    with candidate.open('x', encoding='utf8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    manifest = {
        'status': 'candidate_not_approved_for_training',
        'training_approval': False,
        'training_started': False,
        'source_commits': EXPECTED_COMMITS,
        'input_files': input_files,
        'records': len(rows),
        'candidate_sha256': sha256_path(candidate),
        'extraction': 'remove comments, replace display/inline math and selected environments with 〈公式〉, retain selected structural labels, remove remaining TeX commands and braces, keep blocks with at least 80 chars and 40 CJK characters',
        'limitations': ['Heuristic LaTeX parsing is not semantic parsing.', 'Third-party citations, figures, formulas and rights boundaries remain unresolved.', 'Candidate is technical and narrow, not general Chinese or dialogue data.', 'No training authorization.'],
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = extract_sources(DEFAULT_SOURCES, args.output)
    print(json.dumps({'output': str(args.output), 'records': result['records'], 'training_approval': result['training_approval']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
