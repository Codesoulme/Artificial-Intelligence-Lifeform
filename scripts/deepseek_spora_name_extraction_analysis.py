"""Score saved name extraction outputs, accepting a complete JSON code fence."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/deepseek-spora-name-extraction-probe-20261114.json'
OUTPUT = ROOT / 'runs/deepseek-spora-name-extraction-analysis-20261114.json'


def parse_response(answer):
    if not isinstance(answer, str):
        return None
    text = answer.strip()
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0] in ('```json', '```') and lines[-1] == '```':
        text = '\n'.join(lines[1:-1])
    try:
        result = json.loads(text)
    except ValueError:
        return None
    if (not isinstance(result, list) or any(not isinstance(x, str) or not x for x in result)
            or len(set(result)) != len(result)):
        return None
    return result


def score(rows):
    results = []
    for row in rows:
        parsed = parse_response(row['answer'])
        grounded = parsed is not None and all(name in row['question'] for name in parsed)
        results.append({'group': row['group'], 'question': row['question'], 'expected': row['expected'],
                        'answer': row['answer'], 'parsed': parsed, 'grounded': grounded,
                        'exact': parsed == row['expected']})
    summary = {}
    for group in sorted({row['group'] for row in results}):
        subset = [row for row in results if row['group'] == group]
        summary[group] = {'cases': len(subset), 'parseable': sum(row['parsed'] is not None for row in subset),
                          'grounded': sum(row['grounded'] for row in subset),
                          'exact': sum(row['exact'] for row in subset)}
    return {'summary': summary, 'rows': results}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    original = json.loads(SOURCE.read_text(encoding='utf-8'))
    result = score(original['rows'])
    result['source_base_sha256'] = original['base_sha256']
    result['limitations'] = original['limitations']
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result['summary'], ensure_ascii=False))


if __name__ == '__main__':
    main()
