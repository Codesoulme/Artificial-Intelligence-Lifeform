"""Correct strict scoring to accept exact complete assertions; preserve originals."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts.deepseek_spora_explicit_fact_oracle_audit import CASES, OUTPUT as SOURCE, strict_answer
from scripts.deepseek_local_baseline import sha

OUTPUT = ROOT / 'runs/deepseek-spora-explicit-fact-oracle-rescored-20261113.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_sha = sha(SOURCE)
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    definitions = {group: (name, relation, values) for group, name, relation, _, values in CASES}
    rows = []
    for row in source['rows']:
        name, relation, values = definitions[row['group']]
        value = row['value']
        if value is not None and value not in values:
            raise ValueError('Unexpected fact value')
        new_score = None if value is None else strict_answer(row['answer'], value, name, relation)
        rows.append({**row, 'original_strict_correct': row['strict_correct'], 'strict_correct': new_score})
    summary = {}
    for model in ('original', 'hybrid_no_memory'):
        selected = [row for row in rows if row['model'] == model and row['value'] is not None]
        pairs = [[row for row in selected if row['group'] == group] for group in definitions]
        if len(selected) != 2 * len(definitions) or any(len(pair) != 2 for pair in pairs):
            raise ValueError('Incomplete paired evidence')
        summary[model] = {'correct': sum(row['strict_correct'] for row in selected), 'total': len(selected),
                          'both_pair_correct': sum(all(row['strict_correct'] for row in pair) for pair in pairs),
                          'pairs': len(pairs)}
    if sha(SOURCE) != source_sha:
        raise ValueError('Source results changed')
    result = {'status': 'rescored_exact_assertions_no_generation', 'source_sha256': source_sha,
              'scoring_script_sha256': sha(Path(__file__)), 'summary': summary, 'rows': rows,
              'limitations': ['Exact name/relation assertion or bare value only; other valid paraphrases can be missed.',
                              'Explicit prompt is a diagnostic teacher, not internal memory.',
                              'The underlying data are a small authored development set.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
