"""Rescore the question-focus candidate with semantic paraphrase handling."""
import json
from collections import defaultdict
from pathlib import Path

from scripts.spora_memory_generation_metrics import asserted_fact


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/continuous-reader-question-focus-pilot-256-v2-20261113/result.json'
OUT = ROOT / 'reports/question-focus-semantic-rescore-20261113.json'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    data = json.loads(SOURCE.read_text(encoding='utf-8'))
    rows = [row for row in data['after'] if row['group'] == 'held_name' and row['condition'] == 'memory']
    groups = defaultdict(lambda: {'correct': 0, 'total': 0, 'wrong': []})
    scored = []
    for row in rows:
        correct = asserted_fact(row['answer'], row['name'], row['expected'])
        item = {key: row[key] for key in ('question', 'expected', 'answer')}
        item['correct'] = correct
        scored.append(item)
        key = row['question']
        groups[key]['correct'] += int(correct)
        groups[key]['total'] += 1
        if not correct:
            groups[key]['wrong'].append({'expected': row['expected'], 'answer': row['answer']})
    report = {'status': 'question_focus_semantic_rescore',
              'candidate_sha256': data['candidate_sha256'],
              'groups': dict(groups), 'rows': scored,
              'limitations': ['Semantic rules are conservative heuristics.',
                              'This rescoring does not add new model behavior.',
                              'Lists, negations and conflicting facts remain failures.']}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['groups'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
