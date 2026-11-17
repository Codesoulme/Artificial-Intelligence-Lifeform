"""Rescore saved generations without another model run or threshold tuning."""
import json
from pathlib import Path
from scripts.deepseek_intent_self_judge_pilot_20261117 import OUT, parse_decision, summary

DEST = Path('runs/deepseek-spora-intent-self-judge-rescore-20261117.json')


def main():
    if DEST.exists():
        raise FileExistsError(DEST)
    data = json.loads(OUT.read_text(encoding='utf-8'))
    rows = [{**row, 'decision_rescored': parse_decision(row['output'])} for row in data['rows']]
    scored = [{**row, 'decision': row['decision_rescored']} for row in rows]
    result = {'summary': summary(scored), 'rows': rows,
              'limitations': ['Rescoring rule changed after inspecting outputs; development analysis only.',
                              'Unterminated output may be truncated; exact accepted strings do not prove reliable classification.',
                              'Not an autonomous or deployed memory gate.']}
    with DEST.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result['summary'], ensure_ascii=False))


if __name__ == '__main__':
    main()
