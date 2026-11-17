"""Use selected dialogue evidence values in controlled temporary model memory.

Questions and a common numeric identity are authored; source contexts are NOT
read by the student. This is a controlled transfer diagnostic, not natural
identity resolution, autonomous learning, or a persistent memory evaluation.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_continuous_attention_gain10_ordinary_20261117 as audit
from scripts.prepare_dulemon_evidence_pairs_20261117 import OUT as DATA

SESSIONS = (151, 93, 164)
QUESTIONS = ('小赵最喜欢什么运动？', '小赵最喜欢喝哪种酒？', '小赵从事什么工作？')


def selected_values(data):
    rows = data.get('rows', [])
    result = []
    for session in SESSIONS:
        matches = [r for r in rows if r['session'] == session and r['condition'] == 'after']
        if len(matches) != 1 or not isinstance(matches[0]['answer'], str) or not matches[0]['answer']:
            raise ValueError('Missing or ambiguous evidence value')
        result.append(matches[0]['answer'])
    if len(set(result)) != len(result) or any(v in q for v in result for q in QUESTIONS):
        raise ValueError('Duplicate values or answer text in questions')
    return tuple(result)


def main():
    data = json.loads(DATA.read_text(encoding='utf-8'))
    values = selected_values(data)
    audit.OUT = ROOT / 'runs/deepseek-spora-dialogue-values-memory-pilot-20261117.json'
    audit.QUESTIONS = QUESTIONS
    audit.MEMORY_VALUES = values
    audit.INCLUDE_EMPTY_MEMORY = True
    audit.ADDITIONAL_INPUTS = {'dialogue_values': DATA}
    audit.main()


if __name__ == '__main__':
    main()
