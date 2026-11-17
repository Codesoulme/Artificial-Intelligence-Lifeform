"""Mix question paraphrases and preserve prior competing-fact controls."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import deepseek_fact_competing_train_20261117 as pilot
from scripts import deepseek_spora_runtime_chat as runtime

PARENT = ROOT / 'runs/deepseek-fact-competing-train-20261117'
OUT = ROOT / 'runs/deepseek-fact-paraphrase-train-20261117'


def rewrite(row, form):
    evidence, sep, question = row['text'].partition('。')
    if not sep or not question.startswith('谁') or not question.endswith('？'):
        raise ValueError('Unexpected input syntax')
    claim = question[1:-1]
    prompts = {'train': claim + '的人是谁？', 'unseen': '请问' + claim + '的是哪位？'}
    return {**row, 'text': evidence + sep + prompts[form]}


def build_groups():
    groups = pilot_build_groups()
    original_train = groups['train']
    rewritten = [rewrite(row, 'train') for row in original_train]
    homonym = []
    for a, b in pilot.TRAIN_NAMES:
        for name in (a, b):
            for question in ('谁喜欢喝茶？', '喜欢喝茶的人是谁？'):
                homonym.append({'text': f'有两个人都叫{name}，其中一个喜欢喝茶。' + question,
                                'label': None, 'left': None, 'right': None, 'kind': 'unknown'})
    groups['train'] = original_train + rewritten + homonym
    groups['unseen_query_old'] = [rewrite(row, 'unseen') for row in groups['old_new_values']]
    groups['unseen_query_same'] = [rewrite(row, 'unseen') for row in groups['same_new_values']]
    train_text = {r['text'] for r in groups['train']}
    if len(train_text) != len(groups['train']) or any(
            train_text & {r['text'] for r in rows} for name, rows in groups.items() if name != 'train'):
        raise ValueError('Training overlap')
    return groups


pilot_build_groups = pilot.build_groups


def main():
    report = json.loads((PARENT / 'result.json').read_text(encoding='utf8'))
    if runtime.sha(PARENT / 'head.pt') != report['head_sha256']:
        raise ValueError('Parent candidate does not match its training report')
    pilot.BASE = PARENT / 'head.pt'
    pilot.OUT = OUT
    pilot.build_groups = build_groups
    pilot.main()


if __name__ == '__main__':
    main()
