"""Repeat crossed-fact pilot with eight rather than two training name pairs."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import deepseek_fact_crossed_pilot_20261117 as pilot

NAMES = (('小榆', '阿澄'), ('小楠', '阿芮'), ('小秋', '阿禾'), ('小竹', '阿松'),
         ('小夏', '阿冬'), ('小晴', '阿雪'), ('小蕾', '阿博'), ('小遥', '阿卓'))


def main():
    train_names = {name for pair in NAMES for name in pair}
    held_names = {name for pair in pilot.NAMES_HELD for name in pair}
    if len(train_names) != len(NAMES) * 2 or train_names & held_names:
        raise ValueError('Name overlap or repetition')
    pilot.NAMES_TRAIN = NAMES
    pilot.OUT = ROOT / 'runs/deepseek-fact-crossed-names8-20261117'
    pilot.main()


if __name__ == '__main__':
    main()
