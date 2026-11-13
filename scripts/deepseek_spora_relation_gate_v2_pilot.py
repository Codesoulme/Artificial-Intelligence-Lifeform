"""Add relation and speculation contrasts; the earlier test is now development."""
from copy import deepcopy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

from scripts import deepseek_spora_relation_gate_pilot as pilot

pilot.OUT = pilot.ROOT / 'runs/deepseek-spora-relation-gate-v2-pilot-20261114'
pilot.TEMPLATES = deepcopy(pilot.TEMPLATES)
pilot.TEMPLATES['train']['attribute'] += (
    '请问{name}喜欢的饮料味道是什么样的？',
    '{name}的饮品通常装在多大容量的杯子里？',
    '你知道{name}喜欢的饮料呈什么颜色吗？',
    '请说说{name}钟爱饮品的口感。',
)
pilot.TEMPLATES['train']['hypothesis'] += (
    '推测{name}可能选择喝什么饮料。',
    '猜测{name}可能偏爱哪一种饮品。',
    '请根据季节估计{name}可能会喝什么。',
    '假设{name}想喝茶，写一段话。',
)
pilot.SEED = 70184


if __name__ == '__main__':
    pilot.main()
