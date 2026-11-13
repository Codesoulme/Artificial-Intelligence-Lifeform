"""Expand crossed-value language training with disjoint held-out values."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

from scripts import deepseek_spora_ttt_crossed_values_pilot as crossed

OUT = ROOT / 'runs/deepseek-spora-ttt-expanded-values-20261114'
TRAIN_NAMES = ('小林', '阿宁', '小雨')
HELD_NAMES = ('小赵', '小郑')
TRAIN_VALUES = ('酸奶', '豆浆', '咖啡', '果汁', '可乐', '牛奶', '红茶', '绿茶')
HELD_VALUES = ('奶茶', '椰汁', '橙汁', '矿泉水')


def configure():
    if set(TRAIN_NAMES) & set(HELD_NAMES) or set(TRAIN_VALUES) & set(HELD_VALUES):
        raise ValueError('Training and held-out identities and values must be disjoint')
    crossed.OUT = OUT
    crossed.TRAIN_NAMES = TRAIN_NAMES
    crossed.HELD_NAMES = HELD_NAMES
    crossed.TRAIN_VALUES = TRAIN_VALUES
    crossed.HELD_VALUES = HELD_VALUES
    crossed.STEPS = 576


if __name__ == '__main__':
    configure()
    crossed.main()
