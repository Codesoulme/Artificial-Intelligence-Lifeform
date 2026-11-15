"""Run the semantic-slot bridge with a wider value vocabulary."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

from scripts import deepseek_spora_semantic_slot_pilot as pilot


pilot.OUT = pilot.ROOT / 'runs/deepseek-spora-semantic-slot-expanded-20261116'
pilot.STEPS = 512
pilot.SEED = 74281
pilot.crossed.TRAIN_VALUES = ('酸奶', '豆浆', '咖啡', '果汁', '牛奶', '红茶', '绿茶', '啤酒')
pilot.crossed.HELD_VALUES = ('奶茶', '矿泉水')


if __name__ == '__main__':
    pilot.main()
