"""Full-token objective with trainable writer and direct reader."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_differentiable_writer_pilot as pilot

pilot.audit.OUT = ROOT / 'runs/deepseek-spora-joint-full-fact-pilot-20261113'
pilot.TRAIN = ('苹果', '香蕉', '西瓜', '葡萄', '石榴', '金桔', '桑葚', '枇杷')
pilot.HELD = ('莲雾', '蛇果', '百香果', '椰子')
pilot.TRAIN_WRITER = True
pilot.TRAIN_READER = True
pilot.FULL_SEQUENCE = True

if __name__ == '__main__':
    pilot.audit.main()
    if not pilot.done:
        raise ValueError('Joint pilot did not run')
