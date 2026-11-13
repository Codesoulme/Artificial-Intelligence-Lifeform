"""Writer pilot on previously failed facts with held facts and old-value replay."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_differentiable_writer_pilot as pilot

pilot.audit.OUT = pilot.ROOT / 'runs/deepseek-spora-writer-new-fact-pilot-20261113'
pilot.TRAIN = ('苹果', '香蕉', '西瓜', '葡萄', '石榴', '金桔', '桑葚', '枇杷')
pilot.HELD = ('莲雾', '蛇果', '百香果', '椰子')

if __name__ == '__main__':
    pilot.audit.main()
    if not pilot.done:
        raise ValueError('Writer pilot did not run')
