"""Match approximately the two-name experiment's number of training passes."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import deepseek_fact_crossed_pilot_20261117 as pilot
from scripts.deepseek_fact_crossed_names8_20261117 import NAMES


def main():
    if {name for pair in NAMES for name in pair} & {name for pair in pilot.NAMES_HELD for name in pair}:
        raise ValueError('Training and held names overlap')
    pilot.NAMES_TRAIN = NAMES
    pilot.STEPS = 1200
    pilot.OUT = ROOT / 'runs/deepseek-fact-crossed-names8-1200-20261117'
    pilot.main()


if __name__ == '__main__':
    main()
