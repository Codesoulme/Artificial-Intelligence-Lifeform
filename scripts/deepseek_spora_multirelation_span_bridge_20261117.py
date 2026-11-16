"""Isolated full-value counterfactual pilot with a multi-token training value."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts.deepseek_spora_multirelation_hidden_bridge_pilot_20261117 import pilot

pilot.OUT = ROOT / 'runs/deepseek-spora-multirelation-span-bridge-64-20261117'
pilot.TRAIN_VALUES = (0, 1, 2, 5, 6, 7, 8, 9, 12, 13, 14, 15)
pilot.EVALUATION_VALUES = (3, 4, 10, 11, 16, 17)
pilot.TEACHER_HIDDEN_WEIGHT = 0.0
pilot.MULTITOKEN_COUNTERFACTUAL_WEIGHT = 1.0


if __name__ == '__main__':
    pilot.main()
