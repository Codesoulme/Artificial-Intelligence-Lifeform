"""Matched 64-step control without teacher hidden-state supervision."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts.deepseek_spora_multirelation_hidden_bridge_pilot_20261117 import pilot

pilot.OUT = ROOT / 'runs/deepseek-spora-multirelation-hidden-control-64-20261117'
pilot.TEACHER_HIDDEN_WEIGHT = 0.0


if __name__ == '__main__':
    pilot.main()
