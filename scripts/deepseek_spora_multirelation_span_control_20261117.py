"""Matched control for the full-value counterfactual training pilot."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts.deepseek_spora_multirelation_span_bridge_20261117 import pilot

pilot.OUT = ROOT / 'runs/deepseek-spora-multirelation-span-control-64-20261117'
pilot.MULTITOKEN_COUNTERFACTUAL_WEIGHT = 0.0


if __name__ == '__main__':
    pilot.main()
