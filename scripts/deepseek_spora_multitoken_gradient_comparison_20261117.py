"""Compare span and other gradients on one fixed sample, with no update."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts.deepseek_spora_multitoken_span_bridge_20261117 import pilot

pilot.OUT = ROOT / 'runs/deepseek-spora-multitoken-gradient-comparison-20261117'
pilot.SPAN_GRADIENT_COMPARE = True
pilot.DIAGNOSTIC_ONLY = True


if __name__ == '__main__':
    pilot.main()
