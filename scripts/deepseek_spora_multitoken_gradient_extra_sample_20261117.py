"""Run one additional no-update gradient diagnostic with a fixed new seed."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts.deepseek_spora_multitoken_gradient_comparison_20261117 import pilot

if len(sys.argv) != 2 or sys.argv[1] not in ('72382', '72383'):
    raise ValueError('Specify one of the predeclared additional seeds: 72382 or 72383')
pilot.SEED = int(sys.argv[1])
pilot.OUT = ROOT / f'runs/deepseek-spora-multitoken-gradient-comparison-{pilot.SEED}-20261117'


if __name__ == '__main__':
    pilot.main()
