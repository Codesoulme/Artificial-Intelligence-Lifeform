"""Run the natural observation transaction with retrained coverage heads."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_natural_observation_transaction_pilot as pilot
from scripts.deepseek_spora_natural_route_retrained_stress_audit import build_components


pilot.OUTPUT = ROOT / 'runs/deepseek-spora-retrained-natural-observation-transaction-20261113'
pilot.WORKING = pilot.OUTPUT / 'working-individual.pt'


def main():
    pilot.build_components = build_components
    if '--finalize' in sys.argv:
        pilot.finalize()
    else:
        pilot.run()


if __name__ == '__main__':
    main()
