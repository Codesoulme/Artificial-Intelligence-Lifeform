"""Add paired same-question memory-swap supervision to the broad pilot."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_multirelation_broad_pilot as configuration

pilot = configuration.pilot
pilot.audit.OUT = ROOT / 'runs/deepseek-spora-multirelation-counterfactual-pilot-20261113'
pilot.COUNTERFACTUAL_WEIGHT = 0.5

if __name__ == '__main__':
    pilot.audit.main()
    if not pilot.done:
        raise ValueError('Counterfactual pilot did not run')
