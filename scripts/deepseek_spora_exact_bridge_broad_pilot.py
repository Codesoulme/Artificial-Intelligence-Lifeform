"""Train the memory reader with an exact vocabulary bridge under the broad setup."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_multirelation_broad_pilot as configuration

pilot = configuration.pilot
pilot.audit.OUT = ROOT / 'runs/deepseek-spora-exact-bridge-broad-pilot-20261113'
pilot.BRIDGE_MODE = 'exact'

if __name__ == '__main__':
    pilot.audit.main()
    if not pilot.done:
        raise ValueError('Exact bridge pilot did not run')
