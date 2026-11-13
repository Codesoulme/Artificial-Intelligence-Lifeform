"""Reopen the final exact-bridge experiment in a separate process."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_exact_bridge_broad_pilot as configuration

pilot = configuration.pilot
pilot.RELOAD_FROM = ROOT / 'runs/deepseek-spora-exact-bridge-broad-pilot-20261113'
pilot.audit.OUT = ROOT / 'runs/deepseek-spora-exact-bridge-broad-reload-20261113'

if __name__ == '__main__':
    pilot.audit.main()
    if not pilot.done:
        raise ValueError('Exact bridge recovery audit did not run')
