"""Evaluate the previous candidate on the expanded pilot's exact same cases."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_attention_expanded_values_pilot as configuration

pilot = configuration.pilot
pilot.RELOAD_FROM = ROOT / 'runs/deepseek-spora-attention-higher-lr-pilot-20261113'
pilot.audit.OUT = ROOT / 'runs/deepseek-spora-attention-expanded-baseline-20261113'
pilot.CROSS_EVALUATION = True

if __name__ == '__main__':
    pilot.audit.main()
    if not pilot.done:
        raise ValueError('Expanded baseline did not run')
