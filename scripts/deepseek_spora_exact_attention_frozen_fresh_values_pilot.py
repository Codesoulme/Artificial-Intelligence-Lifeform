"""Matched frozen-gate control for the fresh-value attention schedule."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_exact_attention_higher_lr_pilot as configuration

pilot = configuration.pilot
pilot.audit.OUT = ROOT / 'runs/deepseek-spora-exact-attention-frozen-fresh-values-20261116'
pilot.MEMORY_ATTENTION = True
pilot.MEMORY_QUERY_GATE = False
pilot.MEMORY_PROJECTION_LR = 0.0001
pilot.MEMORY_GAIN_LR = 0.001
pilot.REPLAY_OLD_COUNT = 3
pilot.REPLAY_FACTOR = 1

if __name__ == '__main__':
    pilot.audit.main()
    if not pilot.done:
        raise ValueError('Fresh-value frozen control did not run')
