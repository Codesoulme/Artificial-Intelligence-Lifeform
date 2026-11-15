"""Run query-gated exact attention with a real old/new value schedule."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_exact_attention_query_gate_pilot as configuration

pilot = configuration.pilot
pilot.audit.OUT = ROOT / 'runs/deepseek-spora-exact-attention-query-gate-fresh-values-20261116'
pilot.REPLAY_OLD_COUNT = 3
pilot.REPLAY_FACTOR = 1

if __name__ == '__main__':
    pilot.audit.main()
    if not pilot.done:
        raise ValueError('Fresh-value query gate pilot did not run')
