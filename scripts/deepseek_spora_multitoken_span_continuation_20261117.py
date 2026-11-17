"""Continue the isolated 64-step multi-token bridge with a new order.

This starts a fresh optimizer, so it is not equivalent to uninterrupted
training. A recovery checkpoint is saved after training and before evaluation.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts.deepseek_spora_multitoken_span_bridge_20261117 import pilot

pilot.OUT = ROOT / 'runs/deepseek-spora-multitoken-span-continuation-128-20261117'
pilot.START_CANDIDATE = ROOT / 'runs/deepseek-spora-multitoken-span-bridge-64-20261117/candidate.pt'
pilot.RECOVERY_FILE = ROOT / 'runs/deepseek-spora-multitoken-span-continuation-128-recovery-20261117.pt'
pilot.SEED = 72384
pilot.LANGUAGE_STEPS = 128
pilot.SPAN_GRADIENT_DIAGNOSTIC = False
pilot.SPAN_GRADIENT_COMPARE = False
pilot.DIAGNOSTIC_ONLY = False
pilot.PATH_ABLATION_DIAGNOSTIC = False


if __name__ == '__main__':
    pilot.main()
