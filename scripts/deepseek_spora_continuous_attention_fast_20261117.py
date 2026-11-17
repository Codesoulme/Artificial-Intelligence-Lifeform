"""Matched continuous-slot attention run with a separate branch learning rate."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts.deepseek_spora_continuous_attention_bridge_20261117 import pilot

pilot.OUT = ROOT / 'runs/deepseek-spora-continuous-attention-fast-64-20261117'
pilot.RECOVERY_FILE = ROOT / 'runs/deepseek-spora-continuous-attention-fast-64-recovery-20261117.pt'
pilot.CROSS_ATTENTION_LR = 3e-5


if __name__ == '__main__':
    pilot.main()
