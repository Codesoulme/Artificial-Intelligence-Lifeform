"""Matched continuous-memory attention run with tenfold residual gain."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts.deepseek_spora_continuous_attention_fast_20261117 import pilot

pilot.OUT = ROOT / 'runs/deepseek-spora-continuous-attention-gain10-64-20261117'
pilot.RECOVERY_FILE = ROOT / 'runs/deepseek-spora-continuous-attention-gain10-64-recovery-20261117.pt'
pilot.CROSS_ATTENTION_OUTPUT_GAIN = 10.0


if __name__ == '__main__':
    pilot.main()
