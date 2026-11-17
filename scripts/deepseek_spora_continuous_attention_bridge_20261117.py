"""Train continuous-slot cross-attention on the fixed multi-token research split."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts.deepseek_spora_multitoken_span_bridge_20261117 import pilot

pilot.OUT = ROOT / 'runs/deepseek-spora-continuous-attention-bridge-64-20261117'
pilot.RECOVERY_FILE = ROOT / 'runs/deepseek-spora-continuous-attention-bridge-64-recovery-20261117.pt'
pilot.CROSS_ATTENTION_MEMORY = True
pilot.CROSS_ATTENTION_CONTINUOUS = True
pilot.CROSS_ATTENTION_LAYER = 23
pilot.CROSS_ATTENTION_WIDTH = 64
pilot.SPAN_GRADIENT_DIAGNOSTIC = False
pilot.SPAN_GRADIENT_COMPARE = False


if __name__ == '__main__':
    pilot.main()
