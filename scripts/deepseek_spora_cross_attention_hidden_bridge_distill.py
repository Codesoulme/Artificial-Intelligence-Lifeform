"""Train a query-dependent memory-slot cross-attention bridge."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_hidden_bridge_distill as hidden


hidden.pilot.OUT = ROOT / 'runs/deepseek-spora-cross-attention-hidden-bridge-20261115'
hidden.pilot.CROSS_ATTENTION_MEMORY = True
hidden.pilot.CROSS_ATTENTION_LAYER = 23
hidden.pilot.CROSS_ATTENTION_WIDTH = 64
hidden.pilot.LANGUAGE_STEPS = 320


if __name__ == '__main__':
    hidden.pilot.main()
