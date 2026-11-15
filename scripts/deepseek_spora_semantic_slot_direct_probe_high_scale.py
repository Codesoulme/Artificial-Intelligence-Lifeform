"""Run the direct semantic-slot capacity probe at larger residual scales."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

from scripts import deepseek_spora_semantic_slot_direct_probe as probe


probe.OUT = probe.ROOT / 'runs/deepseek-spora-semantic-slot-direct-high-scale-20261116.json'
probe.SCALES = (4.0, 8.0, 16.0, 32.0, 64.0)


if __name__ == '__main__':
    probe.main()
