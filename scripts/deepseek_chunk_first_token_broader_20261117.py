"""Fixed held-out types and same step budget with broader token supervision."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import deepseek_chunk_first_token_bridge_20261117 as pilot

pilot.OUT = pilot.ROOT / 'runs/deepseek-spora-chunk-first-token-broader-20261117'
pilot.STEPS = 4000
pilot.EXTENDED_TRAIN_TYPES = 12000


if __name__ == '__main__':
    pilot.main()
