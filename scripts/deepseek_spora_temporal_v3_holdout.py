"""Run the existing temporal comparison against replay-trained v3."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import deepseek_spora_temporal_v2_holdout as audit

audit.trainer.OUTPUT = ROOT / 'runs/deepseek-spora-temporal-gate-v3-20261112'
audit.OUTPUT = ROOT / 'runs/deepseek-spora-temporal-v3-holdout-20261112.json'

if __name__ == '__main__':
    audit.main()
