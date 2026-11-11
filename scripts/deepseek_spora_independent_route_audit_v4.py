"""Re-run the fresh wording audit with the conservatively adapted intent gate."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import deepseek_spora_independent_route_audit as audit

audit.INTENT_CHECKPOINT_DIR = ROOT/'runs/deepseek-spora-read-intent-gate-v4-20261111'
audit.OUTPUT = ROOT/'runs/deepseek-spora-independent-route-audit-v4-20261111.json'

if __name__ == '__main__':
    audit.run()
