"""Run the unified fresh audit with the conservatively adapted v4 intent gate."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import deepseek_spora_runtime_chat as runtime
from scripts import deepseek_spora_unified_fresh_audit as audit

runtime.INTENT = ROOT/'runs/deepseek-spora-read-intent-gate-v4-20261111'
audit.OUTPUT = ROOT/'runs/deepseek-spora-unified-fresh-audit-intent-v4-20261111.json'

if __name__ == '__main__':
    audit.run()
