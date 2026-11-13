"""Matched pre-read audit with the separately trained boundary locator."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_consensus_identity_guard_audit as audit
from scripts.deepseek_spora_consensus_identity_varied_audit import CASES

if __name__ == '__main__':
    audit.main(cases=CASES,
               output=ROOT / 'runs/deepseek-spora-consensus-identity-boundary-comparison-20261113.json',
               candidate_dir=ROOT / 'runs/deepseek-spora-boundary-train-pilot-20261112')
