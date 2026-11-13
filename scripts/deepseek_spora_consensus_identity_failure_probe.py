"""Record both locator spans on one failure and one known-name control."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))
from scripts import deepseek_spora_consensus_identity_guard_audit as audit

if __name__ == '__main__':
    audit.main(cases=[('我想知道小林月目前喜欢的饮品。', None),
                      ('我想知道小林目前喜欢的饮品。', 0)],
               output=ROOT / 'runs/deepseek-spora-consensus-identity-failure-probe-20261113.json')
