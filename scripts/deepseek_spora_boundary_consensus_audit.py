"""Reuse the provenance-checked consensus audit for the boundary candidate."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

from scripts import deepseek_spora_expanded_router_consensus_audit as audit

audit.NEW = ROOT / 'runs/deepseek-spora-original-name-boundary-pilot-20261114'
audit.OUTPUT = ROOT / 'runs/deepseek-spora-boundary-consensus-audit-20261114.json'


if __name__ == '__main__':
    audit.main()
