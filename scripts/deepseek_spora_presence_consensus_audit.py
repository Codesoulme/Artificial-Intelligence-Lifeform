"""Apply the existing two-locator veto to the joint routing development set."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

from scripts import deepseek_spora_consensus_identity_guard_audit as consensus
from scripts.deepseek_spora_presence_routing_joint_audit import CASES

OUTPUT = ROOT / 'runs/deepseek-spora-presence-consensus-20261114.json'


if __name__ == '__main__':
    cases = [(question, expected_index) for group, question, expected_index in CASES
             if group in ('known', 'near_name', 'unknown')]
    consensus.main(cases=cases, output=OUTPUT)
