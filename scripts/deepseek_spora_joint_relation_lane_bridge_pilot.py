"""Test an explicit relation lane in the internal memory packet."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_joint_relation_bridge_pilot as relation


pilot = relation.joint.pilot
pilot.OUT = ROOT / 'runs/deepseek-spora-joint-relation-lane-20261115'
pilot.START_CANDIDATE = ROOT / 'runs/deepseek-spora-joint-codec-bridge-continuation-20261115/candidate.pt'
pilot.LOAD_START_BRIDGE = True
pilot.STORE_IDENTITY = False
pilot.STORE_RELATION = True
pilot.RELATION_TOKEN_IDS = None
pilot.LANGUAGE_STEPS = 320


if __name__ == '__main__':
    pilot.main()
