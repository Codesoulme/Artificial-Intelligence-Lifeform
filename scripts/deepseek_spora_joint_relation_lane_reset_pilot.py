"""Train the explicit relation lane with fresh language adapters."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_joint_relation_lane_bridge_pilot as lane


lane.pilot.OUT = ROOT / 'runs/deepseek-spora-joint-relation-lane-reset-20261115'
lane.pilot.RESET_LANGUAGE_ADAPTERS = True
lane.pilot.LANGUAGE_STEPS = 512


if __name__ == '__main__':
    lane.pilot.main()
