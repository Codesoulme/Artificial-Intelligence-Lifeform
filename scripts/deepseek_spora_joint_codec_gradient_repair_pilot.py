"""Re-run relation learning with the codec-to-loss gradient path audited."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_joint_relation_lane_bridge_pilot as lane

pilot = lane.pilot
pilot.OUT = ROOT / 'runs/deepseek-spora-joint-codec-gradient-repair-20261115'
pilot.LANGUAGE_STEPS = 128
pilot.RESET_LANGUAGE_ADAPTERS = True

if __name__ == '__main__':
    pilot.main()
