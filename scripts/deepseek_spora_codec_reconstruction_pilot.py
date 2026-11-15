"""Compare an embedding bottleneck auxiliary loss with the 128-step baseline."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_joint_relation_lane_bridge_pilot as lane

pilot = lane.pilot
pilot.OUT = ROOT / 'runs/deepseek-spora-codec-reconstruction-20261115'
pilot.LANGUAGE_STEPS = 128
pilot.RESET_LANGUAGE_ADAPTERS = True
pilot.CODEC_RECONSTRUCTION_WEIGHT = 5.0

if __name__ == '__main__':
    pilot.main()
