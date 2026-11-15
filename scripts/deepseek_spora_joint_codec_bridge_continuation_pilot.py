"""Continue joint codec training from the warm-start candidate."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_joint_codec_bridge_pilot as joint


joint.pilot.OUT = ROOT / 'runs/deepseek-spora-joint-codec-bridge-continuation-20261115'
joint.pilot.START_CANDIDATE = ROOT / 'runs/deepseek-spora-joint-codec-bridge-warmstart-20261115/candidate.pt'
joint.pilot.LOAD_START_BRIDGE = True
joint.pilot.BRIDGE_STEPS = 0
joint.pilot.LANGUAGE_STEPS = 320


if __name__ == '__main__':
    joint.pilot.main()
