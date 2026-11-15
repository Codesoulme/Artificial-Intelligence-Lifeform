"""Measure relation readout with every value covered during adapter training."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_joint_relation_bridge_pilot as relation


pilot = relation.joint.pilot
pilot.OUT = ROOT / 'runs/deepseek-spora-joint-relation-lane-full-coverage-20261115'
pilot.START_CANDIDATE = ROOT / 'runs/deepseek-spora-joint-relation-lane-reset-20261115/candidate.pt'
pilot.LOAD_START_BRIDGE = True
pilot.BRIDGE_STEPS = 0
pilot.STORE_IDENTITY = False
pilot.STORE_RELATION = True
pilot.LANGUAGE_STEPS = 512
pilot.TRAIN_VALUES = tuple(range(18))
pilot.TRAIN_VALUE_COUNT = len(pilot.TRAIN_VALUES)
pilot.EVALUATION_VALUES = None
pilot.CROSS_ATTENTION_MEMORY = False
pilot.DIRECT_RESIDUAL = True
pilot.FINAL_RESIDUAL = True
pilot.LOGIT_RESIDUAL = False
pilot.RESET_LANGUAGE_ADAPTERS = True


if __name__ == '__main__':
    pilot.main()
