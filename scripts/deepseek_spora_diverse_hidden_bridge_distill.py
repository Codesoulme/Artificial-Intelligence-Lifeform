"""Distill the hidden-state bridge on correction and multi-entity memory states."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_semantic_bridge_pilot as pilot


pilot.OUT = ROOT / 'runs/deepseek-spora-diverse-hidden-bridge-distill-20261115'
pilot.CODEC_SLOT_COUNT = 5
pilot.CODEC_ADD_EOS = True
pilot.STORE_FULL_FACT = True
pilot.STORE_IDENTITY = True
pilot.BRIDGE_CONDITION_SLOTS = 4
pilot.BRIDGE_CONDITION_OFFSET = 0
pilot.REMAP_CONDITION_INIT = False
pilot.WORDS += (
    '山竹', '石榴', '杨梅', '龙眼', '猕猴桃', '哈密瓜', '金桔', '柿子',
    '李子', '杏子', '圣女果', '西梅', '人参果', '牛油果', '鳄梨', '红毛丹',
    '桑葚', '枇杷', '莲雾', '蛇果', '百香果', '椰子',
)
pilot.TRAIN_VALUE_COUNT = 34
pilot.TRAIN_CORRECTION_SEQUENCES = True
pilot.TRAIN_MULTI_ENTITY_STATES = True
pilot.TRAIN_EXTRA_QUESTIONS = False
pilot.DIRECT_RESIDUAL = True
pilot.DIRECT_RESIDUAL_LAYER = 20
pilot.FINAL_RESIDUAL = True
pilot.STRICT_SWAP_EVAL = True
pilot.START_CANDIDATE = ROOT / 'runs/continuous-reader-balanced-multifruit-pilot-20261113/candidate.pt'
pilot.REPLAY_VALUE_START = 18
pilot.REPLAY_VALUE_FACTOR = 2
pilot.EVALUATION_VALUES = (0, 1, 3, 9, 16, 17, *range(18, 28), *range(34, 40))
pilot.EVALUATE_FIRST_MULTI_QUESTION_ONLY = True
pilot.TRAIN_BRIDGE = False
pilot.TEACHER_HIDDEN_WEIGHT = 1.0
pilot.TEACHER_HIDDEN_LOGIT_WEIGHT = 0.0
pilot.TEACHER_HIDDEN_DELTA_WEIGHT = 0.0
pilot.LANGUAGE_LR = 1e-5
pilot.LANGUAGE_STEPS = 512


if __name__ == '__main__':
    pilot.main()
