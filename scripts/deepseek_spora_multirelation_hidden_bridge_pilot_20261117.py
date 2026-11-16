"""Isolated multi-relation continuation of the trained hidden-state bridge.

Controlled facts, addresses and identities are supplied by the experiment.
This does not authorize natural-language writes or deploy a candidate.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_semantic_bridge_pilot as pilot

pilot.OUT = ROOT / 'runs/deepseek-spora-multirelation-hidden-bridge-64-20261117'
pilot.START_CANDIDATE = ROOT / 'runs/deepseek-spora-hidden-bridge-distill-20261115/candidate.pt'
pilot.LOAD_START_BRIDGE = True
pilot.BRIDGE_STEPS = 0
pilot.CODEC_SLOT_COUNT = 5
pilot.CODEC_ADD_EOS = True
pilot.STORE_FULL_FACT = True
pilot.STORE_IDENTITY = True
pilot.STORE_RELATION = False
pilot.BRIDGE_CONDITION_SLOTS = 4
pilot.DIRECT_RESIDUAL = True
pilot.DIRECT_RESIDUAL_LAYER = 20
pilot.FINAL_RESIDUAL = True
pilot.JOINT_CODEC_TRAINING = False
pilot.TRAIN_BRIDGE = False
pilot.TRAIN_CORRECTION_SEQUENCES = False
pilot.TRAIN_MULTI_ENTITY_STATES = False
pilot.TRAIN_EXTRA_QUESTIONS = False
pilot.FOCUS_EXTRA_QUESTIONS = False
pilot.STRICT_SWAP_EVAL = True
pilot.LANGUAGE_STEPS = 64
pilot.LANGUAGE_LR = 3e-6
pilot.TEACHER_HIDDEN_WEIGHT = 1.0
pilot.TEACHER_HIDDEN_LOGIT_WEIGHT = 0.0
pilot.TEACHER_HIDDEN_DELTA_WEIGHT = 0.0
pilot.WORDS = ('茶', '咖啡', '牛奶', '豆浆', '果汁', '可乐',
               '红色', '蓝色', '绿色', '黄色', '白色', '黑色',
               '读书', '跑步', '画画', '唱歌', '写字', '跳舞')
pilot.VALUE_GROUPS = {'drink': (0, 1, 2, 3, 4, 5),
                      'color': (6, 7, 8, 9, 10, 11),
                      'activity': (12, 13, 14, 15, 16, 17)}
pilot.RELATION_TEMPLATES = {
    'drink': ('喜欢喝', ('请用完整句说出{name}喜欢喝什么饮品。',
                     '根据记录，{name}喜欢喝什么？请完整回答。')),
    'color': ('喜欢的颜色是', ('请用完整句说出{name}喜欢什么颜色。',
                       '根据记录，{name}偏爱哪种颜色？请完整回答。')),
    'activity': ('喜欢的活动是', ('请用完整句说出{name}喜欢什么活动。',
                          '根据记录，{name}喜欢做什么？请完整回答。')),
}
pilot.TRAIN_VALUES = (0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15)
pilot.TRAIN_VALUE_COUNT = len(pilot.TRAIN_VALUES)
pilot.EVALUATION_VALUES = (4, 5, 10, 11, 16, 17)


if __name__ == '__main__':
    pilot.main()
