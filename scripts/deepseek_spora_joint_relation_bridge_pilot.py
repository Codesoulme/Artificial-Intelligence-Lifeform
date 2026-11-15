"""Continue joint codec training on independent relation and value domains."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_joint_codec_bridge_pilot as joint


joint.pilot.OUT = ROOT / 'runs/deepseek-spora-joint-relation-bridge-20261115'
joint.pilot.START_CANDIDATE = ROOT / 'runs/deepseek-spora-joint-codec-bridge-continuation-20261115/candidate.pt'
joint.pilot.LOAD_START_BRIDGE = True
joint.pilot.BRIDGE_STEPS = 0
joint.pilot.LANGUAGE_STEPS = 320
joint.pilot.WORDS = (
    '茶', '咖啡', '牛奶', '豆浆', '果汁', '可乐',
    '红色', '蓝色', '绿色', '黄色', '白色', '黑色',
    '读书', '跑步', '画画', '唱歌', '写字', '跳舞',
)
joint.pilot.VALUE_GROUPS = {
    'drink': (0, 1, 2, 3, 4, 5),
    'color': (6, 7, 8, 9, 10, 11),
    'activity': (12, 13, 14, 15, 16, 17),
}
joint.pilot.RELATION_TEMPLATES = {
    'drink': ('喜欢喝', ('请只回答{name}喜欢喝什么饮品。', '记录里{name}喜欢喝什么？只写饮品名称。')),
    'color': ('喜欢的颜色是', ('请只回答{name}的颜色。', '记录里{name}喜欢哪种颜色？只写颜色。')),
    'activity': ('喜欢的活动是', ('请只回答{name}喜欢的活动。', '记录里{name}喜欢做什么？只写活动。')),
}
joint.pilot.TRAIN_VALUES = (0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15)
joint.pilot.TRAIN_VALUE_COUNT = len(joint.pilot.TRAIN_VALUES)
joint.pilot.EVALUATION_VALUES = (4, 5, 10, 11, 16, 17)
joint.pilot.TRAIN_CORRECTION_SEQUENCES = False
joint.pilot.TRAIN_MULTI_ENTITY_STATES = False
joint.pilot.TRAIN_EXTRA_QUESTIONS = False
joint.pilot.FOCUS_EXTRA_QUESTIONS = False
joint.pilot.STRICT_SWAP_EVAL = True
joint.pilot.TEACHER_HIDDEN_WEIGHT = 1.0
joint.pilot.TEACHER_HIDDEN_LOGIT_WEIGHT = 0.0
joint.pilot.TEACHER_HIDDEN_DELTA_WEIGHT = 0.0


if __name__ == '__main__':
    joint.pilot.main()
