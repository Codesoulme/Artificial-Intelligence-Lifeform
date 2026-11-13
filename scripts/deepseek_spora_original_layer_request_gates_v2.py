"""Broaden contrastive request examples; preserve old cases as development data."""
from copy import deepcopy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

from scripts import deepseek_spora_original_layer_request_gates as pilot

pilot.OUT = ROOT / 'runs/deepseek-spora-original-layer-request-gates-v2-20261114'
pilot.SEED = 39429
pilot.TRAIN_TEMPLATES = deepcopy(pilot.TRAIN_TEMPLATES)
pilot.TRAIN_TEMPLATES['recommend'] += (
    '替{name}挑选一款饮料。',
    '帮{name}选一种适合今天的饮品。',
    '给{name}建议一款新的饮料。',
    '请为{name}选择一杯饮品。',
    '想想给{name}买什么饮料比较好。',
)
pilot.TRAIN_TEMPLATES['read'] += (
    '查查记录中{name}喜欢喝的是什么。',
    '你记得{name}平时喜欢的饮料吗？',
)
pilot.MULTI_TRAIN += (
    '{a}喜欢喝什么，{b}又爱喝什么？',
    '说说{a}和{b}各自偏爱的饮品。',
    '我想知道{a}的喜好，以及{b}的饮料偏好。',
    '先告诉我{a}的饮料喜好，再说{b}的。',
)
pilot.CHALLENGE += (
    ('fresh_read', '查看一下阿明的饮品记录，他最爱喝什么？', 1, 1),
    ('fresh_read', '你还记得小岚偏好的饮料名称吗？', 1, 1),
    ('fresh_read', '能说一下阿明平常爱喝哪一种饮品吗？', 1, 1),
    ('fresh_read', '说出保存的小岚的饮料喜好。', 1, 1),
    ('fresh_multi', '小岚的饮料是什么，顺便说说阿明的。', 0, 1),
    ('fresh_multi', '分别回忆阿明、小岚爱喝的东西。', 0, 1),
    ('fresh_multi', '查小岚以及阿明各自的饮品偏好。', 0, 1),
    ('fresh_recommend', '帮阿明挑一杯适合夏天喝的饮品。', 1, 0),
    ('fresh_recommend', '替小岚想想该买什么饮料。', 1, 0),
    ('fresh_recommend', '给阿明提议一种他没喝过的饮品。', 1, 0),
    ('fresh_negative', '先别告诉我小岚喜欢哪种饮料。', 1, 0),
    ('fresh_attribute', '阿明喜欢的那杯饮品温度是多少？', 1, 0),
    ('fresh_write', '帮我把小岚的饮品记录改为牛奶。', 1, 0),
    ('fresh_hypothetical', '如果阿明爱喝茶，请写一首短诗。', 1, 0),
    ('fresh_ordinary', '帮我列出三个学习计划。', 0, 0),
)


if __name__ == '__main__':
    pilot.main()
