"""Audit research gates and multi-span veto in the real temporary read path."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

from scripts import deepseek_spora_original_layer_joint_runtime_audit as audit

EXTRA = (
    ('extra_known', '查看一下阿明的饮品记录，他最爱喝什么？', 7),
    ('extra_known', '你还记得小岚偏好的饮料名称吗？', 6),
    ('extra_known', '能说一下阿明平常爱喝哪一种饮品吗？', 7),
    ('extra_known', '说出保存的小岚的饮料喜好。', 6),
    ('extra_multi', '小岚的饮料是什么，顺便说说阿明的。', None),
    ('extra_multi', '分别回忆阿明、小岚爱喝的东西。', None),
    ('extra_multi', '查小岚以及阿明各自的饮品偏好。', None),
    ('extra_recommend', '帮阿明挑一杯适合夏天喝的饮品。', None),
    ('extra_recommend', '替小岚想想该买什么饮料。', None),
    ('extra_recommend', '给阿明提议一种他没喝过的饮品。', None),
    ('extra_negative', '先别告诉我小岚喜欢哪种饮料。', None),
    ('extra_attribute', '阿明喜欢的那杯饮品温度是多少？', None),
    ('extra_write', '帮我把小岚的饮品记录改为牛奶。', None),
    ('extra_hypothetical', '如果阿明爱喝茶，请写一首短诗。', None),
    ('extra_ordinary', '帮我列出三个学习计划。', None),
    ('new_unknown', '请回忆小凯的饮品偏好。', None),
    ('new_unknown', '阿乐喜欢喝哪种饮料？', None),
    ('new_unknown', '帮我查一下小妍爱喝什么。', None),
    ('new_unknown', '小辰记录的饮料喜好是什么？', None),
    ('new_multi', '小凯喜欢什么饮品，阿乐又喜欢什么？', None),
    ('new_multi', '分别告诉我小妍和小辰的饮品。', None),
    ('new_multi', '小辰与小凯各自的饮料是什么？', None),
    ('new_multi', '阿乐的偏好是什么？小妍呢？', None),
    ('new_ordinary', '今天的气温如何？', None),
    ('new_ordinary', '请给我写一句短诗。', None),
)


if __name__ == '__main__':
    audit.main(output=ROOT / 'runs/deepseek-spora-original-multi-joint-runtime-20261114.json',
               gates_folder=ROOT / 'runs/deepseek-spora-original-layer-request-gates-v2-20261114',
               multi_folder=ROOT / 'runs/deepseek-spora-original-multi-span-pilot-20261114',
               calibration_path=ROOT / 'runs/deepseek-spora-original-layer-joint-calibration-20261114.json',
               extra_cases=EXTRA)
