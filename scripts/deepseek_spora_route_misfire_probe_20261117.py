"""Exercise guarded memory routing with non-retrieval beverage questions.

This wraps the existing controlled history audit with a separate authored
development set; it does not install a guard or modify a real individual.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import deepseek_spora_output_v2_guarded_history_audit as audit

audit.OUTPUT = ROOT / 'runs/deepseek-spora-route-misfire-probe-20261117.json'
audit.CASES = (
    ('current', '从已保存的记录看，小林现在偏爱哪种饮料？', 0, 0, 7),
    ('current', '阿宁目前喜欢喝什么饮品？', 1, 0, 4),
    ('past', '小雨更新之前喜欢喝哪种饮料？', 2, 1, 2),
    ('past', '阿哲改口前偏爱的饮品是什么？', 3, 1, 4),
    ('ordinary', '请写一段关于咖啡店的简短描写。', None, None, None),
    ('ordinary', '给我解释一下饮料和食物有什么区别。', None, None, None),
    ('ordinary', '请将“我喜欢喝茶”翻译成英文。', None, None, None),
    ('ordinary', '帮我把“喜欢喝什么饮料”改成陈述句。', None, None, None),
    ('quotation', '请原样重复“小林现在喜欢喝什么饮料？”。', None, None, None),
    ('quotation', '把“阿宁喜欢喝什么饮料”翻译成英文。', None, None, None),
    ('fiction', '编一个故事，里面有人问小林现在喜欢什么饮料。', None, None, None),
    ('negation', '别读取或告诉我阿宁喜欢喝什么饮料。', None, None, None),
    ('write', '请把小雨喜欢的饮料从牛奶改成茶。', None, None, None),
    ('wrong_relation', '小林喜欢的饮料通常是什么颜色？', None, None, None),
    ('unknown', '小林悦现在偏爱什么饮料？', None, None, None),
    ('multi', '小林和阿宁分别喜欢喝什么饮品？', None, None, None),
)

if __name__ == '__main__':
    audit.main()
