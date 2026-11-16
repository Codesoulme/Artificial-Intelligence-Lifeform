"""Fresh development challenges for the corrected finite-memory route."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import deepseek_spora_output_v2_guarded_history_audit as audit

audit.OUTPUT = ROOT / 'runs/deepseek-spora-route-generalization-probe-20261117.json'
audit.CASES = (
    ('current', '小林当前记录里最喜欢喝什么饮品？', 0, 0, 7),
    ('current', '记忆里阿宁眼下爱喝哪一种饮料？', 1, 0, 4),
    ('current', '只回答小雨最新喜欢的饮品名称。', 2, 0, 6),
    ('past', '阿哲最初喜欢喝什么饮品？', 3, 1, 4),
    ('past', '小林尚未改口时爱喝什么饮料？', 0, 1, 1),
    ('past', '阿宁在改偏好之前喜欢喝什么饮料？', 1, 1, 3),
    ('past', '小雨原先最喜欢的饮品是哪一种？', 2, 1, 2),
    ('attribute', '小林偏爱的饮料看起来是红色还是绿色？', None, None, None),
    ('attribute', '阿宁常喝的饮品一杯大概需要花几元？', None, None, None),
    ('attribute', '小雨喜欢的饮料是甜的还是苦的？', None, None, None),
    ('attribute', '阿哲喜欢喝的那种饮料通常有哪些原料？', None, None, None),
    ('quotation', '翻译这句话：“小林以前喜欢喝哪种饮料？”', None, None, None),
    ('fiction', '给一部小说起个名字，主题是阿宁喜欢的饮料。', None, None, None),
    ('write', '撤回小雨喜欢牛奶的记忆，不用告诉我她喜欢喝什么。', None, None, None),
    ('multi', '小林以前和阿宁现在各喜欢什么饮料？', None, None, None),
    ('unknown', '小林华现在喜欢喝哪种饮料？', None, None, None),
    ('current', '不用告诉我颜色，只告诉我小林现在喜欢哪种饮料。', 0, 0, 7),
    ('current', '别说多少钱，阿宁现在最喜欢的饮料名称是什么？', 1, 0, 4),
)

if __name__ == '__main__':
    audit.main()
