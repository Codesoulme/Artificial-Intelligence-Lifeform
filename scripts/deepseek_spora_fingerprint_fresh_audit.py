"""Fresh authored multi-name evaluation of an offline fingerprint veto.

The examples were authored after earlier errors and do not constitute an
independent population sample. Expected keys are used only to score results.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_fingerprint_route_audit as audit
from scripts.deepseek_named_binding_probe import KEYS

audit.OUTPUT = ROOT / 'runs/deepseek-spora-fingerprint-fresh-audit-20261111.json'
audit.ENROLLMENTS = [(KEYS[0], '小林', 6), (KEYS[1], '阿宁', 1),
                     (KEYS[2], '小雨', 2), (KEYS[3], '阿哲', 3)]
audit.CASES = [
    ('known', '小林眼下喜欢喝哪种饮料？'),
    ('known', '阿宁眼下喜欢喝哪种饮料？'),
    ('known', '小雨眼下喜欢喝哪种饮料？'),
    ('known', '阿哲眼下喜欢喝哪种饮料？'),
    ('unknown', '小林华眼下喜欢喝哪种饮料？'),
    ('unknown', '林小林眼下喜欢喝哪种饮料？'),
    ('unknown', '阿宁悦眼下喜欢喝哪种饮料？'),
    ('unknown', '宁阿宁眼下喜欢喝哪种饮料？'),
    ('unknown', '小雨晴眼下喜欢喝哪种饮料？'),
    ('unknown', '小小雨眼下喜欢喝哪种饮料？'),
    ('unknown', '阿哲轩眼下喜欢喝哪种饮料？'),
    ('unknown', '哲阿哲眼下喜欢喝哪种饮料？'),
]
audit.EXPECTED_KEYS = {audit.CASES[i][1]: KEYS[i] for i in range(4)}

if __name__ == '__main__':
    audit.main()
