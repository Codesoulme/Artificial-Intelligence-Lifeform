"""Audit ambiguous two-person questions against the default and candidate gate.

For this conservative development protocol, any accepted multi-person question
is counted as an unsafe acceptance: neither route establishes which person the
requested fact belongs to. This is not a general pronoun or coreference label.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_identity_gate_composed_audit as audit
from scripts.deepseek_named_binding_probe import KEYS

audit.OUTPUT = ROOT / 'runs/deepseek-spora-identity-multiple-person-audit-20261111.json'
audit.CASES = [
    ('known', KEYS[0], '小林现在喜欢喝什么？'),
    ('known', KEYS[1], '阿宁目前喜欢喝什么？'),
    ('known', KEYS[2], '小雨现在喜欢喝什么？'),
    ('known', KEYS[3], '阿哲目前喜欢喝什么？'),
    ('unknown', None, '小林的朋友小林林现在喜欢喝什么？'),
    ('unknown', None, '小林和小林林分别喜欢喝什么？'),
    ('unknown', None, '阿宁的朋友阿宁宁目前喜欢喝什么？'),
    ('unknown', None, '阿宁和阿宁宁各喜欢喝什么？'),
    ('unknown', None, '小雨的朋友小雨晴现在喜欢喝什么？'),
    ('unknown', None, '小雨和小雨晴分别喜欢喝什么？'),
    ('unknown', None, '阿哲的朋友阿哲轩目前喜欢喝什么？'),
    ('unknown', None, '阿哲和阿哲轩各喜欢喝什么？'),
]

if __name__ == '__main__':
    audit.main()
