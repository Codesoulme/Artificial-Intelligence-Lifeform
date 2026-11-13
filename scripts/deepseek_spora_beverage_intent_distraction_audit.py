"""End-to-end probe for beverage words that do not request stored preferences."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_consensus_identity_guard_audit as audit

OUTPUT = ROOT / 'runs/deepseek-spora-beverage-intent-distraction-audit-20261113.json'
CASES = [
    ('小林现在喜欢喝什么饮料？', 0),
    ('阿宁最喜欢哪种饮品？', 1),
    ('小雨现在爱喝什么？', 2),
    ('阿哲喜欢的饮料是什么？', 3),
    ('请给小林推荐一种饮料。', None),
    ('阿宁把饮料瓶放在了桌子上。', None),
    ('小雨刚才打翻了一杯饮料，怎么清理？', None),
    ('给阿哲买饮料需要带多少钱？', None),
    ('小林问我喝什么，我还没决定。', None),
    ('饮料的保质期通常写在哪里？', None),
    ('阿宁和小雨想调制一种饮品，请给个做法。', None),
    ('小林月现在喜欢喝什么饮料？', None),
]


if __name__ == '__main__':
    audit.main(cases=CASES, output=OUTPUT)
