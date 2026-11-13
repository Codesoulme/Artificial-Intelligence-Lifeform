"""Predeclared varied-wording identity audit; no threshold tuning."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_consensus_identity_guard_audit as audit

OUTPUT = ROOT / 'runs/deepseek-spora-consensus-identity-varied-audit-20261113.json'
# The four base identities were used previously; the new wording and near-name
# variants are authored probes, not certified disjoint from model pretraining.
CASES = [
    ('请查询小林当前偏爱的饮料。', 0),
    ('记录中阿宁喜欢喝哪种饮品？', 1),
    ('小雨的饮料偏好是什么？', 2),
    ('告诉我阿哲现在最爱喝什么。', 3),
    ('我想知道小林目前喜欢的饮品。', 0),
    ('阿宁目前喜欢喝什么饮料？', 1),
    ('小雨现在偏爱哪种饮料？', 2),
    ('阿哲喜欢喝什么？', 3),
    ('请查询小林晓当前偏爱的饮料。', None),
    ('记录中阿宁安喜欢喝哪种饮品？', None),
    ('小雨欣的饮料偏好是什么？', None),
    ('告诉我阿哲明现在最爱喝什么。', None),
    ('我想知道小林月目前喜欢的饮品。', None),
    ('阿宁宁目前喜欢喝什么饮料？', None),
    ('小雨泽现在偏爱哪种饮料？', None),
    ('阿哲宇喜欢喝什么？', None),
    ('林小林目前喜欢什么饮品？', None),
    ('周阿宁喜欢喝什么饮料？', None),
    ('小雨晨现在想喝哪种饮料？', None),
    ('阿哲远现在想喝哪种饮料？', None),
]

if __name__ == '__main__':
    audit.main(cases=CASES, output=OUTPUT)
