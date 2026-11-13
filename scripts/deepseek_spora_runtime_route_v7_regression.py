"""Re-run bounded end-to-end read routing after the v7 default change."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_runtime_adversarial_audit as audit
from scripts import deepseek_spora_runtime_chat as runtime

runtime.INTENT = ROOT / 'runs/deepseek-spora-read-intent-gate-v7-20261112'
audit.OUTPUT = ROOT / 'runs/deepseek-spora-runtime-route-v7-regression-20261113.json'
audit.CASES = list(audit.CASES) + [
    ('near_name', '小林子现在喜欢喝什么？'),
    ('near_name', '小林娜现在喜欢喝什么？'),
    ('near_name', '林小林现在喜欢喝什么？'),
    ('ordinary', '我今天有点累，能给我一些建议吗？'),
    ('ordinary', '一个人迟到了，怎样礼貌地道歉？'),
    ('positive', '请告诉我小林目前爱喝哪种饮料。'),
]

if __name__ == '__main__':
    audit.main()
