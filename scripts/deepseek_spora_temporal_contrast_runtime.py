"""Check jointly misclassified temporal contrasts through actual memory reads."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import deepseek_spora_temporal_regression_runtime as audit

audit.OUTPUT = ROOT / 'runs/deepseek-spora-temporal-contrast-runtime-20261112.json'
audit.QUESTIONS = (
    ('current_contrast', '历史先不管，{name}现在偏好哪种饮品？', 0, '水'),
    ('past_contrast', '更新前{name}喝的饮品是什么？', 1, '咖啡'),
    ('later_contrast', '以前的选择先不管，{name}后来喜欢哪种饮料？', 2, '水'),
)

if __name__ == '__main__':
    audit.main()
