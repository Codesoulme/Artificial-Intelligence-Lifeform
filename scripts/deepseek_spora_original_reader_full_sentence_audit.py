"""Compare two readers on full-sentence requests with controlled memories."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

from scripts import deepseek_spora_original_reader_mixed_regression as audit

audit.CASES = (
    ('叶初', 9, '请用完整的一句话说明叶初记录中喜欢喝什么。', 0, 7),
    ('林澄', 11, '请用一句完整的中文句子说出林澄的饮料偏好。', 2, 4),
    ('夏沐', 13, '请回答完整句：夏沐现在最喜欢哪种饮品？', 1, 3),
    ('周谨', 14, '用完整句子告诉我周谨喜欢喝什么饮料。', 5, 6),
)
audit.ORDINARY = ('请用一句话描述下雨的场景。', '三加二等于多少？')


if __name__ == '__main__':
    audit.main(ROOT / 'runs/deepseek-spora-original-reader-first-256-20261114',
               ROOT / 'runs/deepseek-spora-original-reader-first-sentence-20261114.json',
               baseline_source=ROOT / 'runs/deepseek-spora-original-reader-mixed-256-20261114')
