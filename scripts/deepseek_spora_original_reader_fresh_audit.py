"""Predeclare held-out names and question forms for the 256-step reader."""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

from scripts import deepseek_spora_original_reader_mixed_regression as audit

audit.CASES = (
    ('叶初', 9, '请回想叶初记下的饮料偏好是什么。', 0, 7),
    ('林澄', 11, '据现有记录，林澄最喜欢哪一种饮品？', 2, 4),
    ('夏沐', 13, '告诉我夏沐的当前饮品选择。', 1, 3),
    ('周谨', 14, '你保存的周谨的饮品喜好是哪一种？', 5, 6),
)
audit.ORDINARY = ('请用一句话描述下雨的场景。', '三加二等于多少？')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'runs/deepseek-spora-original-reader-mixed-256-20261114')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/deepseek-spora-original-reader-fresh-256-20261114.json')
    options = parser.parse_args()
    audit.main(options.source, options.output)
