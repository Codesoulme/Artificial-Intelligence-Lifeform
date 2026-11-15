"""Compare native memory with and without SPORA layers on new question wording.

This reuses the existing controlled finite-value audit. The subjects, memory
states and their addresses are supplied by the experiment; neither values nor
prompts are a sealed benchmark. No token injection or output adapter is used.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

from scripts import deepseek_spora_paired_language_memory_20261116 as paired


paired.OUT = ROOT / 'runs/deepseek-spora-paired-new-question-audit-20261116.json'
paired.TEMPLATES = (
    '只写饮料名称：{name}的饮品喜好是什么？',
    '依据记录，{name}爱喝哪种饮料？只回答名称。',
)
paired.ORDINARY = (
    '计算：7减去3等于几？只回答数字。',
    '请回复“早上好”。',
    '我今天感觉有些疲惫，能给我一个休息建议吗？',
    '你知道我昨天喝了什么吗？',
)


if __name__ == '__main__':
    paired.main()
