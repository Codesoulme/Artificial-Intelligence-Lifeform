"""Attribute controlled ordinary-answer changes to individual memory paths."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_continuous_attention_gain10_ordinary_20261117 as audit

audit.OUT = ROOT / 'runs/deepseek-spora-gain10-ordinary-full-paths-20261117.json'
audit.OTHER_PATH_ABLATION = True
# Focus on the concrete unsafe output and two distinct ordinary tasks;
# each still compares all three independently written memory values.
audit.QUESTIONS = ('请计算三加四等于多少。', '用一句话鼓励一个学习新知识的人。',
                   '路口红灯亮了应该怎么做？')


if __name__ == '__main__':
    audit.main()
