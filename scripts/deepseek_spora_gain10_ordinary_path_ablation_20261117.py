"""Repeat ordinary-question controls while individually disabling the new branch."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_continuous_attention_gain10_ordinary_20261117 as audit

audit.OUT = ROOT / 'runs/deepseek-spora-gain10-ordinary-path-ablation-20261117.json'
audit.CROSS_ABLATION = True


if __name__ == '__main__':
    audit.main()
