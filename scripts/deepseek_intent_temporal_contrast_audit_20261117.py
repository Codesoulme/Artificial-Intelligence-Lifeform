"""Compare the new fixed probe against an already-inspected challenge."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_intent_fixed_challenge_20261117 as audit

audit.GATE = ROOT / 'runs/deepseek-spora-intent-temporal-contrast-20261117'
audit.OUT = ROOT / 'runs/deepseek-spora-intent-temporal-contrast-challenge-20261117.json'


if __name__ == '__main__':
    audit.main()
