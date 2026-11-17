"""Recheck a previously reviewed dialogue set using the new intent candidate."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_dulemon_context_intent_challenge_20261117 as audit

audit.GATE = ROOT / 'runs/deepseek-spora-intent-temporal-contrast-20261117'
audit.OUT = ROOT / 'runs/deepseek-spora-intent-temporal-contrast-dialogue-20261117.json'


if __name__ == '__main__':
    audit.main()
