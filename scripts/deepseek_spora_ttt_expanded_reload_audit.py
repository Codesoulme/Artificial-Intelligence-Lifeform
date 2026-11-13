"""Reload expanded-value reader and repeat held-name counterfactuals."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

from scripts import deepseek_spora_ttt_crossed_reload_audit as audit
from scripts import deepseek_spora_ttt_expanded_values_pilot as expanded


if __name__ == '__main__':
    expanded.configure()
    audit.SOURCE = expanded.OUT
    audit.OUT = ROOT / 'runs/deepseek-spora-ttt-expanded-reload-v2-20261114.json'
    audit.main()
