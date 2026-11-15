"""Repeat fact-slot conditioning with a column-aligned checkpoint remap."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_factored_hidden_bridge_distill as factored


factored.pilot.OUT = ROOT / 'runs/deepseek-spora-factored-hidden-bridge-remapped-20261115'
factored.pilot.REMAP_CONDITION_INIT = True


if __name__ == '__main__':
    factored.pilot.main()
