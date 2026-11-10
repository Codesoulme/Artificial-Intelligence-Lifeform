"""Evaluate the sequence-distilled candidate using the existing fixed probe."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import deepseek_spora_fresh_entity_graft_eval as probe

probe.LAYER18 = ROOT/'runs/deepseek-spora-layer18-sequence-distill-20261111/layer-18.pt'
probe.OUTPUT = ROOT/'runs/deepseek-spora-sequence-fresh-entity-eval-20261111.json'

if __name__ == '__main__':
    probe.run()
