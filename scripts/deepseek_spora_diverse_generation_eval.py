"""Evaluate the diverse replay candidate on natural reasoning samples."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import deepseek_spora_natural_distill_generation_eval as probe

probe.NEW = probe.ROOT/'runs/deepseek-spora-diverse-memory-replay-20261111'
probe.OUTPUT = probe.ROOT/'runs/deepseek-spora-diverse-generation-eval-20261111.json'

if __name__ == '__main__':
    probe.run()
