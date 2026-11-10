"""Re-run the sampled natural-generation probe for the replay candidate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import deepseek_spora_natural_distill_generation_eval as probe

probe.NEW = probe.ROOT/'runs/deepseek-spora-natural-memory-replay-20261111'
probe.OUTPUT = probe.ROOT/'runs/deepseek-spora-replay-generation-eval-20261111.json'

if __name__ == '__main__':
    probe.run()
