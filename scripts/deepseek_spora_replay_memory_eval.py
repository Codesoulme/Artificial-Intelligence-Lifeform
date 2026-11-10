"""Re-run the matched memory probe for the replay-trained candidate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import deepseek_spora_natural_distill_memory_eval as probe

probe.NEW = probe.ROOT/'runs/deepseek-spora-natural-memory-replay-20261111'
probe.OUTPUT = probe.ROOT/'runs/deepseek-spora-replay-memory-eval-20261111.json'

if __name__ == '__main__':
    probe.run()
