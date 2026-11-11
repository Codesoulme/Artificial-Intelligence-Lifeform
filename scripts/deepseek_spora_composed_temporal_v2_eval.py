"""Re-run the routed-answer probe using the variant-trained temporal gate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import deepseek_spora_composed_router_answer_eval as probe

probe.TEMPORAL_CHECKPOINT = probe.ROOT/'runs/deepseek-spora-temporal-gate-train-20261111/temporal.pt'
probe.OUTPUT = probe.ROOT/'runs/deepseek-spora-composed-temporal-v2-eval-20261111.json'

if __name__ == '__main__':
    probe.run()
