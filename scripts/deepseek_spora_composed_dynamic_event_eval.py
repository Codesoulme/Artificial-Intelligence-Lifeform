"""Run the composed memory probe with neural effective-time event selection."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import deepseek_spora_composed_router_answer_eval as probe

probe.TEMPORAL_CHECKPOINT = ROOT/'runs/deepseek-spora-temporal-gate-train-20261111/temporal.pt'
probe.OUTPUT = ROOT/'runs/deepseek-spora-composed-dynamic-event-eval-20261111.json'
probe.DYNAMIC_EVENT_SELECTION = True


if __name__ == '__main__':
    probe.run()
