"""Isolated matched comparison: same-candidate, different-state loss."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts.deepseek_spora_multitoken_span_bridge_20261117 import pilot
from spora import research_counterfactual_span_loss
from spora.research_memory_state_contrast import memory_state_contrast_loss

pilot.OUT = ROOT / 'runs/deepseek-spora-multitoken-state-contrast-64-20261117'
pilot.SPAN_GRADIENT_DIAGNOSTIC = False
research_counterfactual_span_loss.counterfactual_span_loss = memory_state_contrast_loss

if __name__ == '__main__':
    pilot.main()
