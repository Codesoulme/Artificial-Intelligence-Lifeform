"""Give the isolated memory attention branch separate faster learning rates."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_pretrained_memory_attention_pilot as configuration

pilot = configuration.pilot
pilot.audit.OUT = ROOT / 'runs/deepseek-spora-attention-higher-lr-pilot-20261113'
pilot.MEMORY_PROJECTION_LR = 0.0001
pilot.MEMORY_GAIN_LR = 0.001

if __name__ == '__main__':
    pilot.audit.main()
    if not pilot.done:
        raise ValueError('Higher-rate memory attention pilot did not run')
