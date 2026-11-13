"""Use virtual rotary positions for memory keys in the exact attention branch."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_exact_attention_higher_lr_pilot as configuration

pilot = configuration.pilot
pilot.audit.OUT = ROOT / 'runs/deepseek-spora-rope-exact-attention-pilot-v3-20261114'
pilot.MEMORY_USE_ROPE = True
pilot.OPTIMIZER_KIND = 'sgd'

if __name__ == '__main__':
    pilot.audit.main()
    if not pilot.done:
        raise ValueError('RoPE exact attention pilot did not run')
