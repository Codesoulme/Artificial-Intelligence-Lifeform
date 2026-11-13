"""Reload and ablate the virtual rotary memory-key candidate."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_rope_exact_attention_pilot as configuration

pilot = configuration.pilot
pilot.RELOAD_FROM = ROOT / 'runs/deepseek-spora-rope-exact-attention-pilot-v3-20261114'
pilot.audit.OUT = ROOT / 'runs/deepseek-spora-rope-exact-attention-reload-20261114'
pilot.EVAL_ATTENTION_ABLATION = True

if __name__ == '__main__':
    pilot.audit.main()
    if not pilot.done:
        raise ValueError('RoPE exact attention reload did not run')
