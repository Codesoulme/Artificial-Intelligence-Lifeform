"""Reopen the final pretrained-memory-attention research candidate."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_pretrained_memory_attention_pilot as configuration

pilot = configuration.pilot
pilot.RELOAD_FROM = ROOT / 'runs/deepseek-spora-pretrained-memory-attention-pilot-20261113'
pilot.audit.OUT = ROOT / 'runs/deepseek-spora-pretrained-memory-attention-reload-20261113'
pilot.EVAL_ATTENTION_ABLATION = True

if __name__ == '__main__':
    pilot.audit.main()
    if not pilot.done:
        raise ValueError('Pretrained memory attention recovery audit did not run')
