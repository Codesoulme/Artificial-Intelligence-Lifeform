"""Reload and ablate the teacher-aligned exact-attention candidate."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_teacher_exact_attention_pilot as configuration

pilot = configuration.pilot
pilot.RELOAD_FROM = ROOT / 'runs/deepseek-spora-teacher-exact-attention-pilot-20261113'
pilot.audit.OUT = ROOT / 'runs/deepseek-spora-teacher-exact-attention-reload-20261113'
pilot.EVAL_ATTENTION_ABLATION = True

if __name__ == '__main__':
    pilot.audit.main()
    if not pilot.done:
        raise ValueError('Teacher exact attention reload did not run')
