"""Use an explicit-fact teacher KL target with the exact bridge and attention branch."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_exact_attention_higher_lr_pilot as configuration

pilot = configuration.pilot
pilot.audit.OUT = ROOT / 'runs/deepseek-spora-teacher-exact-attention-pilot-20261113'
pilot.TEACHER_KL_WEIGHT = 0.2
pilot.TEACHER_KL_TEMPERATURE = 2.0

if __name__ == '__main__':
    pilot.audit.main()
    if not pilot.done:
        raise ValueError('Teacher exact attention pilot did not run')
