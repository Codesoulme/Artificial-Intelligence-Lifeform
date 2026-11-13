"""Reopen the last saved multi-relation research checkpoint in a new process."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_multirelation_train_pilot as pilot

pilot.RELOAD_FROM = ROOT / 'runs/deepseek-spora-multirelation-train-pilot-20261113'
pilot.audit.OUT = ROOT / 'runs/deepseek-spora-multirelation-recovery-audit-v2-20261113'

if __name__ == '__main__':
    pilot.audit.main()
    if not pilot.done:
        raise ValueError('Recovery audit did not run')
