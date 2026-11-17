"""Compare gain-10 recovery and final research checkpoint tensors."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_multitoken_recovery_audit_20261117 as audit

audit.RUN = ROOT / 'runs/deepseek-spora-continuous-attention-gain10-64-20261117'
audit.RECOVERY = ROOT / 'runs/deepseek-spora-continuous-attention-gain10-64-recovery-20261117.pt'
audit.OUT = ROOT / 'runs/deepseek-spora-continuous-attention-gain10-recovery-audit-20261117.json'


if __name__ == '__main__':
    audit.main()
