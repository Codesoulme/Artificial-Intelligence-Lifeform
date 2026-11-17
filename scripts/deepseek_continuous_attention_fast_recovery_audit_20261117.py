"""Compare the fast-branch recovery and final research candidate."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_multitoken_recovery_audit_20261117 as audit

audit.RUN = ROOT / 'runs/deepseek-spora-continuous-attention-fast-64-20261117'
audit.RECOVERY = ROOT / 'runs/deepseek-spora-continuous-attention-fast-64-recovery-20261117.pt'
audit.OUT = ROOT / 'runs/deepseek-spora-continuous-attention-fast-recovery-audit-20261117.json'


if __name__ == '__main__':
    audit.main()
