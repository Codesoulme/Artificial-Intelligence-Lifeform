"""Oracle ordered identity selection into isolated memory states."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
from scripts import deepseek_kernel_paired_reload_audit_20261117 as audit
audit.SOURCE=ROOT/'runs/deepseek-kernel-full-answer-train-20261117'
audit.OUT=ROOT/'runs/deepseek-kernel-ordered-identity-route-20261117.json'
audit.EXTRA_ORDERED_ROUTE=True
if __name__=='__main__':audit.main()
