"""Run the corrected matrix audit in a fresh output directory."""
from scripts import deepseek_spora_correction_memory_audit as audit

audit.OUT = audit.p.ROOT / 'runs/continuous-reader-correction-memory-audit-v2-20261112'
audit.main()
