"""Reload and audit the correction-sequence candidate in a fresh process."""
from scripts import deepseek_spora_bridge_reload_audit as audit


audit.SOURCE = audit.p.ROOT / 'runs/continuous-reader-correction-sequence-pilot-20261112'
audit.OUT = audit.p.ROOT / 'runs/continuous-reader-correction-sequence-reload-audit-20261112'
audit.main()
