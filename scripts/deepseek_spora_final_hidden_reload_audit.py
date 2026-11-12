"""Reload the final-hidden residual candidate in a fresh process."""
from scripts import deepseek_spora_bridge_reload_audit as audit


audit.s.FINAL_RESIDUAL = True
audit.SOURCE = audit.p.ROOT / 'runs/continuous-reader-final-hidden-residual-pilot-20261113'
audit.OUT = audit.p.ROOT / 'runs/continuous-reader-final-hidden-reload-audit-v2-20261113'
audit.main()
