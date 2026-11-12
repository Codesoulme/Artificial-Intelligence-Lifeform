"""Reload the continuous-learning multi-entity candidate in a new process."""
from scripts import deepseek_spora_bridge_reload_audit as audit


audit.SOURCE = audit.p.ROOT / 'runs/continuous-reader-multi-entity-question-finetune-pilot-20261112'
audit.OUT = audit.p.ROOT / 'runs/continuous-reader-multi-entity-finetune-reload-audit-20261112'
audit.main()
