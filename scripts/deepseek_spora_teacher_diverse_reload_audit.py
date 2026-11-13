"""Reload the fact-visible-teacher student with no teacher text at inference."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_bridge_reload_audit as audit

audit.SOURCE = ROOT / 'runs/continuous-reader-teacher-diverse-memory-pilot-20261113'
audit.OUT = ROOT / 'runs/deepseek-spora-teacher-diverse-reload-audit-20261113'
audit.EXTRA_WORDS = ('山竹', '石榴', '猕猴桃', '柿子', '桑葚', '枇杷')
audit.EXTRA_KEYS = (0, 3)
audit.EXTRA_GROUP_PREFIX = 'extra'
audit.EXTRA_ONLY = True
audit.VERIFY_CODEC_READS = True
audit.PARITY_VALUE_INDICES = (0, 1, 3, 9, 16, 17)


if __name__ == '__main__':
    audit.main()
