"""Compare newly supervised facts for a training identity and a held identity."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_bridge_reload_audit as audit

audit.SOURCE = ROOT / 'runs/continuous-reader-corrected-multifruit-pilot-20261113'
audit.OUT = ROOT / 'runs/deepseek-spora-corrected-multifruit-reload-audit-20261113'
audit.EXTRA_WORDS = ('山竹', '石榴', '杨梅', '龙眼', '桑葚', '枇杷')
audit.EXTRA_KEYS = (0, 3)
audit.EXTRA_GROUP_PREFIX = 'extra'
audit.EXTRA_ONLY = True
audit.VERIFY_CODEC_READS = True


if __name__ == '__main__':
    audit.main()
