"""Independent reload audit for the joint codec continuation candidate."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_bridge_reload_audit as audit


audit.SOURCE = ROOT / 'runs/deepseek-spora-joint-codec-bridge-continuation-20261115'
audit.OUT = ROOT / 'runs/deepseek-spora-joint-codec-bridge-reload-audit-20261115'
audit.PARITY_VALUE_INDICES = tuple(range(18))
audit.VERIFY_CODEC_READS = True
audit.EXTRA_ONLY = False
audit.EXTRA_WORDS = ()
audit.EXTRA_KEYS = (3,)


if __name__ == '__main__':
    audit.main()
