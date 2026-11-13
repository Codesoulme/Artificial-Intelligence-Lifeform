"""Matched v4 comparison for the bounded v7 runtime routing audit."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_runtime_route_v7_regression as configured
from scripts import deepseek_spora_runtime_chat as runtime

runtime.INTENT = ROOT / 'runs/deepseek-spora-read-intent-gate-v4-20261111'
configured.audit.OUTPUT = ROOT / 'runs/deepseek-spora-runtime-route-v4-regression-20261113.json'

if __name__ == '__main__':
    configured.audit.main()
