"""Compare contrast-trained temporal candidate with the baseline."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import deepseek_spora_temporal_gate_v5_contrast as candidate
from scripts import deepseek_spora_temporal_contrast_audit as audit

audit.audit.trainer.OUTPUT = ROOT / 'runs/deepseek-spora-temporal-gate-v5-contrast-20261112'
audit.audit.OUTPUT = ROOT / 'runs/deepseek-spora-temporal-v5-contrast-eval-20261112.json'

if __name__ == '__main__':
    audit.main()
