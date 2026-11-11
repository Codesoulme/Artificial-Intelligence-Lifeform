"""Run the contrast-trained temporal head through known full-runtime cases."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import deepseek_spora_temporal_contrast_runtime as contrast

contrast.audit.CANDIDATE_DIR = ROOT / 'runs/deepseek-spora-temporal-gate-v5-contrast-20261112'
contrast.audit.OUTPUT = ROOT / 'runs/deepseek-spora-temporal-v5-contrast-runtime-20261112.json'

if __name__ == '__main__':
    contrast.audit.main()
