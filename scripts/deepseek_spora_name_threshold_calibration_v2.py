"""Calibrate the address threshold for the answer-distilled router variant."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import deepseek_spora_name_threshold_calibration as trainer

trainer.SOURCE = ROOT/'runs/deepseek-spora-memory-answer-distill-v4-20261111'
trainer.ROUTER = ROOT/'runs/deepseek-spora-recalibrate-name-router-v2-20261111'
trainer.OUTPUT = ROOT/'runs/deepseek-spora-name-threshold-calibration-v2-20261111'

if __name__ == '__main__':
    trainer.run()
