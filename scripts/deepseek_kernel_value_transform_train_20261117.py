"""Matched research run with a zero-initialized post-read value transform."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
from scripts import deepseek_kernel_counterfactual_train_20261117 as pilot

pilot.OUTPUT_MAP_ENABLED=True
pilot.OUT=ROOT/'runs/deepseek-kernel-value-transform-train-20261117'

if __name__=='__main__':pilot.main()
