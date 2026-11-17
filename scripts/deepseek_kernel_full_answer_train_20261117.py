"""Isolated paired + complete response training comparison."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
from scripts import deepseek_kernel_counterfactual_train_20261117 as pilot
pilot.OUTPUT_MAP_ENABLED=True
pilot.PAIR_WEIGHT=1.0
pilot.FULL_ANSWER_WEIGHT=1.0
pilot.OUT=ROOT/'runs/deepseek-kernel-full-answer-train-20261117'
if __name__=='__main__':pilot.main()
