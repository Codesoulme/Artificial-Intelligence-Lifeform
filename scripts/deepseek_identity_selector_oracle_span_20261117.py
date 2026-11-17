"""Matched selector with oracle subject span pooling in question and record."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
from scripts import deepseek_identity_selector_probe_20261117 as probe
probe.USE_ORACLE_SUBJECT_SPAN=True
probe.OUT=ROOT/'runs/deepseek-identity-selector-oracle-span-20261117'
if __name__=='__main__':probe.main()
