"""Broaden single-entity numeric templates in the gate development run."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import deepseek_spora_single_entity_gate as trainer

trainer.OUTPUT = trainer.ROOT/'runs/deepseek-spora-single-entity-gate-v2-20261111'
trainer.NUMERIC_TEMPLATES = trainer.TRAIN_TEMPLATES
trainer.SEED = 88103

if __name__ == '__main__':
    trainer.run()
