"""Train an eight-value output candidate; original checkpoint is preserved."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import deepseek_spora_memory_output_adapter as trainer

trainer.OUTPUT = ROOT/'runs/deepseek-spora-memory-output-adapter-v2-20261111'
trainer.TRAIN_VALUES = tuple(range(8))
trainer.STEPS = 256
trainer.SEED = 93012

if __name__ == '__main__':
    trainer.run()
