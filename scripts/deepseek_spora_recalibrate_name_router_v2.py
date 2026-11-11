"""Train the named-address router for the answer-distilled replacement variant."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import deepseek_spora_recalibrate_name_router as trainer

trainer.SOURCE = ROOT/'runs/deepseek-spora-memory-answer-distill-v4-20261111'
trainer.OUTPUT = ROOT/'runs/deepseek-spora-recalibrate-name-router-v2-20261111'
trainer.SEED = 99343

if __name__ == '__main__':
    trainer.run()
