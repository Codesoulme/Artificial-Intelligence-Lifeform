"""Continue memory replay using diverse fact assignments."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import deepseek_spora_natural_memory_replay as trainer

trainer.START = trainer.ROOT/'runs/deepseek-spora-natural-memory-replay-20261111'
trainer.OUTPUT = trainer.ROOT/'runs/deepseek-spora-diverse-memory-replay-20261111'
trainer.VALUE_PAIR_SETS = tuple(tuple(((old+shift) % 8, (new+shift) % 8)
                                      for old, new in trainer.VALUE_PAIRS)
                                for shift in (0, 1, 3))
trainer.STEPS = 120
trainer.SEED = 712912

if __name__ == '__main__':
    trainer.run()
