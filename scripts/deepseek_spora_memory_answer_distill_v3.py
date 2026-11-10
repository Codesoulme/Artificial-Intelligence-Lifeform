"""Distill answer behavior with held-out old-to-new fact assignments."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_memory_answer_distill as trainer

VALUE_PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7), (1, 2), (3, 4), (5, 0), (7, 2))
_original_facts = trainer._facts
trainer._facts = lambda model, state: _original_facts(model, state, VALUE_PAIRS)
trainer.START_CANDIDATES = {
    6: trainer.ROOT / 'runs/deepseek-spora-memory-answer-distill-v2-20261111/layer-6.pt',
    12: trainer.ROOT / 'runs/deepseek-spora-memory-answer-distill-v2-20261111/layer-12.pt',
}
trainer.OUTPUT = trainer.ROOT / 'runs/deepseek-spora-memory-answer-distill-v3-20261111'
trainer.STEPS = 150
trainer.SEED = 24071


if __name__ == '__main__':
    trainer.run()
