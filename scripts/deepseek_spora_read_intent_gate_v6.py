"""Train a research-only read-intent candidate with broader paraphrase contrasts."""
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_read_intent_gate as trainer
from scripts.deepseek_spora_full_stream_loader import load_full_streamed

trainer.OUTPUT = ROOT / 'runs/deepseek-spora-read-intent-gate-v6-20261112'
trainer.INIT_CHECKPOINT = ROOT / 'runs/deepseek-spora-read-intent-gate-v4-20261111/intent.pt'
trainer.EXTRA_READ = (
    '你记得{name}以前喜欢喝什么吗？',
    '还记得{name}以前喜欢什么饮品吗？',
    '可以回想一下{name}之前喜欢的饮料吗？',
    '回忆一下{name}过去喝什么饮料。',
    '查询先前记录的{name}饮料喜好。',
    '从记录里找找{name}过去的饮品偏好。',
    '能告诉我{name}曾经喜欢什么饮品吗？',
    '以前{name}喜欢喝的饮品是什么？',
)
trainer.EXTRA_NON_READ = (
    '猜猜{name}以前喜欢喝什么。',
    '猜一猜{name}现在偏爱什么饮料。',
    '假设{name}过去喜欢喝水，请写个故事。',
    '不要告诉我{name}之前喜欢喝什么。',
    '请为{name}推荐一种饮品。',
    '替{name}把记录里的饮料改成茶。',
    '请把{name}喜欢喝牛奶存进记忆。',
    '如果{name}以前爱喝茶，可能会怎样？',
)
trainer.SEED = 29023
trainer.STEPS = 180
trainer.LEARNING_RATE = 0.0001


def streamed_base(*args, **kwargs):
    # The trainer replaces layers 6 and 12 before feature extraction.  Skipped
    # meta layers prevent loading weights that would be immediately discarded.
    base = load_full_streamed(skip_replaced=(6, 12))
    base.to = lambda *a, **kw: base
    return base


original_extract = trainer.extract


def low_memory_extract(base, tokenizer, questions, batch_size=4):
    if any(p.is_meta for p in base.parameters()) or any(b.is_meta for b in base.buffers()):
        raise ValueError('Unmaterialized layer after structural replacement')
    return original_extract(base, tokenizer, questions, batch_size=1)


if __name__ == '__main__':
    with patch.object(trainer.AutoModelForCausalLM, 'from_pretrained', side_effect=streamed_base), \
            patch.object(trainer, 'extract', side_effect=low_memory_extract):
        trainer.run()
