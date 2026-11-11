"""Train an isolated temporal-routing candidate on additional wording."""
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_temporal_gate_train as trainer
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.temporal_routing import PAST

trainer.OUTPUT = ROOT / 'runs/deepseek-spora-temporal-gate-v2-20261112'
trainer.EXTRA_TRAIN = {**trainer.EXTRA_TRAIN, PAST: trainer.EXTRA_TRAIN[PAST] + (
    '告诉我记录里{name}以前喝什么。',
    '你知道{name}之前想喝什么吗？',
    '回忆{name}先前想喝的饮品。',
    '记载中{name}过去喝的是什么饮料？',
    '查找{name}较早的饮料偏好。',
)}
trainer.TRAIN_ABSTAIN += (
    '猜猜{name}以前喜欢喝什么。',
    '假设{name}以前喝茶，接着会怎样？',
    '别说出记录里{name}以前喝什么。',
    '把{name}以前喝的饮料换成咖啡。',
)
trainer.SEED = 510248


def streamed_base(*args, **kwargs):
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
