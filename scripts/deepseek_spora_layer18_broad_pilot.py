"""Continue layer-18 distillation with more authored ordinary Chinese prompts."""
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_layer18_sequence_distill as trainer
from scripts.deepseek_spora_full_stream_loader import load_full_streamed

trainer.START = ROOT / 'runs/deepseek-spora-layer18-sequence-distill-20261111/layer-18.pt'
trainer.OUTPUT = ROOT / 'runs/deepseek-spora-layer18-broad-pilot-20261112'
trainer.STEPS = 80
trainer.SEED = 37163
trainer.TRAIN_GENERAL = (
    '请用一句话介绍秋天。', '解释什么是守时。', '二加三等于多少？', '描述一棵大树。',
    '朋友紧张时可以怎样安慰？', '写一句晚安问候。', '为什么晚上会看到月亮？', '描述一次安静的阅读。',
    '一周有几天？', '下雨时出门可以带什么？', '介绍图书馆的用途。', '怎样整理书桌？',
    '用一句话描述海边。', '如何认真倾听别人？', '解释分享是什么意思。', '给刚认识的人问好。',
    '冬天出门怎样保暖？', '简单介绍自行车。', '写一句鼓励练习的话。', '为什么植物需要水？',
    '列出两种常见水果。', '怎样节约用纸？', '用一句话描述清晨。', '简述互相帮助的意义。',
)
trainer.HELD_GENERAL = (
    '请用一句话介绍夏天。', '解释什么是耐心。', '三加四等于多少？', '描述一条小河。',
    '朋友失落时可以怎样陪伴？', '写一句早安问候。', '图书为什么需要分类？', '怎样整理背包？',
    '介绍公交车的用途。', '请说一种节约用水的方法。', '为什么树木需要阳光？', '描述一次平静的旅行。',
)
if set(trainer.TRAIN_GENERAL) & set(trainer.HELD_GENERAL):
    raise ValueError('Authored training and held prompts overlap')


def streamed_base(*args, **kwargs):
    base = load_full_streamed(skip_replaced=(6, 12))
    base.to = lambda *a, **kw: base
    return base


original_logits = trainer.model_logits
original_general = trainer.generate_general


def check_materialized(base):
    if any(p.is_meta for p in base.parameters()) or any(b.is_meta for b in base.buffers()):
        raise ValueError('Unmaterialized base layer before inference')


def checked_logits(integrated, base, organism, state, item):
    check_materialized(base)
    return original_logits(integrated, base, organism, state, item)


def checked_general(base, tokenizer, questions):
    check_materialized(base)
    return original_general(base, tokenizer, questions)


if __name__ == '__main__':
    with patch.object(trainer.AutoModelForCausalLM, 'from_pretrained', side_effect=streamed_base), \
            patch.object(trainer, 'model_logits', side_effect=checked_logits), \
            patch.object(trainer, 'generate_general', side_effect=checked_general):
        trainer.run()
