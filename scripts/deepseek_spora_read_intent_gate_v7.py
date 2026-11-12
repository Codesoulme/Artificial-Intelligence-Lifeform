"""Research-only intent head with additional historical read contrasts."""
from unittest.mock import patch

from scripts import deepseek_spora_read_intent_gate_v6 as prior

trainer = prior.trainer
trainer.OUTPUT = trainer.ROOT / 'runs/deepseek-spora-read-intent-gate-v7-20261112'
trainer.INIT_CHECKPOINT = trainer.ROOT / 'runs/deepseek-spora-read-intent-gate-v6-20261112/intent.pt'
trainer.EXTRA_READ = trainer.EXTRA_READ + (
    '偏好发生变化前，{name}爱喝什么饮品？',
    '调整饮料记录以前，{name}的选择是什么？',
    '请查一下{name}更改记录前喜欢喝的东西。',
    '改动之前{name}的饮料偏好是什么？',
    '我想知道{name}曾经选择过哪种饮品。',
    '找出{name}更新前留下的饮料记录。',
    '请告诉我{name}过去保存的饮品喜好。',
    '回头查查{name}原先爱喝的饮料。',
)
trainer.EXTRA_NON_READ = trainer.EXTRA_NON_READ + (
    '请不要查{name}更改前的饮料记录。',
    '把{name}更改前的饮料记录删除。',
    '给{name}推荐一款之前没喝过的饮品。',
    '假设{name}更改前喜欢牛奶，编个故事。',
    '修改{name}以前存下的饮料偏好。',
    '猜测{name}更新之前可能爱喝什么。',
    '不需要告诉我{name}原来喝什么。',
    '把{name}偏好改变以前的记录清空。',
)
trainer.SEED = 29024
trainer.STEPS = 180
trainer.LEARNING_RATE = 0.0001


def main():
    with patch.object(trainer.AutoModelForCausalLM, 'from_pretrained', side_effect=prior.streamed_base), \
         patch.object(trainer, 'extract', side_effect=prior.low_memory_extract):
        trainer.run()


if __name__ == '__main__':
    main()
