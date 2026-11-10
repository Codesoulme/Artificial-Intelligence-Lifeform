"""Continue memory-answer distillation with broader phrasing coverage."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_memory_answer_distill as trainer

trainer.START_CANDIDATES = {
    6: trainer.OUTPUT / 'layer-6.pt',
    12: trainer.OUTPUT / 'layer-12.pt',
}
trainer.OUTPUT = trainer.ROOT / 'runs/deepseek-spora-memory-answer-distill-v2-20261111'
trainer.STEPS = 120
trainer.SEED = 92053
trainer.TRAIN_TEMPLATES = {
    'current': (
        '{name}现在喜欢什么饮品？',
        '目前{name}偏爱喝什么？',
        '{name}最中意喝哪一款？',
        '关于饮料，{name}最新喜欢什么？',
        '给{name}选饮品的话，选什么？',
    ),
    'past': (
        '{name}过去喜欢哪种饮料？',
        '更正前{name}喜欢喝什么？',
        '偏好更新之前，{name}喜欢饮用哪种？',
        '更正记录之前的答案是什么？',
        '早些时候{name}爱喝什么？',
    ),
    'later': (
        '更正后{name}改喜欢什么饮品？',
        '{name}后来最喜欢喝什么？',
        '替换旧偏好之后，{name}喜欢哪一种？',
        '将最新记录作为准，{name}偏好什么饮料？',
        '这次更正后，适合给{name}什么饮品？',
    ),
}


if __name__ == '__main__':
    trainer.run()
