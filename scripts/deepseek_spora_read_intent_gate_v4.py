"""Conservatively adapt the validated v2 intent gate to new read wording."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import deepseek_spora_read_intent_gate as trainer

trainer.OUTPUT = trainer.ROOT/'runs/deepseek-spora-read-intent-gate-v4-20261111'
trainer.INIT_CHECKPOINT = trainer.ROOT/'runs/deepseek-spora-read-intent-gate-v2-20261111/intent.pt'
trainer.EXTRA_READ = (
    '我想知道{name}眼下爱喝哪种东西。',
    '能否从记录里找出{name}此刻喝的饮品？',
    '调出{name}最新一条饮品偏好。',
    '请检索{name}当前保留的饮料信息。',
    '有关{name}的较早饮品记录，请查一下。',
    '回看{name}上一次改动前的饮料。',
    '查找{name}更新之后保存的饮品。',
    '从历史里找出{name}后来选的饮料。',
)
trainer.EXTRA_NON_READ = (
    '不用打开{name}的饮品记忆。',
    '请不要检索{name}的饮料信息。',
    '替{name}构思一段生日祝词。',
    '把{name}的偏好改写成一条新记录。',
    '我只想听{name}的近况，不用查饮料。',
    '你认为{name}会不会喜欢茶？',
)
trainer.SEED = 29021
trainer.STEPS = 180
trainer.LEARNING_RATE = 0.0001

if __name__ == '__main__':
    trainer.run()
