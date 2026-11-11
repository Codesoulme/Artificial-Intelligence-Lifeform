"""Train a domain-broader memory-read predicate gate from the v4 candidate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import deepseek_spora_read_intent_gate as trainer

trainer.OUTPUT = trainer.ROOT/'runs/deepseek-spora-read-intent-gate-v5-20261111'
trainer.INIT_CHECKPOINT = trainer.ROOT/'runs/deepseek-spora-read-intent-gate-v4-20261111/intent.pt'
trainer.EXTRA_READ = (
    '{name}记录的颜色偏好是什么？',
    '请查询{name}平时喜欢哪项运动。',
    '保存的信息里{name}从事什么工作？',
    '从记忆中找出{name}的兴趣爱好。',
    '我想知道{name}当前留下的个人资料。',
    '请回忆{name}之前记录过的其他信息。',
)
trainer.EXTRA_NON_READ = (
    '请猜猜{name}可能喜欢什么颜色。',
    '你觉得{name}适合做什么工作？',
    '替{name}写一段运动会加油词。',
    '把{name}的资料修改成新的内容。',
    '请推荐一个适合{name}的礼物。',
    '不要查询{name}的任何记忆。',
)
trainer.SEED = 29022
trainer.STEPS = 180
trainer.LEARNING_RATE = 0.0001

if __name__ == '__main__':
    trainer.run()
