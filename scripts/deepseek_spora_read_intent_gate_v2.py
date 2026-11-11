"""Add varied historical-read and non-read expressions to the intent gate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import deepseek_spora_read_intent_gate as trainer

trainer.OUTPUT = trainer.ROOT/'runs/deepseek-spora-read-intent-gate-v2-20261111'
trainer.EXTRA_READ = ('{name}更早记录的饮品是什么？', '请查找{name}较早的饮料偏好。',
                      '说说之前记下的{name}爱喝什么。', '在保存的信息中查询{name}的当前偏好。')
trainer.EXTRA_NON_READ = ('不需要查阅{name}的饮料记录。', '关于{name}的饮品记录是谁写的？',
                          '帮{name}想一句温暖的问候。', '把{name}的记录修改成牛奶。')
trainer.SEED = 29019

if __name__ == '__main__':
    trainer.run()
