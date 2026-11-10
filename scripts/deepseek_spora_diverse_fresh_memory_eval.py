"""Evaluate both pairs on fresh entities, addresses, assignments and wording."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import deepseek_spora_natural_distill_memory_eval as probe

probe.NEW = probe.ROOT/'runs/deepseek-spora-diverse-memory-replay-20261111'
probe.OUTPUT = probe.ROOT/'runs/deepseek-spora-diverse-fresh-memory-eval-20261111.json'
probe.FACTS = (('林默', 7, 0, 4), ('桐雨', 8, 2, 7), ('星河', 9, 1, 5),
               ('柳鸣', 11, 3, 6), ('若竹', 12, 5, 2), ('柯岚', 13, 7, 1))
probe.TEMPLATES = {'current': '请说出{name}目前记录的饮品偏好。',
                   'past': '在记录修改以前，{name}偏爱的饮料是什么？'}

if __name__ == '__main__':
    probe.run()
