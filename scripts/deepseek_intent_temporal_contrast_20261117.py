"""Isolated intent probe with temporal, negation and operation contrasts.

Artificial templates are not an open-dialogue dataset. The fixed historical
challenge has already informed this development set and is not final testing.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import deepseek_spora_intent_contrast_probe as probe

probe.OUT = ROOT / 'runs/deepseek-spora-intent-temporal-contrast-20261117'
probe.SEED = 117211
probe.NAMES = {'train': ('小岚', '阿川', '小渝', '阿诺', '小言', '阿迪', '小泽', '小秋'),
               'cal': ('阿霖', '小溪', '阿南', '小恒'),
               'test': ('小茗', '阿辰', '小桐', '阿源')}
probe.TEMPLATES = {
    'train': (
        ('请告诉我{n}现在最爱喝的饮料名称。', True),
        ('查一下{n}修改以前偏爱的饮料。', True),
        ('{n}之前喜欢的饮品是哪种？', True),
        ('根据记录回答{n}如今喝什么。', True),
        ('不说价格，只说{n}原先喜欢喝哪种饮品。', True),
        ('{n}以前喜欢的饮料叫什么名字？', True),
        ('不要透露{n}喜欢的饮料名称。', False),
        ('删除{n}的饮品偏好记录。', False),
        ('更新{n}的饮料偏好为茶。', False),
        ('{n}喜欢的饮品是甜的吗？', False),
        ('{n}偏爱的饮料价格是多少？', False),
        ('{n}喜欢的颜色是什么？', False),
        ('翻译“{n}喜欢什么饮料”。', False),
        ('给{n}写关于喝茶的小故事。', False),
    ),
    'cal': (
        ('说说{n}过去喜欢喝哪款饮品。', True),
        ('{n}改口之前爱喝什么？', True),
        ('只回答{n}目前的饮料喜好。', True),
        ('请查找{n}以前喜欢的饮料名称。', True),
        ('请不要公布{n}的饮料喜好。', False),
        ('清除{n}保存的饮料记录。', False),
        ('{n}喜欢的饮料闻起来怎么样？', False),
        ('告诉我{n}最爱哪部电影。', False),
        ('复述问题“{n}喝什么”。', False),
        ('替{n}推荐一款饮品。', False),
    ),
    'test': (
        ('查阅{n}现在喜欢的饮料是什么。', True),
        ('回忆{n}更改想法前的饮品。', True),
        ('先前{n}偏爱喝哪一种？', True),
        ('把{n}原来的饮品名称告诉我。', True),
        ('不用回答{n}的饮料偏好。', False),
        ('去掉{n}的饮品偏好记录。', False),
        ('{n}喜欢的饮料是什么颜色？', False),
        ('查询{n}喜欢的运动。', False),
        ('把{n}的喜好改成咖啡。', False),
        ('写一首关于{n}和饮料的诗。', False),
    ),
}


if __name__ == '__main__':
    probe.main()
