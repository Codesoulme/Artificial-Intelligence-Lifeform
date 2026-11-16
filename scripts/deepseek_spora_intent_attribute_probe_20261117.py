"""Train a non-deployed finite-relation intent probe with split templates."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import deepseek_spora_intent_contrast_probe as probe

probe.OUT = ROOT / 'runs/deepseek-spora-intent-attribute-probe-20261117'
probe.SEED = 117113
probe.NAMES = {'train': ('阿川', '小岚', '小渝', '阿诺', '小言', '阿迪', '小泽', '小秋'),
               'cal': ('阿霖', '小溪', '阿南', '小恒'),
               'test': ('小茗', '阿辰', '小桐', '阿源')}
probe.TEMPLATES = {
    'train': (
        ('请回答{n}最喜欢喝哪一种饮品。', True),
        ('只说{n}爱喝的饮料名称。', True),
        ('查找{n}记录中的饮品偏好。', True),
        ('不说颜色，请告诉我{n}爱喝什么饮料。', True),
        ('{n}喜欢的饮料是红色还是绿色？', False),
        ('{n}喜欢的饮品售价是几元？', False),
        ('{n}爱喝的饮料通常是什么味道？', False),
        ('列出{n}喜欢的饮料的配料。', False),
        ('请把{n}喜欢喝哪种饮品翻译成英文。', False),
        ('不要告诉我{n}喜欢喝什么饮料。', False),
        ('把{n}的饮料改成茶。', False),
        ('写一个关于{n}和饮品的故事。', False),
    ),
    'cal': (
        ('请查询{n}偏爱的饮品是哪一个。', True),
        ('别提价格，回答{n}爱喝哪种饮料。', True),
        ('{n}的饮品偏好是什么？', True),
        ('{n}偏爱的饮料是冷的还是热的？', False),
        ('{n}爱喝的饮品含糖量有多高？', False),
        ('把“{n}爱喝什么饮料”原样抄写。', False),
        ('不要查询{n}的饮品记录。', False),
        ('更正{n}的饮料偏好。', False),
    ),
    'test': (
        ('记录里的{n}目前最爱喝什么饮料？', True),
        ('不需要价格，只列出{n}偏爱的饮品名称。', True),
        ('告诉我{n}喝什么。', True),
        ('{n}喜欢的饮料看上去透明吗？', False),
        ('{n}喜欢的饮品通常含有什么原料？', False),
        ('描述{n}喜欢喝的饮料的口感。', False),
        ('不要说出{n}的饮品喜好。', False),
        ('给{n}写一首有关饮品的诗。', False),
    ),
}

if __name__ == '__main__':
    probe.main()
