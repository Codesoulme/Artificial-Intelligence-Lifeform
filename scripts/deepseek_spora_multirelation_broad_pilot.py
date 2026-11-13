"""Broader controlled fact coverage with separate same/alternate question tests."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_multirelation_train_pilot as pilot

pilot.audit.OUT = ROOT / 'runs/deepseek-spora-multirelation-broad-pilot-20261113'
pilot.STEPS = 288
pilot.HELD_VALUE_Q = 0
pilot.EVAL_ALTERNATIVE_QUESTION = True
pilot.DOMAINS = {
    'drink': ('喜欢喝', ('请只回答{name}喜欢喝什么饮品。', '{name}喜欢喝什么？只说饮品名称。'),
              ('茶', '咖啡', '牛奶', '豆浆', '果汁', '可乐'), ('红茶', '酸奶')),
    'color': ('喜欢的颜色是', ('请只回答{name}喜欢的颜色。', '{name}喜欢哪种颜色？只写颜色。'),
              ('红色', '蓝色', '绿色', '黄色', '白色', '黑色'), ('紫色', '粉色')),
    'hobby': ('喜欢的活动是', ('请只回答{name}喜欢的活动。', '{name}喜欢做什么活动？只写活动。'),
              ('读书', '跑步', '画画', '唱歌', '写字', '跳舞'), ('游泳', '下棋')),
    'fruit': ('喜欢的水果是', ('请只回答{name}喜欢的水果。', '{name}喜欢哪种水果？只写名称。'),
              ('苹果', '香蕉', '西瓜', '葡萄', '草莓', '桃子'), ('菠萝', '柚子')),
}

if __name__ == '__main__':
    pilot.audit.main()
    if not pilot.done:
        raise ValueError('Broad pilot did not run')
