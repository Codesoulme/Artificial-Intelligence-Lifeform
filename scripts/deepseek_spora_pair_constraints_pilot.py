"""Continue a paired-layer candidate with manually authored constraint examples."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import deepseek_spora_pair_general_replay as trainer

trainer.PAIR = ROOT / 'runs/deepseek-spora-pair-general-replay-20261112'
trainer.REFERENCE = trainer.PAIR / 'result.json'
trainer.REFERENCE_PAIR_FIELD = 'candidate_sha256'
trainer.OUTPUT = ROOT / 'runs/deepseek-spora-pair-constraints-pilot-20261112'
trainer.STEPS = 160
trainer.SEED = 88042
trainer.CONSTRAINT_EXAMPLES = (
    ('列出两种形状，不要出现圆形。', '正方形、三角形。'),
    ('写两句问候语，不要出现数字。', '早上好。晚上好。'),
    ('列出两种出行方式，不要提汽车。', '步行、骑自行车。'),
    ('用一句不含雨字的话描述晴天。', '阳光明媚，天空湛蓝。'),
    ('只用一个数字回答：五减二等于几？', '3'),
    ('推荐两种水果，不要香蕉。', '苹果、梨。'),
    ('用两个字描写安静的环境。', '宁静'),
    ('说两种家畜，不要提狗。', '牛、羊。'),
    ('只写两个正整数，它们相加等于十。', '4、6'),
    ('写一句不含猫字的鸟儿描写。', '小鸟站在枝头歌唱。'),
    ('写一个包含山字的地理词。', '山谷'),
    ('给出两种文具，不要说铅笔。', '橡皮、尺子。'),
    ('只列两个季节，不要夏天。', '春天、秋天。'),
    ('说两种颜色，不能包含蓝色。', '红色、黄色。'),
    ('用一句不含水字的话描述沙漠。', '沙丘在阳光下延伸。'),
    ('只回答一个数字：二加四是多少？', '6'),
)


if __name__ == '__main__':
    trainer.main()
