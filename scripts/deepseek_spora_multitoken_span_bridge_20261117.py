"""Multi-token values, fixed-slot memory and opt-in span-gradient diagnostics."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts.deepseek_spora_multirelation_span_bridge_20261117 import pilot

pilot.OUT = ROOT / 'runs/deepseek-spora-multitoken-span-bridge-64-20261117'
pilot.WORDS = ('气泡水', '柠檬水', '葡萄汁', '苹果汁', '奶昔', '可可奶',
               '天蓝色', '浅绿色', '银灰色', '深紫色', '淡粉色', '象牙白',
               '踢足球', '打羽毛球', '弹吉他', '骑自行车', '练瑜伽', '逛公园')
pilot.TRAIN_VALUES = (0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15)
pilot.EVALUATION_VALUES = (4, 5, 10, 11, 16, 17)
pilot.SPAN_GRADIENT_DIAGNOSTIC = True


if __name__ == '__main__':
    pilot.main()
