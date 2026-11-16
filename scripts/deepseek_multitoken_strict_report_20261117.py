"""Apply conservative full-fact scoring to multi-token research results."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.deepseek_multirelation_strict_report_20261117 import score_phase

VALUES = {
    **{x: '饮品' for x in ('气泡水', '柠檬水', '葡萄汁', '苹果汁', '奶昔', '可可奶')},
    **{x: '颜色' for x in ('天蓝色', '浅绿色', '银灰色', '深紫色', '淡粉色', '象牙白')},
    **{x: '活动' for x in ('踢足球', '打羽毛球', '弹吉他', '骑自行车', '练瑜伽', '逛公园')},
}
DEFAULT_RESULT = ROOT / 'runs/deepseek-spora-multitoken-span-bridge-64-20261117/result.json'


def main():
    if len(sys.argv) > 2:
        raise ValueError('Usage: script [result.json]')
    path = Path(sys.argv[1]) if len(sys.argv) == 2 else DEFAULT_RESULT
    data = json.loads(path.read_text(encoding='utf-8'))
    print(json.dumps({name: score_phase(data[name], VALUES) for name in ('before', 'after')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
