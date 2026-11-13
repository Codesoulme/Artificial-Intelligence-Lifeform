"""Train a wider-context original-layer name router for controlled comparison."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

from scripts import deepseek_spora_original_layer_name_router as pilot

pilot.OUT = ROOT / 'runs/deepseek-spora-original-layer-name-router-expanded-20261114'
pilot.TRAIN += (
    '你能回忆{name}爱喝的饮料吗？',
    '说说{name}平常喝哪种饮品。',
    '在过去的记录里，{name}最喜欢什么饮料？',
    '请帮我找{name}目前偏爱的饮品。',
    '能否查一下{name}平常喝的是什么？',
    '现在想知道{name}选的饮料是什么。',
)
pilot.HELD += (
    '据保存的偏好，{name}通常想喝什么饮品？',
    '请查一下{name}记录里的饮料名称。',
)


if __name__ == '__main__':
    pilot.main()
