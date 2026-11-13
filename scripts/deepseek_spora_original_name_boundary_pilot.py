"""Train name-boundary contrasts without binding longer names to old keys."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

from scripts import deepseek_spora_original_layer_name_router_expanded as expanded

pilot = expanded.pilot
pilot.OUT = ROOT / 'runs/deepseek-spora-original-name-boundary-pilot-20261114'
pilot.LOCATOR_ONLY = tuple((name + suffix, template.format(name=name + suffix))
                           for name in pilot.NAMES for suffix in ('成', '鹏')
                           for template in ('{name}现在喜欢喝什么饮料？', '请查{name}的饮品偏好。'))
pilot.NEAR += tuple(name + '晨' for name in pilot.NAMES)


if __name__ == '__main__':
    pilot.main()
