"""Evaluate the saved attention candidate on additional unseen question templates."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_attention_higher_lr_pilot as configuration

pilot = configuration.pilot
pilot.RELOAD_FROM = ROOT / 'runs/deepseek-spora-attention-higher-lr-pilot-20261113'
pilot.audit.OUT = ROOT / 'runs/deepseek-spora-attention-novel-question-audit-20261113'
pilot.EVAL_NOVEL_QUESTION = True
pilot.EVAL_ATTENTION_ABLATION = True
novel = {
    'drink': '关于{name}的饮品偏好，请给出饮品的名称。',
    'color': '关于{name}的颜色偏好，请给出颜色的名称。',
    'hobby': '关于{name}的活动偏好，请给出活动的名称。',
    'fruit': '关于{name}的水果偏好，请给出水果的名称。',
}
pilot.DOMAINS = {key: (relation, (*questions, novel[key]), train, held)
                 for key, (relation, questions, train, held) in pilot.DOMAINS.items()}

if __name__ == '__main__':
    pilot.audit.main()
    if not pilot.done:
        raise ValueError('Novel-question audit did not run')
