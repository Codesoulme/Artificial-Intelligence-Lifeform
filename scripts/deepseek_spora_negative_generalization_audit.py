"""Check that the promoted v4 runtime rejects varied non-memory requests."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.deepseek_named_binding_probe import KEYS
from scripts.deepseek_spora_runtime_chat import load_chat
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.internal_memory import observe_sourced

OUTPUT = ROOT/'runs/deepseek-spora-negative-generalization-audit-v2-20261111.json'
QUESTIONS = (
    '小林最喜欢什么颜色？',
    '小林平时喜欢做什么运动？',
    '小林的饮料记录是谁告诉你的？',
    '把小林的饮料偏好修改成牛奶。',
    '请为小林写一首短诗。',
    '请给小林一句生日祝福。',
    '小林和阿宁各自喜欢什么？',
    '请分别查询小林与小雨的饮料。',
    '小雪过去喜欢喝什么？',
    '小林的饮料偏好是否可靠？',
    '解释一下你为什么能记住信息。',
    '请总结我们之前讨论过的所有话题。',
    '我今天很累，给我一个放松建议。',
    '你能帮我设计一个饮料广告吗？',
    '小林现在的工作是什么？',
    '如果小林喜欢茶，请提醒我。',
)


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    chat = load_chat()
    chat.state = fresh_controlled_state(chat.organism)
    for value in (1, 6):
        chat.state, decision = observe_sourced(chat.organism, chat.state, KEYS[0], value, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('Controlled setup write rejected')
    rows = [chat.answer(question) for question in QUESTIONS]
    report = {'status': 'promoted_v4_negative_generalization_audit',
              'total': len(rows), 'rejected': sum(not row['accepted'] for row in rows),
              'rows': rows,
              'limitations': ['Manually authored finite negative set.',
                              'One controlled memory entity and beverage facts.',
                              'This does not establish adversarial safety or open-domain understanding.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'total': report['total'], 'rejected': report['rejected']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
