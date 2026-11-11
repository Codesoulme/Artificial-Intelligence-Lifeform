"""Probe the unified route entry on controlled current/history and negatives."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.deepseek_named_binding_probe import KEYS
from scripts.deepseek_spora_runtime_chat import load_chat
from spora.internal_memory import observe_sourced

OUTPUT = ROOT/'runs/deepseek-spora-runtime-chat-probe-20261111.json'


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    chat = load_chat()
    state = chat.state
    for value in (1, 6):
        state, decision = observe_sourced(chat.organism, state, KEYS[0], value, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('Controlled probe write rejected')
    chat.state = state
    questions = (
        '小林现在喜欢喝什么？',
        '小林改口以前喜欢什么饮料？',
        '小林后来喜欢喝什么？',
        '请给小林写一句鼓励的话。',
    )
    rows = [chat.answer(question) for question in questions]
    report = {'status': 'unified_route_entry_controlled_probe', 'rows': rows,
              'accepted': sum(row['accepted'] for row in rows),
              'rejected': sum(not row['accepted'] for row in rows),
              'limitations': ['One controlled entity and two finite values.',
                              'The key/value writes are supplied by the probe.',
                              'This is not an open-world or independent language evaluation.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'accepted': report['accepted'], 'rejected': report['rejected']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
