"""Audit the default runtime against near-name aliasing in a temporary state."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_named_binding_probe import KEYS
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.internal_memory import observe_sourced

OUTPUT = ROOT / 'runs/deepseek-spora-default-near-name-audit-20261111.json'

# Near-name negatives are distinct people by the experimental setup. The
# expected key is for scoring only and is never passed into the query.
CASES = [('known', '小林现在喜欢喝什么？'),
         ('known', '小林目前最喜欢哪种饮料？'),
         ('unknown', '小林林现在喜欢喝什么？'),
         ('unknown', '林小林现在喜欢喝什么？'),
         ('unknown', '小林子现在喜欢喝什么？'),
         ('unknown', '小林娜现在喜欢喝什么？'),
         ('unknown', '小林林目前最喜欢哪种饮料？'),
         ('unknown', '林小林目前最喜欢哪种饮料？'),
         ('unknown', '小林子目前最喜欢哪种饮料？'),
         ('unknown', '小林娜目前最喜欢哪种饮料？')]


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.set_num_threads(2)
    source_hash = runtime.sha(runtime.CHECKPOINT)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = runtime.load_chat()
        chat.state = fresh_controlled_state(chat.organism)
        chat.state, decision = observe_sourced(chat.organism, chat.state, KEYS[0], 6, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('Controlled memory initialization refused')
        rows = []
        for group, question in CASES:
            result = chat.answer(question)
            rows.append({'group': group, **result})
            print(json.dumps({'group': group, 'question': question, 'accepted': result['accepted'],
                              'reason': result['reason'], 'key': result.get('key')}, ensure_ascii=False), flush=True)
    if runtime.sha(runtime.CHECKPOINT) != source_hash:
        raise ValueError('Bootstrap checkpoint changed')
    summary = {'known_total': sum(r['group'] == 'known' for r in rows),
               'known_correct_key': sum(r['group'] == 'known' and r['accepted'] and r.get('key') == KEYS[0] for r in rows),
               'unknown_total': sum(r['group'] == 'unknown' for r in rows),
               'unknown_false_accept': sum(r['group'] == 'unknown' and r['accepted'] for r in rows)}
    report = {'status': 'isolated_default_runtime_near_name_audit', 'summary': summary, 'rows': rows,
              'source_checkpoint_sha256': source_hash, 'elapsed_seconds': time.perf_counter() - started,
              'limitations': ['Ten authored queries and one known memory; not a population estimate.',
                              'Near names are treated as separate entities by the experiment.',
                              'No live user state changed and no language parsing write was performed.',
                              'The source checkpoint and default gate combination are used without modification.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
