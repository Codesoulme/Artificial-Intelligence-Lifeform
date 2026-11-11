"""Run selected intent contrasts through the full temporary chat path."""
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
from scripts.deepseek_spora_boundary_runtime_pilot import ENROLLMENTS
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.internal_memory import enable_event_key_index, enable_independent_provenance, observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-intent-end-to-end-audit-20261112.json'
CASES = (('standard_read', '{name}以前喜欢哪种饮料？'),
         ('read_paraphrase', '你记得{name}以前喜欢喝什么吗？'),
         ('guess', '猜猜{name}以前喜欢喝什么。'),
         ('write', '把{name}喜欢喝咖啡写进记忆。'),
         ('negation', '不要告诉我{name}喜欢喝什么。'))


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = runtime.sha(runtime.CHECKPOINT)
    torch.set_num_threads(2)
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        if any(p.is_meta for p in chat.base.parameters()) or any(b.is_meta for b in chat.base.buffers()):
            raise ValueError('Unmaterialized layer')
        for key, name, _ in ENROLLMENTS[:3]:
            state = enable_independent_provenance(chat.organism,
                enable_event_key_index(chat.organism, fresh_controlled_state(chat.organism)))
            for value in (1, 6):
                state, decision = observe_sourced(chat.organism, state, key, value, 'direct', 1.0)
                if not decision['current_write']:
                    raise ValueError('Initialization was not accepted')
            chat.state = state
            before = frozen_digest(chat.organism, state)
            for kind, template in CASES:
                result = chat.answer(template.format(name=name))
                selection = result.get('memory_event_selection') or {}
                row = {'kind': kind, 'name': name, 'key': key, 'result': result,
                       'selected_expected': bool(result.get('key') == key and selection.get('event_time') == 0),
                       'answer_is_old_value': result.get('answer') == '咖啡'}
                rows.append(row)
                print(json.dumps({'kind': kind, 'key': key, 'accepted': result['accepted'],
                                  'reason': result['reason'], 'selected_expected': row['selected_expected'],
                                  'answer_is_old_value': row['answer_is_old_value']}), flush=True)
            if frozen_digest(chat.organism, state) != before:
                raise ValueError('Read modified temporary state')
    if runtime.sha(runtime.CHECKPOINT) != sha:
        raise ValueError('Protected checkpoint changed')
    summary = {}
    for kind, _ in CASES:
        subset = [r for r in rows if r['kind'] == kind]
        summary[kind] = {'cases': len(subset), 'accepted': sum(r['result']['accepted'] for r in subset),
                         'selected_expected': sum(r['selected_expected'] for r in subset),
                         'answer_is_old_value': sum(r['answer_is_old_value'] for r in subset)}
    result = {'status': 'authored_intent_full_runtime_audit', 'summary': summary, 'rows': rows,
              'source_checkpoint_sha256': sha, 'seconds': time.perf_counter() - started,
              'limitations': ['Three known names, one old and one current beverage, five authored questions.',
                              'Guess wording can be ambiguous: routing to stored facts is observed, not a safety verdict.',
                              'Manual numeric direct writes into temporary individual, not natural-language learning.',
                              'Known questions used to choose this development audit; no independent holdout.',
                              'Research loader; no write through user entry point, no original individual change.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
