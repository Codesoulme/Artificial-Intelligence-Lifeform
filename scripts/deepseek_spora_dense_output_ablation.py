"""Compare vocabulary bias on dense synthetic historical reads; development only."""
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_boundary_runtime_pilot import ENROLLMENTS
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.internal_memory import enable_event_key_index, enable_independent_provenance, observe_sourced
from spora.language_training import frozen_digest
from spora.memory_output import MemoryOutputLogitsProcessor

OUTPUT = ROOT / 'runs/deepseek-spora-dense-output-ablation-20261112.json'
VALUES = (1, 6, 3, 4)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_sha = runtime.sha(runtime.CHECKPOINT)
    torch.set_num_threads(2)
    start = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        if any(p.is_meta for p in chat.base.parameters()) or any(b.is_meta for b in chat.base.buffers()):
            raise ValueError('Unmaterialized layer')
        for key, name, _ in ENROLLMENTS:
            for shift in (0, 1):
                state = enable_independent_provenance(chat.organism,
                    enable_event_key_index(chat.organism, fresh_controlled_state(chat.organism)))
                for event in range(64):
                    value = VALUES[(event + shift) % len(VALUES)]
                    state, decision = observe_sourced(chat.organism, state, key, value, 'direct', 1.0)
                    if not decision['current_write']:
                        raise ValueError('Write rejected')
                chat.state = state
                digest = frozen_digest(chat.organism, state)
                question = f'{name}以前喜欢哪种饮料？'
                enabled = chat.answer(question)
                with patch.object(MemoryOutputLogitsProcessor, '__call__', lambda self, ids, scores: scores):
                    disabled = chat.answer(question)
                if frozen_digest(chat.organism, state) != digest:
                    raise ValueError('Read modified state')
                expected = WORDS[VALUES[(62 + shift) % len(VALUES)]]
                current = WORDS[VALUES[(63 + shift) % len(VALUES)]]
                row = {'name': name, 'key': key, 'shift': shift, 'expected': expected, 'current': current,
                       'enabled': enabled, 'disabled': disabled,
                       'enabled_correct': enabled.get('answer') == expected,
                       'disabled_correct': disabled.get('answer') == expected,
                       'both_selected_expected_event': all(
                           (r.get('memory_event_selection') or {}).get('event_time') == 62
                           and r.get('key') == key for r in (enabled, disabled))}
                rows.append(row)
                print(json.dumps({k: row[k] for k in ('name', 'shift', 'enabled_correct', 'disabled_correct',
                                                       'both_selected_expected_event')}, ensure_ascii=False), flush=True)
    if runtime.sha(runtime.CHECKPOINT) != source_sha:
        raise ValueError('Protected checkpoint changed')
    summary = {'cases': len(rows), 'enabled_correct': sum(r['enabled_correct'] for r in rows),
               'disabled_correct': sum(r['disabled_correct'] for r in rows),
               'both_selected_expected_event': sum(r['both_selected_expected_event'] for r in rows)}
    result = {'status': 'dense_history_development_ablation', 'summary': summary, 'rows': rows,
              'source_checkpoint_sha256': source_sha, 'seconds': time.perf_counter() - start,
              'limitations': ['Known names, repeated authored template, deterministic synthetic numeric writes.',
                              'Only two expected values and one history length; not independent holdout.',
                              'Disabled bias is temporary; other memory hooks remain active.',
                              'Correct answers alone do not establish causal memory use or general chat.',
                              'Research loader, temporary states, no original individual writes.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
