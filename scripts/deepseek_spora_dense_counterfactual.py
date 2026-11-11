"""Isolate one historical write and ablate reader and vocabulary bias."""
from contextlib import ExitStack
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
from spora.growing import query
from spora.internal_memory import enable_event_key_index, enable_independent_provenance, observe_sourced
from spora.language_training import frozen_digest
from spora.memory_output import MemoryOutputLogitsProcessor

OUTPUT = ROOT / 'runs/deepseek-spora-dense-counterfactual-20261112.json'
VALUES = (1, 3, 4, 6)
TEMPLATES = ('{name}以前喜欢哪种饮料？', '你记得{name}以前喜欢喝什么吗？')
MODES = (('both', True, True), ('reader_only', True, False),
         ('bias_only', False, True), ('neither', False, False))


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checkpoint_sha = runtime.sha(runtime.CHECKPOINT)
    torch.set_num_threads(2)
    start = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        if any(p.is_meta for p in chat.base.parameters()) or any(b.is_meta for b in chat.base.buffers()):
            raise ValueError('Unmaterialized layer')
        key, name, _ = ENROLLMENTS[0]
        for target in VALUES:
            state = enable_independent_provenance(chat.organism,
                enable_event_key_index(chat.organism, fresh_controlled_state(chat.organism)))
            decisions = []
            for event in range(64):
                value = target if event == 62 else (0 if event == 63 else VALUES[event % len(VALUES)])
                state, decision = observe_sourced(chat.organism, state, key, value, 'direct', 1.0)
                decisions.append(decision)
            chat.state = state
            digest = frozen_digest(chat.organism, state)
            keys = torch.tensor([key], dtype=torch.long, device=next(chat.organism.parameters()).device)
            with torch.no_grad():
                historical_value = int(query(chat.organism, state, keys, torch.ones_like(keys),
                                             torch.full_like(keys, 62)).argmax(-1)[0])
            for ti, template in enumerate(TEMPLATES):
                row = {'target': target, 'expected': WORDS[target], 'template': ti,
                       'historical_value': historical_value, 'target_write': decisions[62],
                       'current_write': decisions[63], 'modes': {}}
                for mode, reader_on, bias_on in MODES:
                    with ExitStack() as stack:
                        if reader_on:
                            reader = stack.enter_context(patch.object(chat.decoder.reader, 'forward',
                                                                      wraps=chat.decoder.reader.forward))
                        else:
                            reader = stack.enter_context(patch.object(chat.decoder.reader, 'forward',
                                side_effect=lambda hidden, *args, **kwargs: hidden))
                        if not bias_on:
                            stack.enter_context(patch.object(MemoryOutputLogitsProcessor, '__call__',
                                                             lambda self, ids, scores: scores))
                        result = chat.answer(template.format(name=name))
                    selected = (result.get('memory_event_selection') or {}).get('event_time') == 62
                    if result.get('accepted') and reader.call_count == 0:
                        raise ValueError('Reader intervention not exercised')
                    row['modes'][mode] = {'result': result, 'reader_calls': reader.call_count,
                                          'selected': selected and result.get('key') == key,
                                          'correct': bool(result.get('accepted') and selected
                                                          and result.get('key') == key
                                                          and result.get('answer') == WORDS[target])}
                rows.append(row)
                print(json.dumps({'target': target, 'template': ti, 'historical_value': historical_value,
                                  **{m: row['modes'][m]['correct'] for m, _, _ in MODES}}), flush=True)
            if frozen_digest(chat.organism, state) != digest:
                raise ValueError('Read modified state')
    if runtime.sha(runtime.CHECKPOINT) != checkpoint_sha:
        raise ValueError('Protected checkpoint changed')
    summary = {'cases': len(rows), 'historical_value_matches': sum(r['historical_value'] == r['target'] for r in rows)}
    for mode, _, _ in MODES:
        summary[mode] = {'correct': sum(r['modes'][mode]['correct'] for r in rows),
                         'selected': sum(r['modes'][mode]['selected'] for r in rows),
                         'accepted': sum(bool(r['modes'][mode]['result'].get('accepted')) for r in rows)}
    result = {'status': 'dense_history_counterfactual_development', 'summary': summary, 'rows': rows,
              'source_checkpoint_sha256': checkpoint_sha, 'seconds': time.perf_counter() - start,
              'limitations': ['One known name, four values, two authored templates, one 64-event trajectory.',
                              'Only event 62 changes; event 63 has fixed current value.',
                              'Interventions affect reader sites jointly; routing and selection remain active.',
                              'Finite numeric direct writes and exact answer matching, not open-domain conversation.',
                              'Research loader and temporary states; original individual unchanged.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
