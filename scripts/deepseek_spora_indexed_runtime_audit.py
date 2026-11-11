"""Compare indexed and unindexed temporary histories through the real decoder."""
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
from scripts.deepseek_spora_delayed_fact_audit import SCENARIOS
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_indexed_history_audit import OUTPUT as PRIOR, same_neural_state
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import capacity, grow
from spora.historical_resolutions import HistoricalResolutions
from spora.internal_memory import enable_event_key_index, observe_sourced
from spora.language_training import frozen_digest
from spora.memory_output import MemoryOutputLogitsProcessor

OUTPUT = ROOT / 'runs/deepseek-spora-indexed-runtime-audit-20261112.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checkpoint_sha = runtime.sha(runtime.CHECKPOINT)
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    if (prior['source_checkpoint_sha256'] != checkpoint_sha or not prior['rows'][2]['indexed']['correct']
            or prior['rows'][2]['regular']['result']['status'] != 'abstain'):
        raise ValueError('Prior index evidence mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        if any(p.is_meta for p in chat.base.parameters()) or any(b.is_meta for b in chat.base.buffers()):
            raise ValueError('Unmaterialized replacement layer')
        device = next(chat.organism.parameters()).device
        chat.organism.historical_resolutions = HistoricalResolutions(chat.organism.config.key_count, 4).to(device)
        regular = fresh_controlled_state(chat.organism)
        indexed = enable_event_key_index(chat.organism, fresh_controlled_state(chat.organism))
        key, name, _ = ENROLLMENTS[0]
        distractor = ENROLLMENTS[3][0]

        def write_pair(write_key, value, valid_at=None):
            nonlocal regular, indexed
            if regular.cursor == capacity(chat.organism, regular):
                regular, indexed = grow(chat.organism, regular), grow(chat.organism, indexed)
            regular, a = observe_sourced(chat.organism, regular, write_key, value, 'direct', 1.0, valid_at=valid_at)
            indexed, b = observe_sourced(chat.organism, indexed, write_key, value, 'direct', 1.0, valid_at=valid_at)
            if a != b:
                raise ValueError('Write decisions diverged')

        for subject, value, valid_at in SCENARIOS[1]:
            write_pair(key if subject == 'T' else distractor, value, valid_at)
        chat.organism.historical_resolutions.record(chat.organism, indexed, key, 4, value=3,
                                                     valid_at=1, decision='confirm')
        keys = [k for k in range(10, 51) if k not in (key, distractor)]
        for i in range(128):
            write_pair(keys[i % len(keys)], (i * 7 + 2) % chat.organism.config.unknown)
            if (i + 1) % 32 == 0:
                print(json.dumps({'interference_writes': i + 1}), flush=True)
        if not same_neural_state(regular, indexed):
            raise ValueError('Index changed neural state')
        before_regular = frozen_digest(chat.organism, regular)
        before_indexed = frozen_digest(chat.organism, indexed)
        question = f'{name}以前喜欢哪种饮料？'
        chat.state = regular
        plain = chat.answer(question)
        chat.state = indexed
        indexed_answer = chat.answer(question)
        with patch.object(MemoryOutputLogitsProcessor, '__call__', lambda self, ids, scores: scores):
            indexed_no_bias = chat.answer(question)
        if (frozen_digest(chat.organism, regular) != before_regular
                or frozen_digest(chat.organism, indexed) != before_indexed):
            raise ValueError('Read modified temporary individual')
    if runtime.sha(runtime.CHECKPOINT) != checkpoint_sha:
        raise ValueError('Protected checkpoint changed')
    selection = indexed_answer.get('memory_event_selection') or {}
    summary = {'same_neural_state': True, 'unindexed_reason': plain['reason'],
               'unindexed_accepted': plain['accepted'], 'indexed_reason': indexed_answer['reason'],
               'indexed_correct_event': selection.get('event_time') == 4,
               'indexed_correct_key': indexed_answer.get('key') == key,
               'indexed_correct_answer': indexed_answer.get('answer') == WORDS[3],
               'indexed_no_bias_correct_answer': indexed_no_bias.get('answer') == WORDS[3]}
    report = {'status': 'controlled_indexed_full_runtime_audit', 'summary': summary,
              'unindexed': plain, 'indexed': indexed_answer, 'indexed_no_bias': indexed_no_bias,
              'source_checkpoint_sha256': checkpoint_sha, 'seconds': time.perf_counter() - started,
              'limitations': ['One authored question, one known person and one synthetic trajectory.',
                              'The additional event-key tensor is exact metadata, not learned neural memory.',
                              'Manually written facts and oracle confirmation; no natural-language learning.',
                              'Research-only fresh-state index, no migration of an existing individual.',
                              'Full model loaded via the experimental low-peak loader.',
                              'No real user individual, standalone checkpoint restart or concurrency evaluated.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
