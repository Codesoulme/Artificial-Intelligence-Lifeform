"""Full decoder probe of an explicitly confirmed delayed historical fact."""
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
from scripts.deepseek_spora_delayed_fact_audit import SCENARIOS, load_individual_from_verified_checkpoint
from scripts.deepseek_spora_delayed_runtime_guard_audit import OUTPUT as PRIOR
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_event_scan_selector import select_event
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.historical_resolutions import HistoricalResolutions
from spora.integrated_checkpoint import save_integrated
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest
from spora.memory_output import MemoryOutputLogitsProcessor

OUTPUT = ROOT / 'runs/deepseek-spora-resolution-runtime-pilot-20261112.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checkpoint_sha = runtime.sha(runtime.CHECKPOINT)
    previous = json.loads(PRIOR.read_text(encoding='utf-8'))
    if previous['source_checkpoint_sha256'] != checkpoint_sha or previous['summary']['past_guarded'] != 4:
        raise ValueError('Prior abstention evidence mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        if any(p.is_meta for p in chat.base.parameters()) or any(b.is_meta for b in chat.base.buffers()):
            raise ValueError('Replacement layers were not fully materialized')
        key, name, _ = ENROLLMENTS[0]
        distractor = ENROLLMENTS[3][0]
        chat.organism.historical_resolutions = HistoricalResolutions(chat.organism.config.key_count, 4).to(
            next(chat.organism.parameters()).device)
        chat.state = fresh_controlled_state(chat.organism)
        for event, (subject, value, valid_at) in enumerate(SCENARIOS[1]):
            chat.state, decision = observe_sourced(chat.organism, chat.state, key if subject == 'T' else distractor,
                                                   value, 'direct', 1.0, valid_at=valid_at)
        prior_answer = chat.answer(f'{name}以前喜欢哪种饮料？')
        before_memory = [tensor.detach().clone() for tensor in chat.state.current]
        chat.organism.historical_resolutions.record(chat.organism, chat.state, key, 4,
                                                    value=3, valid_at=1, decision='confirm')
        current = chat.answer(f'{name}现在喜欢哪种饮料？')
        past = chat.answer(f'{name}以前喜欢哪种饮料？')
        with patch.object(MemoryOutputLogitsProcessor, '__call__', lambda self, ids, scores: scores):
            past_no_bias = chat.answer(f'{name}以前喜欢哪种饮料？')
        current_unchanged = all(torch.equal(a, b) for a, b in zip(before_memory, chat.state.current))
        print(json.dumps({'phase': 'before_integrated_save', 'prior_abstained': not prior_answer['accepted'],
                          'current_accepted': current['accepted'], 'past_accepted': past['accepted'],
                          'past_event': (past.get('memory_event_selection') or {}).get('event_time'),
                          'past_answer_correct': past.get('answer') == WORDS[3],
                          'past_no_bias_correct': past_no_bias.get('answer') == WORDS[3],
                          'current_memory_unchanged': current_unchanged}), flush=True)
        memory_digest = frozen_digest(chat.organism, chat.state)
        path = Path(directory) / 'explicit-confirmation-integrated.pt'
        save_integrated(path, chat.organism, chat.state, chat.decoder.reader, runtime.MODEL,
                        chat.decoder.layer_indices, replacements=chat._replacements(),
                        output_adapter=chat.decoder.output_adapter)
        loaded, restored = load_individual_from_verified_checkpoint(path)
        restored_digest = frozen_digest(loaded, restored)
        selection = select_event(loaded, restored, key, 'effective_previous', as_of=restored.cursor - 1)
        saved_resolution = loaded.historical_resolutions.lookup(key, 4)
        reload_same = memory_digest == restored_digest and selection.get('event_time') == 4
    if runtime.sha(runtime.CHECKPOINT) != checkpoint_sha:
        raise ValueError('Protected checkpoint changed')
    summary = {'prior_abstained': not prior_answer['accepted'] and prior_answer['reason'] == 'memory_event_abstain',
               'current_key_correct': current.get('key') == key,
               'current_answer_correct': current.get('answer') == WORDS[6],
               'past_event_correct': (past.get('memory_event_selection') or {}).get('event_time') == 4,
               'past_answer_correct': past.get('answer') == WORDS[3],
               'past_no_bias_correct': past_no_bias.get('answer') == WORDS[3],
               'current_memory_unchanged': current_unchanged, 'integrated_individual_reload_same': reload_same}
    report = {'status': 'temporary_explicit_resolution_runtime_pilot', 'summary': summary,
              'prior': prior_answer, 'current': current, 'past': past, 'past_no_bias': past_no_bias,
              'restored_selection': selection, 'restored_resolution': saved_resolution,
              'source_checkpoint_sha256': checkpoint_sha, 'seconds': time.perf_counter() - started,
              'limitations': ['One synthetic scenario and one known identity; explicit oracle confirms the late fact.',
                              'Exact tensor metadata, not learned truth or autonomous confirmation.',
                              'Prototype record mutates in place and is not exposed in the default runtime.',
                              'Integrated file was written to a temporary directory; only individual tensors were reloaded.',
                              'No fresh-process reload, crash, concurrency, natural-language update or real user state tested.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
