"""Full-model comparison for opt-in provenance precision on a dense history."""
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
from scripts.deepseek_spora_independent_provenance_state_audit import OUTPUT as PRIOR, equal_value_state
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.internal_memory import enable_event_key_index, enable_independent_provenance, observe_sourced
from spora.language_training import frozen_digest
from spora.memory_output import MemoryOutputLogitsProcessor

OUTPUT = ROOT / 'runs/deepseek-spora-independent-provenance-runtime-20261112.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_sha = runtime.sha(runtime.CHECKPOINT)
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    if prior['source_checkpoint_sha256'] != source_sha or prior['summary']['64']['candidate_correct'] != 12:
        raise ValueError('Prior state evidence mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        if any(p.is_meta for p in chat.base.parameters()) or any(b.is_meta for b in chat.base.buffers()):
            raise ValueError('Unmaterialized replacement layer')
        regular = enable_event_key_index(chat.organism, fresh_controlled_state(chat.organism))
        candidate = enable_independent_provenance(chat.organism,
                                                  enable_event_key_index(chat.organism, fresh_controlled_state(chat.organism)))
        key, name, _ = ENROLLMENTS[0]
        for event in range(64):
            value = (1, 6, 3, 4)[event % 4]
            regular, old = observe_sourced(chat.organism, regular, key, value, 'direct', 1.0)
            candidate, new = observe_sourced(chat.organism, candidate, key, value, 'direct', 1.0)
            if old != new:
                raise ValueError('Write decisions diverged')
        if not equal_value_state(regular, candidate):
            raise ValueError('Value memory differed')
        before_regular, before_candidate = frozen_digest(chat.organism, regular), frozen_digest(chat.organism, candidate)
        question = f'{name}以前喜欢哪种饮料？'
        chat.state = regular
        regular_answer = chat.answer(question)
        chat.state = candidate
        candidate_answer = chat.answer(question)
        with patch.object(MemoryOutputLogitsProcessor, '__call__', lambda self, ids, scores: scores):
            candidate_no_bias = chat.answer(question)
        if (frozen_digest(chat.organism, regular) != before_regular
                or frozen_digest(chat.organism, candidate) != before_candidate):
            raise ValueError('Reading changed temporary state')
    if runtime.sha(runtime.CHECKPOINT) != source_sha:
        raise ValueError('Protected checkpoint changed')
    selected = candidate_answer.get('memory_event_selection') or {}
    summary = {'value_state_equal': True, 'regular_reason': regular_answer['reason'],
               'regular_accepted': regular_answer['accepted'], 'candidate_reason': candidate_answer['reason'],
               'candidate_key_correct': candidate_answer.get('key') == key,
               'candidate_event_correct': selected.get('event_time') == 62,
               'candidate_answer_correct': candidate_answer.get('answer') == WORDS[3],
               'candidate_no_bias_correct': candidate_no_bias.get('answer') == WORDS[3]}
    report = {'status': 'temporary_independent_provenance_runtime_pilot', 'summary': summary,
              'regular': regular_answer, 'candidate': candidate_answer, 'candidate_no_bias': candidate_no_bias,
              'source_checkpoint_sha256': source_sha, 'seconds': time.perf_counter() - started,
              'limitations': ['One known identity, one synthetic question and 64 controlled numerical writes.',
                              'The fresh state uses an exact event-key tensor index and explicit direct observations.',
                              'Candidate independent precision changes provenance writing, not model parameters.',
                              'Bias suppression only for diagnostic comparison; default output behavior is unchanged.',
                              'Research-only loader and state; no real user data, natural-language writing or restart.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
