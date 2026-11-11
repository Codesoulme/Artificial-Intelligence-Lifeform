"""Compare current/past memory answers after isolated fact corrections."""
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
from scripts.deepseek_spora_reader_bias_factorial import FRESH_OUTPUT as PRIOR
from scripts.spora_event_scan_selector import select_event
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import query
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest
from spora.memory_output import MemoryOutputLogitsProcessor

OUTPUT = ROOT / 'runs/deepseek-spora-temporal-counterfactual-audit-20261112.json'
PERSONS = (0, 3)
PAIRS = ((1, 6), (6, 3), (3, 1))
QUESTIONS = (('current', '{name}现在喜欢哪种饮料？', 0), ('past', '{name}以前喜欢哪种饮料？', 1))


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checkpoint_sha = runtime.sha(runtime.CHECKPOINT)
    if json.loads(PRIOR.read_text(encoding='utf-8'))['source_checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Prior checkpoint mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        runtime.AutoModelForCausalLM.from_pretrained = lambda *args, **kwargs: load_full_streamed()
        chat = runtime.load_chat()
        device = next(chat.organism.parameters()).device
        rows, no_history = [], []
        for person in PERSONS:
            key, name, _ = ENROLLMENTS[person]
            for old, new in PAIRS:
                chat.state = fresh_controlled_state(chat.organism)
                for value in (old, new):
                    chat.state, decision = observe_sourced(chat.organism, chat.state, key, value, 'direct', 1.0)
                    if not decision['current_write']:
                        raise ValueError('Controlled correction rejected')
                before = frozen_digest(chat.organism, chat.state)
                keys = torch.tensor([key], dtype=torch.long, device=device)
                zeros = torch.zeros_like(keys)
                ones = torch.ones_like(keys)
                with torch.no_grad():
                    now_logits = query(chat.organism, chat.state, keys, zeros, zeros)
                    old_logits = query(chat.organism, chat.state, keys, ones, zeros)
                now_top, old_top = int(now_logits.argmax(-1)[0]), int(old_logits.argmax(-1)[0])
                selection = select_event(chat.organism, chat.state, key, 'effective_previous', as_of=chat.state.cursor - 1)
                for mode, template, temporal_label in QUESTIONS:
                    question = template.format(name=name)
                    expected = new if mode == 'current' else old
                    enabled = chat.answer(question)
                    with patch.object(MemoryOutputLogitsProcessor, '__call__', lambda self, ids, scores: scores):
                        disabled = chat.answer(question)
                    row = {'person': person, 'old_value': old, 'new_value': new, 'mode': mode,
                           'expected_value': expected, 'current_top': now_top, 'past_at_zero_top': old_top,
                           'independent_previous_event': selection, 'enabled': enabled, 'disabled': disabled}
                    for label in ('enabled', 'disabled'):
                        result = row[label]
                        row[label + '_route_correct'] = bool(result['accepted'] and result.get('key') == key
                                                             and result.get('temporal_intent') == temporal_label)
                        row[label + '_answer_correct'] = bool(row[label + '_route_correct']
                                                              and result.get('answer') == WORDS[expected])
                    rows.append(row)
                    print(json.dumps({'person': person, 'old': old, 'new': new, 'mode': mode,
                                      'native_current_correct': now_top == new, 'native_past_correct': old_top == old,
                                      'selected_event': selection.get('event_time'),
                                      'enabled_reason': enabled['reason'], 'disabled_reason': disabled['reason'],
                                      'enabled_correct': row['enabled_answer_correct'],
                                      'disabled_correct': row['disabled_answer_correct']}), flush=True)
                if frozen_digest(chat.organism, chat.state) != before:
                    raise ValueError('Reading modified temporary state')
            chat.state = fresh_controlled_state(chat.organism)
            chat.state, decision = observe_sourced(chat.organism, chat.state, key, 1, 'direct', 1.0)
            if not decision['current_write']:
                raise ValueError('Single-fact initialization refused')
            result = chat.answer(QUESTIONS[1][1].format(name=name))
            no_history.append({'person': person, 'result': result})
            print(json.dumps({'person': person, 'no_history_reason': result['reason'],
                              'no_history_accepted': result['accepted']}), flush=True)
    if runtime.sha(runtime.CHECKPOINT) != checkpoint_sha:
        raise ValueError('Protected checkpoint changed')
    summary = {}
    for mode in ('current', 'past'):
        subset = [r for r in rows if r['mode'] == mode]
        summary[mode] = {'total': len(subset),
                         'native_expected_top': sum((r['current_top'] if mode == 'current' else r['past_at_zero_top'])
                                                    == r['expected_value'] for r in subset),
                         'previous_event_zero': sum(r['independent_previous_event'].get('event_time') == 0 for r in subset),
                         **{label + '_route_correct': sum(r[label + '_route_correct'] for r in subset)
                            for label in ('enabled', 'disabled')},
                         **{label + '_answer_correct': sum(r[label + '_answer_correct'] for r in subset)
                            for label in ('enabled', 'disabled')}}
    summary['no_history'] = {'total': len(no_history), 'accepted': sum(r['result']['accepted'] for r in no_history)}
    report = {'status': 'controlled_temporal_counterfactual', 'summary': summary, 'rows': rows,
              'no_history': no_history, 'source_checkpoint_sha256': checkpoint_sha,
              'seconds': time.perf_counter() - started,
              'limitations': ['Twelve authored questions for two known names and three value pairs.',
                              'Two consecutive manual writes for one key; no natural-language fact extraction.',
                              'Direct history query uses oracle event zero; runtime uses its own selected event.',
                              'Exact single-word answer matching; no synonyms or longer answers accepted.',
                              'No multiple people, delayed effective times, restarts, or real user state.',
                              'Output-bias ablation retains both decoder memory readers.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
