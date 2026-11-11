"""Audit current and previous reads with interleaved controlled memory events."""
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
from scripts.deepseek_spora_temporal_counterfactual_audit import OUTPUT as PRIOR
from scripts.spora_event_scan_selector import select_event
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import query
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest
from spora.memory_output import MemoryOutputLogitsProcessor

OUTPUT = ROOT / 'runs/deepseek-spora-temporal-interleaved-audit-20261112.json'
# Sequence entries use T for the asked-about person and D for another person.
SCENARIOS = (
    (('T', 1), ('D', 2), ('T', 6)),
    (('D', 2), ('T', 1), ('D', 5), ('T', 6)),
    (('T', 1), ('T', 6), ('D', 2), ('T', 3)),
)
QUESTIONS = (('current', '{name}目前喜欢哪种饮料？', 0), ('past', '{name}之前喜欢哪种饮料？', 1))


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
        rows = []
        for person in (0, 3):
            key, name, _ = ENROLLMENTS[person]
            distractor = ENROLLMENTS[3 if person == 0 else 0][0]
            for scenario_index, scenario in enumerate(SCENARIOS):
                chat.state = fresh_controlled_state(chat.organism)
                events = []
                for event_index, (subject, value) in enumerate(scenario):
                    write_key = key if subject == 'T' else distractor
                    chat.state, decision = observe_sourced(chat.organism, chat.state, write_key, value, 'direct', 1.0)
                    if not decision['current_write']:
                        raise ValueError('Controlled write rejected')
                    if subject == 'T':
                        events.append((event_index, value))
                if len(events) < 2:
                    raise ValueError('Need two target events')
                past_time, past_value = events[-2]
                _, current_value = events[-1]
                before = frozen_digest(chat.organism, chat.state)
                keys = torch.tensor([key], dtype=torch.long, device=device)
                zeros = torch.zeros_like(keys)
                with torch.no_grad():
                    now = int(query(chat.organism, chat.state, keys, zeros, zeros).argmax(-1)[0])
                    past = int(query(chat.organism, chat.state, keys, torch.ones_like(keys),
                                     torch.full_like(keys, past_time)).argmax(-1)[0])
                selected = select_event(chat.organism, chat.state, key, 'effective_previous', as_of=chat.state.cursor - 1)
                for mode, template, temporal_label in QUESTIONS:
                    question = template.format(name=name)
                    expected = current_value if mode == 'current' else past_value
                    enabled = chat.answer(question)
                    with patch.object(MemoryOutputLogitsProcessor, '__call__', lambda self, ids, scores: scores):
                        disabled = chat.answer(question)
                    row = {'person': person, 'scenario': scenario_index, 'mode': mode,
                           'expected_value': expected, 'expected_previous_event': past_time,
                           'native_current_top': now, 'native_past_top': past,
                           'expected_current_value': current_value, 'expected_past_value': past_value,
                           'independent_selection': selected, 'enabled': enabled, 'disabled': disabled}
                    for label in ('enabled', 'disabled'):
                        result = row[label]
                        selection = result.get('memory_event_selection')
                        event_correct = mode == 'current' or (isinstance(selection, dict)
                                                              and selection.get('event_time') == past_time)
                        row[label + '_route_correct'] = bool(result['accepted'] and result.get('key') == key
                                                             and result.get('temporal_intent') == temporal_label and event_correct)
                        row[label + '_answer_correct'] = bool(row[label + '_route_correct']
                                                              and result.get('answer') == WORDS[expected])
                    rows.append(row)
                    print(json.dumps({'person': person, 'scenario': scenario_index, 'mode': mode,
                                      'oracle_previous': past_time, 'selected_previous': selected.get('event_time'),
                                      'memory_current': now == current_value, 'memory_previous': past == past_value,
                                      'enabled_reason': enabled['reason'], 'disabled_reason': disabled['reason'],
                                      'enabled_correct': row['enabled_answer_correct'],
                                      'disabled_correct': row['disabled_answer_correct']}), flush=True)
                if frozen_digest(chat.organism, chat.state) != before:
                    raise ValueError('Inference modified state')
    if runtime.sha(runtime.CHECKPOINT) != checkpoint_sha:
        raise ValueError('Protected checkpoint changed')
    summary = {}
    for mode in ('current', 'past'):
        subset = [r for r in rows if r['mode'] == mode]
        summary[mode] = {'total': len(subset),
                         'native_expected_top': sum((r['native_current_top'] if mode == 'current' else r['native_past_top'])
                                                    == r['expected_value'] for r in subset),
                         'independent_previous_correct': sum(r['independent_selection'].get('event_time')
                                                             == r['expected_previous_event'] for r in subset),
                         **{label + '_route_correct': sum(r[label + '_route_correct'] for r in subset)
                            for label in ('enabled', 'disabled')},
                         **{label + '_answer_correct': sum(r[label + '_answer_correct'] for r in subset)
                            for label in ('enabled', 'disabled')}}
    report = {'status': 'controlled_interleaved_temporal_audit', 'summary': summary, 'rows': rows,
              'source_checkpoint_sha256': checkpoint_sha, 'seconds': time.perf_counter() - started,
              'limitations': ['Twelve authored questions: two known people and three hand-built short event sequences.',
                              'Manual numerical writes; event-time oracle used only for independent native score.',
                              'Event selector uses neural provenance which may omit events; no confidence guarantee.',
                              'Exact one-word answer matching; no synonyms or longer answers accepted.',
                              'No delayed effective times, long history, restart, natural language writes, or real user data.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
