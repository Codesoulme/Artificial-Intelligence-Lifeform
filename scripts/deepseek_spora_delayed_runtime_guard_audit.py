"""Check delayed-fact abstention in the actual temporary chat route."""
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
from scripts.deepseek_spora_delayed_fact_audit import OUTPUT as PRIOR, SCENARIOS
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-delayed-runtime-guard-audit-20261112.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checkpoint_sha = runtime.sha(runtime.CHECKPOINT)
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    if (prior['source_checkpoint_sha256'] != checkpoint_sha or prior['summary']['previous_abstained'] != 6):
        raise ValueError('Guard baseline evidence missing')
    torch.set_num_threads(2)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        runtime.AutoModelForCausalLM.from_pretrained = lambda *args, **kwargs: load_full_streamed()
        chat = runtime.load_chat()
        rows = []
        for person in (0, 3):
            key, name, _ = ENROLLMENTS[person]
            distractor = ENROLLMENTS[3 if person == 0 else 0][0]
            for scenario_index in (0, 1):
                chat.state = fresh_controlled_state(chat.organism)
                target_events = []
                for index, (subject, value, valid_at) in enumerate(SCENARIOS[scenario_index]):
                    chat.state, decision = observe_sourced(chat.organism, chat.state,
                                                          key if subject == 'T' else distractor,
                                                          value, 'direct', 1.0, valid_at=valid_at)
                    if subject == 'T':
                        target_events.append((index, value, valid_at))
                expected_current = max(target_events, key=lambda item: (item[2], item[0]))[1]
                before = frozen_digest(chat.organism, chat.state)
                current = chat.answer(f'{name}现在喜欢哪种饮料？')
                with patch.object(chat.decoder, 'generate', wraps=chat.decoder.generate) as generate:
                    past = chat.answer(f'{name}以前喜欢哪种饮料？')
                selection = past.get('memory_event_selection') or {}
                row = {'person': person, 'scenario': scenario_index, 'current': current, 'past': past,
                       'current_key_correct': bool(current['accepted'] and current.get('key') == key),
                       'current_answer_correct': bool(current['accepted'] and current.get('key') == key
                                                      and current.get('answer') == WORDS[expected_current]),
                       'past_guarded': bool(not past['accepted'] and past.get('reason') == 'memory_event_abstain'
                                            and selection.get('reason') == 'unresolved_factual_observation'),
                       'past_generation_calls': generate.call_count}
                if row['past_guarded'] and generate.call_count:
                    raise ValueError('Generated an answer despite temporal abstention')
                rows.append(row)
                print(json.dumps({'person': person, 'scenario': scenario_index,
                                  'current_reason': current['reason'], 'past_reason': past['reason'],
                                  'past_guarded': row['past_guarded'], 'past_generation_calls': generate.call_count}), flush=True)
                if frozen_digest(chat.organism, chat.state) != before:
                    raise ValueError('Inference modified temporary individual')
    if runtime.sha(runtime.CHECKPOINT) != checkpoint_sha:
        raise ValueError('Protected checkpoint changed')
    summary = {'cases': len(rows), 'current_key_correct': sum(r['current_key_correct'] for r in rows),
               'current_answer_correct': sum(r['current_answer_correct'] for r in rows),
               'past_guarded': sum(r['past_guarded'] for r in rows),
               'past_generations': sum(r['past_generation_calls'] for r in rows)}
    report = {'status': 'controlled_delayed_runtime_guard', 'summary': summary, 'rows': rows,
              'source_checkpoint_sha256': checkpoint_sha, 'seconds': time.perf_counter() - started,
              'limitations': ['Four synthetic cases, two names and two short event sequences.',
                              'Facts are manually written using numerical keys and effective times.',
                              'Known identities only; no opt-in identity guard or natural-language fact learning.',
                              'Temporal abstention is a conservative mitigation, not recovery of the late fact.',
                              'No long histories, crash recovery, or real user individual used.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
