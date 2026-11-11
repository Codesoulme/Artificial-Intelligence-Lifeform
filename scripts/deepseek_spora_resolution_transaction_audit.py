"""Temporary full-checkpoint transaction with a fresh-process individual read."""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_boundary_runtime_pilot import ENROLLMENTS
from scripts.deepseek_spora_delayed_fact_audit import SCENARIOS, load_individual_from_verified_checkpoint
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_resolution_runtime_pilot import OUTPUT as PRIOR
from scripts.spora_event_scan_selector import select_event
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-resolution-transaction-audit-20261112.json'


def verify_child(path):
    torch.set_num_threads(2)
    model, state = load_individual_from_verified_checkpoint(path)
    key = ENROLLMENTS[0][0]
    selection = select_event(model, state, key, 'effective_previous', as_of=state.cursor - 1)
    record = model.historical_resolutions.lookup(key, 4)
    print(json.dumps({'selection': selection, 'record': record, 'individual_digest': frozen_digest(model, state)},
                     ensure_ascii=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-child')
    args = parser.parse_args()
    if args.verify_child:
        verify_child(Path(args.verify_child))
        return
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checkpoint_sha = runtime.sha(runtime.CHECKPOINT)
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    if prior['source_checkpoint_sha256'] != checkpoint_sha or not prior['summary']['past_answer_correct']:
        raise ValueError('Prior runtime evidence mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'working.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        if any(p.is_meta for p in chat.base.parameters()) or any(b.is_meta for b in chat.base.buffers()):
            raise ValueError('Unmaterialized replacement layer')
        key, name, _ = ENROLLMENTS[0]
        distractor = ENROLLMENTS[3][0]
        chat.state = fresh_controlled_state(chat.organism)
        for subject, value, valid_at in SCENARIOS[1]:
            chat.state, _ = observe_sourced(chat.organism, chat.state, key if subject == 'T' else distractor,
                                           value, 'direct', 1.0, valid_at=valid_at)
        before = chat.answer(f'{name}以前喜欢哪种饮料？')
        before_model, before_state = chat.organism, chat.state
        before_digest = frozen_digest(before_model, before_state)
        result = chat.resolve_historical(key, 4, value=3, valid_at=1, decision='confirm')
        after = chat.answer(f'{name}以前喜欢哪种饮料？')
        current = chat.answer(f'{name}现在喜欢哪种饮料？')
        committed_digest = frozen_digest(chat.organism, chat.state)
        original_unchanged = frozen_digest(before_model, before_state) == before_digest
        checkpoint_file_sha = runtime.sha(runtime.WORKING_CHECKPOINT)
        print(json.dumps({'phase': 'committed', 'prior_abstained': not before['accepted'],
                          'past_event': (after.get('memory_event_selection') or {}).get('event_time'),
                          'past_correct': after.get('answer') == WORDS[3],
                          'current_correct': current.get('answer') == WORDS[6],
                          'original_unchanged': original_unchanged}), flush=True)
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--verify-child',
                                str(runtime.WORKING_CHECKPOINT)], capture_output=True, text=True,
                               encoding='utf-8', timeout=240)
        if child.returncode != 0:
            raise RuntimeError(f'Fresh-process read failed: {child.stderr[-2000:]}')
        restored = json.loads(child.stdout.strip())
        child_same = (restored['individual_digest'] == committed_digest
                      and restored['record']['decision'] == 'confirm'
                      and restored['selection'].get('event_time') == 4)
    if runtime.sha(runtime.CHECKPOINT) != checkpoint_sha:
        raise ValueError('Protected source checkpoint changed')
    summary = {'prior_abstained': not before['accepted'], 'past_event_correct':
               (after.get('memory_event_selection') or {}).get('event_time') == 4,
               'past_answer_correct': after.get('answer') == WORDS[3],
               'current_answer_correct': current.get('answer') == WORDS[6],
               'candidate_published': chat.organism is not before_model and chat.state is not before_state,
               'original_unchanged': original_unchanged, 'fresh_process_individual_matches': child_same}
    report = {'status': 'temporary_resolution_transaction_audit', 'summary': summary, 'result': result,
              'source_checkpoint_sha256': checkpoint_sha, 'temporary_checkpoint_sha256': checkpoint_file_sha,
              'before': before, 'after': after, 'current': current, 'child': restored,
              'seconds': time.perf_counter() - started,
              'limitations': ['One manually confirmed synthetic event; no natural-language learning or real user state.',
                              'Fresh process restores and checks individual tensors only; it does not instantiate the decoder.',
                              'No crash at transaction boundary or concurrent process was exercised in this experiment.',
                              'Explicit confirmation is API-only and caller is responsible for authority and identity.',
                              'Research loader skips replaced base layers; general loader equivalence is unproven.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
