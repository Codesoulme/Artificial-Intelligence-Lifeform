"""Locate whether a controlled incorrect answer starts in memory or decoding."""
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
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_boundary_runtime_pilot import CASES, ENROLLMENTS, OUTPUT as PRIOR
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import query
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-output-failure-probe-20261112.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    checkpoint_sha = runtime.sha(runtime.CHECKPOINT)
    if prior['source_checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Reference checkpoint mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        runtime.AutoModelForCausalLM.from_pretrained = lambda *args, **kwargs: load_full_streamed()
        chat = runtime.load_chat()
        chat.state = fresh_controlled_state(chat.organism)
        for key, _, value in ENROLLMENTS:
            chat.state, decision = observe_sourced(chat.organism, chat.state, key, value, 'direct', 1.0)
            if not decision['current_write']:
                raise ValueError('Controlled write rejected')
        before = frozen_digest(chat.organism, chat.state)
        device = next(chat.organism.parameters()).device
        rows = []
        for i, (key, _, value) in enumerate(ENROLLMENTS):
            key_tensor = torch.tensor([key], dtype=torch.long, device=device)
            zero = torch.zeros_like(key_tensor)
            with torch.no_grad():
                logits = query(chat.organism, chat.state, key_tensor, zero, zero)
            if not bool(torch.isfinite(logits).all()):
                raise ValueError('Nonfinite memory query')
            predicted = int(logits.argmax(-1)[0])
            rows.append({'enrollment_index': i, 'key': key, 'expected_value': value,
                         'memory_top_value': predicted, 'memory_matches': predicted == value,
                         'prior_answer_matches': prior['rows'][i]['default']['correct_answer']})
        result = chat.answer(CASES[3][1])
        if frozen_digest(chat.organism, chat.state) != before:
            raise ValueError('Temporary individual changed during reads')
    if runtime.sha(runtime.CHECKPOINT) != checkpoint_sha:
        raise ValueError('Source checkpoint changed')
    report = {'status': 'controlled_output_failure_diagnostic', 'rows': rows,
              'problem_case': {'answer': result['answer'], 'accepted': result['accepted'],
                               'key': result.get('key'), 'expected_word': WORDS[ENROLLMENTS[3][2]],
                               'same_as_prior': result.get('answer') == prior['rows'][3]['default']['actual']['answer']},
              'source_checkpoint_sha256': checkpoint_sha, 'seconds': time.perf_counter() - started,
              'limitations': ['Four numerical-key fact writes; no language-based observation or real user data.',
                              'Memory argmax alone does not explain every generation failure.',
                              'Single question reproduced; not a population or causal ablation.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
