"""Temporarily remove only the output bias and compare four controlled answers."""
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
from scripts.deepseek_spora_boundary_runtime_pilot import CASES, ENROLLMENTS
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_output_logits_probe import OUTPUT as PRIOR
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest
from spora.memory_output import MemoryOutputLogitsProcessor

OUTPUT = ROOT / 'runs/deepseek-spora-output-bias-ablation-20261112.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_sha = runtime.sha(runtime.CHECKPOINT)
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    if prior['source_checkpoint_sha256'] != source_sha:
        raise ValueError('Prior checkpoint mismatch')
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
        rows = []
        for i, (_, question, _) in enumerate(CASES[:4]):
            enabled = chat.answer(question)
            # Only suppress this processor's vocabulary bias; memory reader
            # and all other runtime behavior remain active.
            with patch.object(MemoryOutputLogitsProcessor, '__call__', lambda self, ids, scores: scores):
                disabled = chat.answer(question)
            expected = WORDS[ENROLLMENTS[i][2]]
            rows.append({'case_index': i, 'expected_word': expected,
                         'enabled_answer': enabled.get('answer'), 'disabled_answer': disabled.get('answer'),
                         'enabled_correct': enabled.get('answer') == expected,
                         'disabled_correct': disabled.get('answer') == expected,
                         'same_enabled_as_prior': enabled.get('answer') == prior['rows'][i]['actual_answer'],
                         'enabled_key': enabled.get('key'), 'disabled_key': disabled.get('key')})
            print(json.dumps({'case_index': i, 'enabled_correct': rows[-1]['enabled_correct'],
                              'disabled_correct': rows[-1]['disabled_correct']}), flush=True)
        if frozen_digest(chat.organism, chat.state) != before:
            raise ValueError('Temporary individual changed')
    if runtime.sha(runtime.CHECKPOINT) != source_sha:
        raise ValueError('Protected source changed')
    result = {'status': 'controlled_output_bias_ablation', 'rows': rows,
              'enabled_correct': sum(r['enabled_correct'] for r in rows),
              'disabled_correct': sum(r['disabled_correct'] for r in rows),
              'source_checkpoint_sha256': source_sha, 'seconds': time.perf_counter() - started,
              'limitations': ['Four authored questions and manually initialized numerical facts only.',
                              'Disabling output bias leaves the other memory-reader hooks active.',
                              'Correct answers without this bias do not prove independence from memory or general improvement.',
                              'No change to trained weights, production settings, or the real individual.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'enabled_correct': result['enabled_correct'], 'disabled_correct': result['disabled_correct']}), flush=True)


if __name__ == '__main__':
    main()
