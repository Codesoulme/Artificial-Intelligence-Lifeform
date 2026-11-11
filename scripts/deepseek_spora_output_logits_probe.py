"""Inspect first-token memory bias on controlled answer-generation failures."""
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
from scripts.deepseek_spora_boundary_runtime_pilot import CASES, ENROLLMENTS, OUTPUT as PRIOR
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest
from spora.memory_output import MemoryOutputLogitsProcessor

OUTPUT = ROOT / 'runs/deepseek-spora-output-logits-probe-20261112.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    source_sha = runtime.sha(runtime.CHECKPOINT)
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
        original = MemoryOutputLogitsProcessor.__call__
        for i, (_, question, _) in enumerate(CASES[:4]):
            expected_word = WORDS[ENROLLMENTS[i][2]]
            target_ids = chat.tokenizer(expected_word, add_special_tokens=False)['input_ids']
            if not target_ids:
                raise ValueError('Empty target encoding')
            target = target_ids[0]
            capture = []

            def wrapped(processor, input_ids, scores):
                first = not processor._applied
                raw = scores.detach().float().clone() if first else None
                adjusted = original(processor, input_ids, scores)
                if first:
                    post = adjusted.detach().float()
                    winner_before = int(raw[0].argmax())
                    winner_after = int(post[0].argmax())
                    capture.append({'target_token_id': target, 'target_token_count': len(target_ids),
                                    'target_raw': float(raw[0, target]),
                                    'target_bias': float(post[0, target] - raw[0, target]),
                                    'target_adjusted': float(post[0, target]),
                                    'raw_winner_id': winner_before, 'raw_winner_score': float(raw[0, winner_before]),
                                    'adjusted_winner_id': winner_after,
                                    'adjusted_winner_score': float(post[0, winner_after]),
                                    'target_rank_after': int((post[0] > post[0, target]).sum()) + 1})
                return adjusted

            with patch.object(MemoryOutputLogitsProcessor, '__call__', wrapped):
                result = chat.answer(question)
            if len(capture) != 1:
                raise ValueError('Expected one first-token capture')
            rows.append({'case_index': i, 'expected_word': expected_word, 'actual_answer': result['answer'],
                         'prior_answer_same': result['answer'] == prior['rows'][i]['default']['actual']['answer'],
                         'first_token': capture[0]})
            print(json.dumps({'case_index': i, 'target_rank_after': capture[0]['target_rank_after'],
                              'winner_is_target': capture[0]['adjusted_winner_id'] == target}), flush=True)
        if frozen_digest(chat.organism, chat.state) != before:
            raise ValueError('Temporary individual changed')
    if runtime.sha(runtime.CHECKPOINT) != source_sha:
        raise ValueError('Protected source changed')
    report = {'status': 'first_token_diagnostic', 'rows': rows, 'source_checkpoint_sha256': source_sha,
              'seconds': time.perf_counter() - started,
              'limitations': ['Four synthetic examples and one decoding step, not a causal training or ablation study.',
                              'Expected first token is derived from standalone value tokenization; generated phrasing may differ.',
                              'Processor receives logits after any earlier generation processors; no real individual changed.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'rows': len(rows), 'source_sha': source_sha}), flush=True)


if __name__ == '__main__':
    main()
