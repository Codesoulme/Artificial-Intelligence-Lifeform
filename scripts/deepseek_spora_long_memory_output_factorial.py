"""Ablate two read sites and output bias in long prompted memory generation."""
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
from transformers import AutoTokenizer
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_boundary_runtime_pilot import ENROLLMENTS
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_long_memory_generate_parity import FACTS, FILLER, MAX_NEW, OUTPUT as PRIOR
from scripts.deepseek_spora_memory_answer_distill import _memory_args
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import query
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-long-memory-output-factorial-20261112.json'
MODES = (('both', True, True), ('reader_only', True, False),
         ('bias_only', False, True), ('neither', False, False))


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_sha = sha(runtime.CHECKPOINT)
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    if prior['source_checkpoint_sha256'] != source_sha:
        raise ValueError('Prior checkpoint mismatch')
    previous = {(r['key'], r['mode']): r for r in prior['rows']}
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        base, decoder = chat.base.eval(), chat.decoder.eval()
        if any(p.is_meta for p in base.parameters()) or any(b.is_meta for b in base.buffers()):
            raise ValueError('Unmaterialized layer')
        original_digest = frozen_digest(chat.organism, chat.state)
        state = fresh_controlled_state(chat.organism)
        for person, old, new in FACTS:
            for value in (old, new):
                state, decision = observe_sourced(chat.organism, state, ENROLLMENTS[person][0], value, 'direct', 1.0)
                if not decision['current_write']:
                    raise ValueError('Synthetic write rejected')
        state_digest = frozen_digest(chat.organism, state)
        device = next(chat.organism.parameters()).device
        for index, (person, old, new) in enumerate(FACTS):
            key, name, _ = ENROLLMENTS[person]
            for temporal, target in (('current', new), ('past', old)):
                question = f'{name}现在喜欢哪种饮料？' if temporal == 'current' else f'{name}以前喜欢哪种饮料？'
                inputs = closed_think_inputs(tokenizer, FILLER * 4 + question, 'cuda')
                length = inputs['input_ids'].shape[1]
                if not (128 < length and length + MAX_NEW <= 768):
                    raise ValueError('Unexpected position range')
                spec = _memory_args(chat.organism, state, {'key': key, 'intent': temporal, 'event_time': index * 2})
                keys = torch.tensor([key], dtype=torch.long, device=device)
                modes = torch.zeros_like(keys) if temporal == 'current' else torch.ones_like(keys)
                times = torch.zeros_like(keys) if temporal == 'current' else torch.full_like(keys, index * 2)
                with torch.no_grad():
                    logits = query(chat.organism, state, keys, modes, times)
                if int(logits.argmax(-1)[0]) != target:
                    raise ValueError('Native evidence mismatch')
                results = {}
                for mode, reader_on, bias_on in MODES:
                    change = ({'wraps': decoder.reader.forward} if reader_on else
                              {'side_effect': lambda hidden, *args, **kwargs: hidden})
                    with patch.object(decoder.reader, 'forward', **change) as reader, torch.inference_mode():
                        ids = decoder.generate(**inputs, **{**spec, 'use_cache': True},
                            memory_output_logits=logits if bias_on else None, max_new_tokens=MAX_NEW, do_sample=False,
                            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                    if reader.call_count < 2:
                        raise ValueError('Reader intervention was not invoked')
                    tail = ids[0, length:].tolist()
                    answer = tokenizer.decode(tail, skip_special_tokens=True).strip()
                    results[mode] = {'answer': answer, 'token_ids': tail, 'reader_calls': reader.call_count,
                                     'exact': answer == WORDS[target]}
                expected_prior = previous[(key, temporal)]['outputs']['cached']['token_ids']
                rows.append({'key': key, 'temporal': temporal, 'expected': WORDS[target], 'modes': results,
                             'both_reproduces_prior': results['both']['token_ids'] == expected_prior})
                print(json.dumps({'key': key, 'temporal': temporal,
                                  'same_as_prior': rows[-1]['both_reproduces_prior'],
                                  **{mode: results[mode]['exact'] for mode, _, _ in MODES}}), flush=True)
        if frozen_digest(chat.organism, state) != state_digest or frozen_digest(chat.organism, chat.state) != original_digest:
            raise ValueError('Individual state changed')
    if sha(runtime.CHECKPOINT) != source_sha or sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected source changed')
    summary = {'cases': len(rows), 'both_reproduces_prior': sum(r['both_reproduces_prior'] for r in rows),
               **{mode + '_exact': sum(r['modes'][mode]['exact'] for r in rows) for mode, _, _ in MODES}}
    report = {'status': 'long_memory_reader_bias_factorial_development', 'summary': summary, 'rows': rows,
              'source_checkpoint_sha256': source_sha, 'base_sha256': WEIGHT_SHA, 'prior_sha256': sha(PRIOR),
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Four previously inspected numeric facts for two known keys; same long filler.',
                              'Two reader sites are ablated together; no causal claim about each individual site.',
                              'Native evidence, addresses and event times are supplied; not natural-language routing.',
                              'Exact word scoring can undercount compatible answers and miss unsupported detail.',
                              'No default policy, checkpoint, or real individual changed.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
