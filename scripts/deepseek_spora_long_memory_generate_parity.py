"""Compare cached and uncached generate with an immutable internal memory snapshot."""
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
from scripts.deepseek_spora_memory_answer_distill import _memory_args
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import query
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-long-memory-generate-parity-20261112.json'
FILLER = ('窗边有一本书，桌旁放着一盏灯。清晨路边的树叶轻轻摇动。'
          '图书馆有人整理书架，公园里有人缓慢散步。傍晚路灯照亮安静的街道。')
FACTS = ((0, 1, 6), (1, 3, 4))
MAX_NEW = 24


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_sha = sha(runtime.CHECKPOINT)
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
            key = ENROLLMENTS[person][0]
            for value in (old, new):
                state, decision = observe_sourced(chat.organism, state, key, value, 'direct', 1.0)
                if not decision['current_write']:
                    raise ValueError('Synthetic write rejected')
        state_digest = frozen_digest(chat.organism, state)
        device = next(chat.organism.parameters()).device
        for index, (person, old, new) in enumerate(FACTS):
            key, name, _ = ENROLLMENTS[person]
            for mode, target in (('current', new), ('past', old)):
                question = f'{name}现在喜欢哪种饮料？' if mode == 'current' else f'{name}以前喜欢哪种饮料？'
                text = FILLER * 4 + question
                inputs = closed_think_inputs(tokenizer, text, 'cuda')
                length = inputs['input_ids'].shape[1]
                if not (128 < length and length + MAX_NEW <= 768):
                    raise ValueError('Prompt must exercise positions beyond 128 within 768')
                spec = _memory_args(chat.organism, state, {'key': key, 'intent': mode, 'event_time': index * 2})
                keys = torch.tensor([key], dtype=torch.long, device=device)
                modes = torch.zeros_like(keys) if mode == 'current' else torch.ones_like(keys)
                times = torch.zeros_like(keys) if mode == 'current' else torch.full_like(keys, index * 2)
                with torch.no_grad():
                    logits = query(chat.organism, state, keys, modes, times)
                if int(logits.argmax(-1)[0]) != target:
                    raise ValueError('Native memory value does not match controlled target')
                outputs = {}
                for cache in (False, True):
                    args = {**spec, 'use_cache': cache}
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    with patch.object(decoder.reader, 'forward', wraps=decoder.reader.forward) as reader, torch.inference_mode():
                        generated = decoder.generate(**inputs, **args, memory_output_logits=logits,
                                                     max_new_tokens=MAX_NEW, do_sample=False,
                                                     pad_token_id=tokenizer.pad_token_id,
                                                     eos_token_id=tokenizer.eos_token_id)
                    torch.cuda.synchronize()
                    tail = generated[0, length:].tolist()
                    if reader.call_count < 2:
                        raise ValueError('Both memory reader sites did not execute')
                    outputs['cached' if cache else 'uncached'] = {
                        'token_ids': tail, 'answer': tokenizer.decode(tail, skip_special_tokens=True).strip(),
                        'reader_calls': reader.call_count, 'seconds': time.perf_counter() - begin}
                a, b = outputs['uncached']['token_ids'], outputs['cached']['token_ids']
                common = 0
                for x, y in zip(a, b):
                    if x != y:
                        break
                    common += 1
                row = {'key': key, 'mode': mode, 'expected_value': WORDS[target], 'input_tokens': length,
                       'outputs': outputs, 'tokens_match': a == b, 'matching_prefix_tokens': common,
                       'uncached_exact': outputs['uncached']['answer'] == WORDS[target],
                       'cached_exact': outputs['cached']['answer'] == WORDS[target]}
                rows.append(row)
                print(json.dumps({'key': key, 'mode': mode, 'input_tokens': length,
                                  'tokens_match': row['tokens_match'], 'matching_prefix_tokens': common,
                                  'uncached_exact': row['uncached_exact'], 'cached_exact': row['cached_exact']}), flush=True)
        if frozen_digest(chat.organism, state) != state_digest or frozen_digest(chat.organism, chat.state) != original_digest:
            raise ValueError('Individual state changed during generation')
    if sha(runtime.CHECKPOINT) != source_sha or sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected source changed')
    summary = {'cases': len(rows), 'tokens_match': sum(r['tokens_match'] for r in rows),
               'uncached_exact': sum(r['uncached_exact'] for r in rows), 'cached_exact': sum(r['cached_exact'] for r in rows)}
    report = {'status': 'long_memory_generate_cache_development', 'summary': summary, 'rows': rows,
              'source_checkpoint_sha256': source_sha, 'base_sha256': WEIGHT_SHA,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Two known keys, four controlled numeric queries, repeated synthetic filler.',
                              'Addresses, event slots and output evidence supplied by the test, not natural-language routing.',
                              'Exact 24-token capped answer comparison is not a general quality measure.',
                              'Only immutable per-call memory snapshots; no reuse across writes or sessions.',
                              'No real individual changed; research streamed model loader.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
