"""Check teacher-forced extended cache parity with a fixed memory snapshot."""
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
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_boundary_runtime_pilot import ENROLLMENTS
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_long_cache_teacher_forced import TEXT, CHECKPOINTS, compare_logits
from scripts.deepseek_spora_memory_answer_distill import _memory_args
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import query
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-long-cache-memory-audit-20261112.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_sha = sha(runtime.CHECKPOINT)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokens = tokenizer(TEXT * 20, add_special_tokens=False, return_tensors='pt').input_ids
    if tokens.shape[1] < CHECKPOINTS[-1]:
        raise ValueError('Synthetic input too short')
    ids = tokens[:, :CHECKPOINTS[-1]].to('cuda')
    rows = []
    neutral_effects = []
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        base, decoder = chat.base.eval(), chat.decoder.eval()
        if any(p.is_meta for p in base.parameters()) or any(b.is_meta for b in base.buffers()):
            raise ValueError('Unmaterialized layer')
        original_digest = frozen_digest(chat.organism, chat.state)
        key = ENROLLMENTS[0][0]
        state = fresh_controlled_state(chat.organism)
        for value in (1, 6):
            state, decision = observe_sourced(chat.organism, state, key, value, 'direct', 1.0)
            if not decision['current_write']:
                raise ValueError('Controlled memory write refused')
        state_digest = frozen_digest(chat.organism, state)
        device = next(chat.organism.parameters()).device
        keys = torch.tensor([key], dtype=torch.long, device=device)
        native_values = {}
        for mode in ('current', 'past'):
            mode_ids = torch.zeros_like(keys) if mode == 'current' else torch.ones_like(keys)
            with torch.no_grad():
                native_values[mode] = int(query(chat.organism, state, keys, mode_ids, torch.zeros_like(keys)).argmax(-1)[0])
        if native_values != {'current': 6, 'past': 1}:
            raise ValueError('Synthetic memory values did not read as expected')

        for mode in ('current', 'past'):
            spec = _memory_args(chat.organism, state, {'key': key, 'intent': mode, 'event_time': 0})
            with patch.object(decoder.reader, 'forward', wraps=decoder.reader.forward) as reader, \
                    decoder._snapshot(state, spec['memory_addresses'], False, None, spec['memory_slots']), \
                    torch.inference_mode():
                first = CHECKPOINTS[0]
                before_calls = reader.call_count
                out = base(input_ids=ids[:, :first], attention_mask=torch.ones_like(ids[:, :first]), use_cache=True)
                cache = out.past_key_values
                if cache is None or reader.call_count - before_calls != 2:
                    raise ValueError('Cache or both reader sites missing during prefill')
                last = out.logits[0, -1].float()
                del out

                def compare(length, cached_logits):
                    before = reader.call_count
                    full = base(input_ids=ids[:, :length], attention_mask=torch.ones_like(ids[:, :length]),
                                use_cache=False).logits[0, -1].float()
                    if reader.call_count - before != 2:
                        raise ValueError('Both reader sites must run during full comparison')
                    row = {'mode': mode, 'length': length, **compare_logits(cached_logits, full)}
                    rows.append(row)
                    print(json.dumps(row), flush=True)

                compare(first, last)
                for index in range(first, CHECKPOINTS[-1]):
                    before = reader.call_count
                    out = base(input_ids=ids[:, index:index+1], attention_mask=torch.ones_like(ids[:, :index+1]),
                               past_key_values=cache, use_cache=True)
                    cache = out.past_key_values
                    if reader.call_count - before != 2:
                        raise ValueError('Both reader sites must run during cache update')
                    length = index + 1
                    if length in CHECKPOINTS:
                        compare(length, out.logits[0, -1].float())
                    if length % 128 == 0:
                        print(json.dumps({'mode': mode, 'processed_tokens': length}), flush=True)
                    del out
                call_count = reader.call_count
            # Compare memory-enabled vs memory-free logits to ensure the
            # intervention is active; this is not an answer correctness test.
            for length in (32, 512):
                with torch.inference_mode():
                    neutral = base(input_ids=ids[:, :length], attention_mask=torch.ones_like(ids[:, :length]),
                                   use_cache=False).logits[0, -1].float()
                    with decoder._snapshot(state, spec['memory_addresses'], False, None, spec['memory_slots']):
                        active = base(input_ids=ids[:, :length], attention_mask=torch.ones_like(ids[:, :length]),
                                      use_cache=False).logits[0, -1].float()
                effect = float((active - neutral).abs().max())
                neutral_effects.append({'mode': mode, 'length': length, 'max_absolute_difference': effect,
                                        'reader_calls_in_cache_comparison': call_count})
                if not effect > 0:
                    raise ValueError('Memory did not affect logits')
        if frozen_digest(chat.organism, state) != state_digest or frozen_digest(chat.organism, chat.state) != original_digest:
            raise ValueError('Memory state changed during inference')
    if sha(runtime.CHECKPOINT) != source_sha or sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected source changed')
    summary = {mode: {'checkpoints': len([r for r in rows if r['mode'] == mode]),
                      'top_matches': sum(r['top_match'] for r in rows if r['mode'] == mode),
                      'max_absolute_difference': max(r['max_absolute_difference'] for r in rows if r['mode'] == mode),
                      'max_kl': max(r['kl_full_to_cached'] for r in rows if r['mode'] == mode)}
               for mode in ('current', 'past')}
    report = {'status': 'memory_snapshot_teacher_forced_cache_development', 'summary': summary,
              'rows': rows, 'memory_enabled_vs_neutral': neutral_effects, 'native_values': native_values,
              'source_checkpoint_sha256': source_sha, 'base_sha256': WEIGHT_SHA,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['One repeated synthetic text, one known key and two explicitly supplied numeric facts.',
                              'Teacher-forced tokens; six checkpoints, not all positions or free generation.',
                              'Private snapshot test seam models a single immutable generate call; external cache reuse is not supported.',
                              'Memory was explicitly addressed; no language routing, natural writes or answer correctness.',
                              'Nonzero logit change is not proof that memory meaning was used correctly.',
                              'No mutation of actual individual or crash recovery test; research streamed loader.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
