"""Probe decoder page selection on one constructed two-page SPORA history."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_memory_answer_distill import SOURCE_CHECKPOINT, _load_candidate, _memory_args
from scripts.deepseek_spora_recalibrate_name_router import SOURCE
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import grow, query
from spora.integrated_checkpoint import load_integrated
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT/'runs/deepseek-spora-cross-page-answer-audit-20261111.json'


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    integrated, organism, original = load_integrated(SOURCE_CHECKPOINT, base, MODEL, individual_device='cuda')
    original_digest = frozen_digest(organism, original)
    state = fresh_controlled_state(organism)
    key = KEYS[0]
    state, first = observe_sourced(organism, state, key, 1, 'direct', 1.0)
    if not first['current_write']:
        raise ValueError('First controlled event was not written')
    for i in range(1, organism.config.time_count):
        other_key = (key + 1 + (i % (organism.config.key_count - 1))) % organism.config.key_count
        state, _ = observe_sourced(organism, state, other_key, i % organism.config.unknown, 'direct', 1.0)
    state = grow(organism, state)
    later_time = state.cursor
    state, later = observe_sourced(organism, state, key, 6, 'direct', 1.0)
    if not later['current_write']:
        raise ValueError('Second controlled event was not written')
    for layer in (6, 12):
        base.model.layers[layer] = _load_candidate(SOURCE/f'layer-{layer}.pt', layer, base).eval()
    question = f'{NAMES[0]}后来喜欢的饮品是什么？'
    prefix = closed_think_inputs(tokenizer, question, 'cuda')
    rows = []
    with torch.inference_mode():
        for event_time, expected in ((0, 1), (later_time, 6)):
            keys = torch.tensor([key], device='cuda')
            times = torch.tensor([event_time], device='cuda')
            native = int(query(organism, state, keys, torch.ones_like(keys), times).argmax(-1))
            args = _memory_args(organism, state, {'key': key, 'intent': 'later', 'event_time': event_time})
            ids = integrated.generate(**prefix, **args, max_new_tokens=8, do_sample=False,
                                      pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            answer = tokenizer.decode(ids[0, prefix['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            rows.append({'event_time': event_time, 'memory_slot': int(args['memory_slots'][0]),
                         'expected_value': expected, 'expected_word': WORDS[expected],
                         'native_value': native, 'answer': answer,
                         'answer_exact': answer in (WORDS[expected], WORDS[expected]+'.', WORDS[expected]+'。')})
    if frozen_digest(organism, original) != original_digest or sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected input changed')
    report = {'status': 'explicit_event_cross_page_development_only', 'base_sha256': WEIGHT_SHA,
              'source_sha256': sha(SOURCE_CHECKPOINT), 'candidate_sha256': {
                  str(layer): sha(SOURCE/f'layer-{layer}.pt') for layer in (6, 12)},
              'question': question, 'cursor': state.cursor, 'page_count': len(state.pages),
              'first_write': first, 'later_write': later, 'rows': rows,
              'elapsed_seconds': time.perf_counter()-start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Event times and entity address are supplied by the experiment, not inferred.',
                              'One synthetic two-page trajectory and one question; no broad language claim.',
                              'Tests decoder output alongside native finite-symbol value; either may fail.',
                              'Does not evaluate arbitrary timeline resolution, autonomous writing or restart.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'rows': rows, 'elapsed_seconds': report['elapsed_seconds'],
                      'peak_allocated_bytes': report['peak_allocated_bytes']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
