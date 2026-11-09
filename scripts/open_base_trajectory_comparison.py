"""Compare native SPORA, unmodified Qwen, and Qwen with SPORA memory.

Development experiment: several new event trajectories share one original state.
No target answer is supplied in the model prompts or external addresses.
"""
import copy
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.open_base_addressed_alignment import QUERY_PATH, QUERY_SHA, neural_addresses
from scripts.open_base_baseline import MODEL, INDIVIDUAL, INDIVIDUAL_SHA, WEIGHT_SHA, REVISION, sha
from scripts.open_base_real_memory_alignment import native_read
from scripts.open_base_synthetic_alignment import encode_prompt
from spora.growing import load
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest
from spora.open_base_memory import OpenBaseMemory, memory_hooks

ADAPTER_DIR = ROOT / 'runs/open-base-addressed-full-answer-20261109'
OUTPUT = ROOT / 'runs/open-base-trajectory-comparison-v2-20261109.json'
PROMPTS = ('对象{key}的值是多少？', '请问对象{key}目前的值是多少？')


def build_trajectories(organism, original):
    rng = random.Random(4601)
    trajectories = []
    for trajectory in range(4):
        state = copy.deepcopy(original)
        keys = list(range(20 + trajectory * 4, 24 + trajectory * 4))
        events = []
        for key in keys:
            events.append((key, rng.randrange(16), 'initial'))
        for key in keys[:2]:
            previous = next(value for k, value, _ in events if k == key)
            events.append((key, (previous + rng.randrange(1, 16)) % 16, 'correction'))
        for key in range(40, 48):
            events.append((key, rng.randrange(16), 'interference'))
        targets, log = {}, []
        for key, value, role in events:
            state, decision = observe_sourced(organism, state, key, value, 'direct', 1.0)
            targets[key] = value
            log.append({'key': key, 'value': value, 'role': role, 'action': decision['action'],
                        'current_write': decision['current_write']})
        trajectories.append({'id': trajectory, 'state': state, 'keys': keys,
                             'targets': targets, 'events': log})
    return trajectories


def numeric_correct(response, value):
    return response.strip() in (str(value), str(value) + '.', str(value) + '。')


@torch.inference_mode()
def generate(model, tokenizer, question, *, layers=None, adapter=None, state=None, addresses=None):
    inputs = encode_prompt(tokenizer, question, 'cuda')
    start = time.perf_counter()
    if adapter is None:
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    else:
        with memory_hooks(layers, adapter, state, current_only=True, addresses=addresses):
            output = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    response = tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return response, time.perf_counter() - start


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    training = json.loads((ADAPTER_DIR / 'result.json').read_text(encoding='utf-8'))
    adapter_path = ADAPTER_DIR / 'adapter.pt'
    adapter_sha = training['adapter_sha256']
    def verify_files():
        return (sha(MODEL / 'model.safetensors') == WEIGHT_SHA and sha(INDIVIDUAL) == INDIVIDUAL_SHA
                and sha(QUERY_PATH) == QUERY_SHA and sha(adapter_path) == adapter_sha)
    if not verify_files():
        raise ValueError('Protected source mismatch')
    checkpoint = torch.load(adapter_path, map_location='cpu', weights_only=True)
    if (checkpoint.get('format') != 'spora-open-base-p26-full-answer-v1'
            or checkpoint.get('individual_sha256') != INDIVIDUAL_SHA
            or checkpoint.get('weight_sha256') != WEIGHT_SHA or checkpoint.get('query_sha256') != QUERY_SHA
            or checkpoint.get('layers') != [5, 17] or checkpoint.get('current_only') is not True):
        raise ValueError('Adapter provenance mismatch')
    torch.manual_seed(4601)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    reader, reader_state, _ = load(QUERY_PATH, 'cpu')
    if any(not torch.equal(a.current_basis, b.current_basis) for a, b in zip(organism.memories, reader.memories)):
        raise ValueError('Incompatible address basis')
    before = frozen_digest(organism, original)
    reader_before = frozen_digest(reader, reader_state)
    trajectories = build_trajectories(organism, original)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='eager').to('cuda').eval()
    model.requires_grad_(False)
    adapter = OpenBaseMemory(model.config.hidden_size, organism.config.memory_width).to('cuda').eval()
    adapter.load_state_dict(checkpoint['state'], strict=True)
    layers = [model.model.layers[5], model.model.layers[17]]
    rows = []
    for item in trajectories:
        state = item['state']
        for key in item['keys']:
            expected = item['targets'][key]
            native = native_read(organism, state, [key])[0]
            for template in PROMPTS:
                question = template.format(key=key)
                baseline, baseline_seconds = generate(model, tokenizer, question)
                row = {'trajectory': item['id'], 'key': key, 'question': question, 'expected': expected,
                       'native': native, 'native_correct': native == expected, 'base': baseline,
                       'base_correct': numeric_correct(baseline, expected), 'base_seconds': baseline_seconds}
                try:
                    addresses, parsed = neural_addresses(reader, organism, question)
                    if parsed['entity'] != key:
                        raise ValueError('Wrong parsed entity')
                except ValueError as exc:
                    row.update(combined=None, combined_correct=False, address_accepted=False, address_error=str(exc))
                else:
                    response, seconds = generate(model, tokenizer, question, layers=layers, adapter=adapter,
                                                 state=state, addresses=addresses)
                    row.update(combined=response, combined_seconds=seconds,
                               combined_correct=numeric_correct(response, expected), address_accepted=True)
                rows.append(row)
        print(f"trajectory {item['id'] + 1}/4 complete", flush=True)
    unknowns = []
    candidates = list(range(48, 64))
    original_values = native_read(organism, original, candidates)
    for key in [k for k, value in zip(candidates, original_values) if value == organism.config.unknown][:4]:
        question = PROMPTS[0].format(key=key)
        native = native_read(organism, trajectories[0]['state'], [key])[0]
        row = {'key': key, 'question': question, 'native': native}
        try:
            addresses, parsed = neural_addresses(reader, organism, question)
            if parsed['entity'] != key:
                raise ValueError('Wrong parsed entity')
        except ValueError as exc:
            row['address_error'] = str(exc)
        else:
            row['combined'], row['seconds'] = generate(model, tokenizer, question, layers=layers, adapter=adapter,
                                                       state=trajectories[0]['state'], addresses=addresses)
        unknowns.append(row)
    counterfactuals = []
    for item in trajectories:
        key = item['keys'][2]
        old_target = item['targets'][key]
        new_target = (old_target + 5) % 16
        updated, decision = observe_sourced(organism, copy.deepcopy(item['state']), key, new_target, 'direct', 1.0)
        question = PROMPTS[0].format(key=key)
        row = {'trajectory': item['id'], 'key': key, 'question': question,
               'update': {'value': new_target, 'action': decision['action'], 'current_write': decision['current_write']},
               'conditions': []}
        try:
            addresses, parsed = neural_addresses(reader, organism, question)
            if parsed['entity'] != key:
                raise ValueError('Wrong parsed entity')
        except ValueError as exc:
            row['address_error'] = str(exc)
        else:
            for condition, state, expected in (('before', item['state'], old_target), ('after', updated, new_target)):
                native = native_read(organism, state, [key])[0]
                response, seconds = generate(model, tokenizer, question, layers=layers, adapter=adapter,
                                             state=state, addresses=addresses)
                row['conditions'].append({'condition': condition, 'expected': expected, 'native': native,
                                          'combined': response, 'correct': numeric_correct(response, expected),
                                          'seconds': seconds})
        counterfactuals.append(row)
    print('counterfactuals complete', flush=True)
    general_probes = []
    address, parsed = neural_addresses(reader, organism, PROMPTS[0].format(key=20))
    if parsed['entity'] != 20:
        raise ValueError('Wrong probe address')
    for question in ('请用一句话解释什么是雨。', '请列出三种水果。', '一加一等于几？'):
        baseline, baseline_seconds = generate(model, tokenizer, question)
        forced, forced_seconds = generate(model, tokenizer, question, layers=layers, adapter=adapter,
                                          state=trajectories[0]['state'], addresses=address)
        general_probes.append({'question': question, 'base': baseline, 'base_seconds': baseline_seconds,
                               'forced_memory': forced, 'forced_memory_seconds': forced_seconds})
    print('general probes complete', flush=True)
    if (not verify_files() or frozen_digest(organism, original) != before
            or frozen_digest(reader, reader_state) != reader_before or any(p.grad is not None for p in model.parameters())):
        raise ValueError('Protected input changed')
    summary = {'questions': len(rows), 'native_correct': sum(r['native_correct'] for r in rows),
               'base_correct': sum(r['base_correct'] for r in rows),
               'combined_correct': sum(r['combined_correct'] for r in rows),
               'address_accepted': sum(r['address_accepted'] for r in rows)}
    result = {'revision': REVISION, 'individual_sha256': INDIVIDUAL_SHA, 'weight_sha256': WEIGHT_SHA,
              'query_sha256': QUERY_SHA, 'adapter_sha256': adapter_sha, 'summary': summary, 'rows': rows,
              'trajectories': [{k: v for k, v in item.items() if k != 'state'} for item in trajectories],
              'native_unknown_cases': unknowns, 'counterfactuals': counterfactuals,
              'general_forced_memory_probes': general_probes, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Four event trajectories share one original individual and address basis.',
                              'Prompts were used in earlier development evaluation; all values were in adapter training; keys 20-35 were held out.',
                              'P26 may have trained on the held-out keys and prompts.',
                              'Native answers repeat across two prompts; question count is not independent sample count.',
                              'Unknown candidates are selected by native prediction, not independently verified event history.',
                              'Direct observations are treated as ground truth even if the writer rejects them.',
                              'Exact numeric scoring retains full responses for inspection.',
                              'General prompts forcibly use an unrelated numbered address; this is a stress test, not normal routing.',
                              'Generation stops at 20 tokens and may truncate responses.',
                              'No general language quality, historical recall or cross-individual claim.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    run()
