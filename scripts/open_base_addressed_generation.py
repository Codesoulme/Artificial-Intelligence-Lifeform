"""Unrestricted generation against controlled and original neural memory."""
import copy
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.open_base_baseline import MODEL, INDIVIDUAL, INDIVIDUAL_SHA, WEIGHT_SHA, REVISION, sha
from scripts.open_base_addressed_alignment import QUERY_PATH, QUERY_SHA, neural_addresses
from scripts.open_base_real_memory_alignment import variants, pack_states, native_read
from scripts.open_base_synthetic_alignment import encode_prompt
from spora.growing import load
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest
from spora.open_base_memory import OpenBaseMemory, memory_hooks

ADAPTER = ROOT/'runs/open-base-addressed-memory-20261109/adapter.pt'
ADAPTER_SHA = '5fd8bd928967867546efe841c683a4f6b5a3b63418ee6822080eb4b0e44cb455'
OUTPUT = ROOT/'runs/open-base-addressed-generation-20261109.json'


def altered_values(model, original, pair, assignments):
    states = []
    for values in assignments:
        state = copy.deepcopy(original)
        for key, value in zip(pair, values):
            state, _ = observe_sourced(model, state, key, value, 'direct', 1.0)
        if native_read(model, state, list(pair)) != list(values):
            raise ValueError('Native read differs from controlled events')
        states.append(state)
    return states


def run(full_answer=False):
    adapter_path, output_path, adapter_sha = ADAPTER, OUTPUT, ADAPTER_SHA
    expected_format = 'spora-open-base-p26-addressed-v1'
    if full_answer:
        directory = ROOT/'runs/open-base-addressed-full-answer-20261109'
        with (directory/'result.json').open(encoding='utf-8') as stream:
            training_report = json.load(stream)
        adapter_path = directory/'adapter.pt'
        adapter_sha = training_report['adapter_sha256']
        output_path = ROOT/'runs/open-base-full-answer-original-probe-v2-20261109.json'
        expected_format = 'spora-open-base-p26-full-answer-v1'
    if output_path.exists():
        raise FileExistsError(output_path)
    if (sha(MODEL/'model.safetensors') != WEIGHT_SHA or sha(INDIVIDUAL) != INDIVIDUAL_SHA
            or sha(QUERY_PATH) != QUERY_SHA or sha(adapter_path) != adapter_sha):
        raise ValueError('Protected input mismatch')
    checkpoint = torch.load(adapter_path, map_location='cpu', weights_only=True)
    if (checkpoint.get('format') != expected_format
            or checkpoint.get('weight_sha256') != WEIGHT_SHA or checkpoint.get('query_sha256') != QUERY_SHA
            or checkpoint.get('individual_sha256') != INDIVIDUAL_SHA or checkpoint.get('current_only') is not True):
        raise ValueError('Adapter provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    query_model, query_state, _ = load(QUERY_PATH, 'cpu')
    if any(not torch.equal(a.current_basis, b.current_basis) for a, b in zip(organism.memories, query_model.memories)):
        raise ValueError('Address basis differs')
    before = frozen_digest(organism, original)
    query_before = frozen_digest(query_model, query_state)
    seen_values, _ = variants(organism, original, ((8, 9),))
    unseen_values = altered_values(organism, original, (8, 9), ((3, 4), (4, 3)))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='eager').to('cuda').eval()
    adapter = OpenBaseMemory(model.config.hidden_size, organism.config.memory_width).to('cuda').eval()
    adapter.load_state_dict(checkpoint['state'], strict=True)
    layers = [model.model.layers[5], model.model.layers[17]]
    rows = []
    address_probes = []
    if full_answer:
        for question in ('对象99现在的值是多少？', '今天天气怎么样？', '对象12昨天的值是多少？',
                         '对象63现在的值是多少？'):
            try:
                parsed = query_model.chinese_query.parse([question])[0]
                address_probes.append({'question': question, 'parsed': parsed})
            except ValueError as exc:
                address_probes.append({'question': question, 'error': str(exc)})
    with torch.inference_mode():
        for name, states, pair, expected in (
                ('held_keys_seen_values', seen_values[(8, 9)], (8, 9), ((1, 2), (2, 1))),
                (('held_keys_values_3_4' if full_answer else 'held_keys_new_values'),
                 unseen_values, (8, 9), ((3, 4), (4, 3)))):
            for position, key in enumerate(pair):
                question = f'对象{key}的值是多少？'
                addresses, parsed = neural_addresses(query_model, organism, question)
                if parsed['entity'] != key:
                    raise ValueError('Neural address differs from question')
                inputs = encode_prompt(tokenizer, question, 'cuda')
                inputs = {k: v.expand(2, -1) for k, v in inputs.items()}
                with memory_hooks(layers, adapter, pack_states(states), current_only=True, addresses=addresses):
                    output = model.generate(**inputs, max_new_tokens=12, do_sample=False,
                                            pad_token_id=tokenizer.eos_token_id)
                responses = [tokenizer.decode(row[inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip() for row in output]
                correct = [responses[i] in (str(expected[i][position]), str(expected[i][position])+'。',
                                            str(expected[i][position])+'.') for i in range(2)]
                rows.append({'case': name, 'key': key, 'question': question, 'expected': [v[position] for v in expected],
                             'responses': responses, 'exact_numeric': correct})
        question = '对象12现在的值是多少？'
        expected = native_read(organism, original, [12])[0]
        addresses, parsed = neural_addresses(query_model, organism, question)
        if parsed['entity'] != 12 or expected != 7:
            raise ValueError('Original memory probe differs')
        inputs = encode_prompt(tokenizer, question, 'cuda')
        with memory_hooks(layers, adapter, original, current_only=True, addresses=addresses):
            output = model.generate(**inputs, max_new_tokens=12, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        rows.append({'case': 'original_existing_memory', 'key': 12, 'question': question,
                     'expected': [expected], 'responses': [response],
                     'exact_numeric': [response in ('7', '7。', '7.')]})
        if full_answer:
            question = '对象63现在的值是多少？'
            expected = native_read(organism, original, [63])[0]
            addresses, parsed = neural_addresses(query_model, organism, question)
            if parsed['entity'] != 63:
                raise ValueError('Neural address differs from question')
            inputs = encode_prompt(tokenizer, question, 'cuda')
            with memory_hooks(layers, adapter, original, current_only=True, addresses=addresses):
                output = model.generate(**inputs, max_new_tokens=12, do_sample=False,
                                        pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            rows.append({'case': 'original_unchecked_key', 'key': 63, 'question': question,
                         'native_value': expected, 'responses': [response]})
    if (sha(MODEL/'model.safetensors') != WEIGHT_SHA or sha(INDIVIDUAL) != INDIVIDUAL_SHA
            or sha(QUERY_PATH) != QUERY_SHA or sha(adapter_path) != adapter_sha
            or frozen_digest(organism, original) != before or frozen_digest(query_model, query_state) != query_before):
        raise ValueError('Protected input changed')
    report = {'status': 'unrestricted_generation_development_only', 'revision': REVISION,
              'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA,
              'query_sha256': QUERY_SHA, 'adapter_sha256': adapter_sha, 'rows': rows,
              'address_probes': address_probes,
              'seconds': time.perf_counter()-start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Only one controlled background, two held keys and one original memory are tested.',
                              ('The full-answer adapter trained all values from 0 to 15.' if full_answer else
                               'The adapter learned only digits 1 and 2; values 3, 4 and original value 7 are out of training.'),
                              'Exact numeric comparison rejects alternative valid wording; full responses are retained.',
                              'P26 may have seen these numbered questions during its own training.']}
    with output_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(rows, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-answer', action='store_true')
    run(parser.parse_args().full_answer)
