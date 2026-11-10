"""Exercise generated memory answers before and after an online correction."""
import json
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_memory_answer_distill import LAYERS
from scripts.deepseek_spora_memory_answer_distill_v4 import VALUE_PAIRS
from scripts.deepseek_spora_memory_answer_distill import _load_candidate
from scripts.open_base_baseline import INDIVIDUAL_SHA
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.integrated_checkpoint import IntegratedSPORARuntime, load_integrated
from spora.internal_memory import read_provenance

SOURCE_CHECKPOINT = ROOT / 'runs/deepseek-temporal-intent-router-native-spora-site-20261110/historical-individual.pt'
CANDIDATES = ROOT / 'runs/deepseek-spora-memory-answer-distill-v4-20261111'
OUTPUT = ROOT / 'runs/deepseek-spora-online-correction-answer-20261111.json'
ENTITY_INDEX = 5


def _addresses(runtime, key, intent, event_time):
    organism = runtime.organism
    keys = torch.tensor([key], dtype=torch.long, device='cuda')
    if intent == 'current':
        return [torch.nn.functional.normalize(site.current_basis[key:key + 1].to('cuda').float(), dim=-1)
                for site in organism.memories], 0
    modes = torch.ones_like(keys)
    times = torch.tensor([event_time], dtype=torch.long, device='cuda')
    _, bound = organism.representation(keys, modes, times % organism.config.time_count)
    return [organism.memory_address(site, bound, keys, modes) for site in organism.memories], 1


def _query(runtime, tokenizer, key, question, intent, event_time, expected, condition):
    addresses, slot = _addresses(runtime, key, intent, event_time)
    inputs = closed_think_inputs(tokenizer, question, 'cuda')
    old_layers = {layer: runtime.decoder.base.model.layers[layer] for layer in LAYERS}
    if condition == 'spora_pair':
        for layer in LAYERS:
            runtime.decoder.base.model.layers[layer] = runtime._probe_blocks[layer]
    try:
        with torch.inference_mode():
            generated = runtime.generate(
                **inputs, memory_addresses=addresses,
                memory_slots=torch.tensor([slot], dtype=torch.long, device='cuda'),
                current_only=False, max_new_tokens=8, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    finally:
        for layer, module in old_layers.items():
            runtime.decoder.base.model.layers[layer] = module
    answer = tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:],
                              skip_special_tokens=True).strip()
    return {'condition': condition, 'question': question, 'intent': intent,
            'expected': WORDS[expected], 'answer': answer,
            'correct': WORDS[expected] in answer}


def run(entity_index=ENTITY_INDEX, output=OUTPUT):
    if type(entity_index) is not int or not 0 <= entity_index < 8:
        raise ValueError('Entity index must be in [0, 7]')
    output = Path(output)
    entity = NAMES[entity_index]
    key = KEYS[entity_index]
    old_value, first_value = VALUE_PAIRS[entity_index]
    corrected_value = next(value for value in range(len(WORDS)) if value not in (old_value, first_value))
    if output.exists():
        raise FileExistsError(output)
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected DeepSeek base hash mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float16,
        attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    decoder, organism, _ = load_integrated(SOURCE_CHECKPOINT, base, MODEL, individual_device='cuda')
    state = fresh_controlled_state(organism)
    runtime = IntegratedSPORARuntime(decoder, organism, state, MODEL)
    blocks = {layer: _load_candidate(CANDIDATES / f'layer-{layer}.pt', layer, base)
              for layer in LAYERS}
    runtime._probe_blocks = blocks
    events = []
    for value in (old_value, first_value):
        decision = runtime.observe_sourced(key, value, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError(f'Initial event write rejected: {decision}')
        events.append({'time': runtime.state.cursor - 1, 'value': value, 'action': decision['action']})
    queries = [
        ('改正后{name}最喜欢喝什么饮料？'.format(name=entity), 'current', 1, first_value),
        ('偏好调整之前，{name}喝什么？'.format(name=entity), 'past', 0, old_value),
        ('第一次更正后，{name}喝什么？'.format(name=entity), 'later', 1, first_value),
    ]
    rows = []
    for question, intent, event_time, value in queries:
        for condition in ('base', 'spora_pair'):
            rows.append(_query(runtime, tokenizer, key, question, intent, event_time, value, condition))
    decision = runtime.observe_sourced(key, corrected_value, 'direct', 1.0)
    if decision['action'] != 'correct' or not decision['current_write']:
        raise ValueError(f'Online correction rejected: {decision}')
    events.append({'time': runtime.state.cursor - 1, 'value': corrected_value,
                   'action': decision['action']})
    queries_after = [
        ('更正后{name}现在爱喝什么？'.format(name=entity), 'current', 2, corrected_value),
        ('最早的记录里{name}喜欢什么饮料？'.format(name=entity), 'past', 0, old_value),
        ('上一次调整后的偏好是什么？'.format(name=entity), 'later', 1, first_value),
        ('最新改过之后，{name}喝什么？'.format(name=entity), 'later', 2, corrected_value),
    ]
    for question, intent, event_time, value in queries_after:
        for condition in ('base', 'spora_pair'):
            rows.append(_query(runtime, tokenizer, key, question, intent, event_time, value, condition))
    provenance = [read_provenance(organism, runtime.state, key, t) for t in range(runtime.state.cursor)]
    summary = {}
    for condition in ('base', 'spora_pair'):
        subset = [row for row in rows if row['condition'] == condition]
        summary[condition] = {'correct': sum(row['correct'] for row in subset),
                              'total': len(subset),
                              'accuracy': sum(row['correct'] for row in subset) / len(subset)}
    report = {
        'status': 'online_correction_generated_answer_development_probe',
        'base_weight_sha256': WEIGHT_SHA,
        'source_individual_sha256': INDIVIDUAL_SHA,
        'source_integrated_checkpoint_sha256': sha(SOURCE_CHECKPOINT),
        'replacement_candidate_sha256': {str(layer): sha(CANDIDATES / f'layer-{layer}.pt')
                                         for layer in LAYERS},
        'entity': entity,
        'entity_index': entity_index,
        'events': events,
        'event_provenance': provenance,
        'rows': rows,
        'summary': summary,
        'elapsed_seconds': time.perf_counter() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'device': torch.cuda.get_device_name(0),
        'limitations': [
            'One entity, one key, three sequential drink-preference events and seven authored questions.',
            'Entity address and temporal page are supplied by the probe, not parsed by the model.',
            'Eight-token generations and exact word-in-answer scoring are not open-chat acceptance.',
            'Memory write controller and base language weights remain frozen.',
        ],
    }
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity-index', type=int, default=ENTITY_INDEX)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.entity_index, args.output)
