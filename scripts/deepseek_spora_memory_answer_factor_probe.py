"""Measure generated memory-answer accuracy across reader strengths and layer pairs."""
import json
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
from scripts.deepseek_spora_memory_strength_factor_probe import _address, _module
from scripts.deepseek_spora_memory_strength_factor_probe import CONDITION_PATHS
from scripts.open_base_baseline import INDIVIDUAL_SHA
from scripts.spora_unknown_state_probe import fresh_controlled_state
from scripts.deepseek_semantic_value_pilot import WORDS
from spora.integrated_checkpoint import load_integrated
from spora.internal_memory import observe_sourced

SOURCE_CHECKPOINT = ROOT / 'runs/deepseek-temporal-intent-router-native-spora-site-20261110/historical-individual.pt'
OUTPUT = ROOT / 'runs/deepseek-spora-memory-answer-factor-20261111.json'
STRENGTHS = (0.1, 0.25, 0.5, 0.75)
INTENTS = {
    'current': ('{name}现在喜欢什么饮品？', 6, 0),
    'past': ('{name}过去喜欢哪种饮料？', 1, 1),
    'later': ('更正后{name}改喜欢什么饮品？', 6, 1),
}
CONDITIONS = ('base', 'plain_pair', 'memory_conditioned_layer6', 'joint_memory_v2')


def _answer_correct(answer, expected, other):
    return WORDS[expected] in answer and WORDS[other] not in answer


def _batch_questions(tokenizer, names):
    inputs = [closed_think_inputs(tokenizer, name, 'cuda') for name in names]
    max_len = max(row['input_ids'].shape[1] for row in inputs)
    ids = torch.cat([torch.nn.functional.pad(row['input_ids'], (max_len-row['input_ids'].shape[1], 0),
                                              value=tokenizer.pad_token_id)
                     for row in inputs], dim=0)
    mask = torch.cat([torch.nn.functional.pad(row['attention_mask'],
                                              (max_len-row['attention_mask'].shape[1], 0), value=0)
                      for row in inputs], dim=0)
    return {'input_ids': ids, 'attention_mask': mask}, max_len


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base model hash mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        raise ValueError('Tokenizer must provide a pad token')
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float16,
        attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    integrated, organism, _ = load_integrated(SOURCE_CHECKPOINT, base, MODEL, individual_device='cuda')
    state = fresh_controlled_state(organism)
    for key in KEYS[:4]:
        for value in (1, 6):
            state, decision = observe_sourced(organism, state, key, value, 'direct', 1.0)
            if not decision['current_write']:
                raise ValueError(f'Controlled write rejected: key={key}, decision={decision}')
    original = {layer: base.model.layers[layer] for layer in (6, 12)}
    replacements = {condition: {layer: _module(path, layer, base)
                                for layer, path in paths.items()}
                    for condition, paths in CONDITION_PATHS.items()}
    condition_modules = {'base': None, **replacements}
    rows = []
    try:
        with torch.inference_mode():
            for strength in STRENGTHS:
                integrated.reader.strength = strength
                for intent, (template, expected, slot) in INTENTS.items():
                    questions = [template.format(name=name) for name in NAMES[:4]]
                    inputs, prompt_len = _batch_questions(tokenizer, questions)
                    addresses = [_address(organism, key, intent, index)[0]
                                 for index, key in enumerate(KEYS[:4])]
                    batched_addresses = [torch.cat([entity[site] for entity in addresses], dim=0)
                                         for site in range(2)]
                    slots = torch.full((len(questions),), slot, dtype=torch.long, device='cuda')
                    for condition, modules in condition_modules.items():
                        for layer in (6, 12):
                            base.model.layers[layer] = original[layer] if modules is None else modules[layer]
                        generated = integrated.generate(
                            **inputs, memory_state=state, memory_addresses=batched_addresses,
                            memory_slots=slots, current_only=False, max_new_tokens=8, do_sample=False,
                            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                        for index, name in enumerate(NAMES[:4]):
                            continuation = generated[index, prompt_len:]
                            answer = tokenizer.decode(continuation, skip_special_tokens=True).strip()
                            wrong = 1 if expected == 6 else 6
                            rows.append({'strength': strength, 'condition': condition,
                                         'name': name, 'key': KEYS[index], 'intent': intent,
                                         'question': questions[index], 'expected': WORDS[expected],
                                         'answer': answer,
                                         'correct': _answer_correct(answer, expected, wrong),
                                         'generated_tokens': int(continuation.numel())})
    finally:
        integrated.reader.strength = 0.5
        for layer, module in original.items():
            base.model.layers[layer] = module
    summary = {}
    for strength in STRENGTHS:
        for condition in CONDITIONS:
            selected = [row for row in rows if row['strength'] == strength and row['condition'] == condition]
            summary[f'{condition}@{strength}'] = {
                'correct': sum(row['correct'] for row in selected),
                'total': len(selected),
                'accuracy': sum(row['correct'] for row in selected) / len(selected),
            }
    report = {
        'status': 'multi_entity_native_memory_generated_answer_factor_probe',
        'base_weight_sha256': WEIGHT_SHA,
        'individual_checkpoint_sha256': sha(SOURCE_CHECKPOINT),
        'strengths': list(STRENGTHS),
        'entities': list(NAMES[:4]),
        'conditions': list(CONDITIONS),
        'queries': len(rows),
        'summary': summary,
        'rows': rows,
        'elapsed_seconds': time.perf_counter() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'device': torch.cuda.get_device_name(0),
        'limitations': [
            'Four entities, one old and one current value per entity, and three authored temporal query forms.',
            'Eight-token greedy answer with exact expected-word check; paraphrases may be marked incorrect.',
            'All four entities share the same two values, so it does not test interference between distinct values.',
            'This is a development probe, not independent chat or memory safety acceptance.',
        ],
    }
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
