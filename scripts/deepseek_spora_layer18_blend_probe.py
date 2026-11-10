"""Calibrate a progressive SPORA graft after memory site 17."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_memory_answer_distill import (
    LAYERS as PAIR_LAYERS, OUTPUT as ANSWER_V4, SOURCE_CHECKPOINT,
    _facts, _load_candidate, _memory_args, _rows,
)
from scripts.deepseek_spora_memory_answer_distill_v4 import TRAIN_INDICES, VALUE_PAIRS
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.integrated_checkpoint import load_integrated
from spora.structure_transfer import SPORALayerBlend, SPORALayerReplacement

LAYER = 18
LAYER18 = ROOT / 'runs/deepseek-spora-postmemory-layer18-answer-20261111/layer-18.pt'
OUTPUT = ROOT / 'runs/deepseek-spora-layer18-blend-20261111.json'
STRENGTHS = (0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 1.0)


def _batch(rows, tokenizer, state, organism, device, include_answers=True):
    examples, controls = [], []
    for row in rows:
        inputs = row['prefix']
        memory = _memory_args(organism, state, row)
        full = (torch.cat((inputs['input_ids'], row['answer_ids']), dim=1) if include_answers
                else inputs['input_ids'])
        examples.append((full, torch.ones_like(full), inputs['input_ids'].shape[1], row['answer_ids'].shape[1]))
        controls.append(memory)
    max_len = max(item[0].shape[1] for item in examples)
    ids, masks, labels = [], [], []
    for full, mask, prefix_len, answer_len in examples:
        pad = max_len - full.shape[1]
        ids.append(F.pad(full, (pad, 0), value=tokenizer.pad_token_id))
        masks.append(F.pad(mask, (pad, 0), value=0))
        labels.append((pad + prefix_len - 1, answer_len))
    batch_ids, batch_mask = torch.cat(ids), torch.cat(masks)
    addresses = [torch.cat([item['memory_addresses'][site] for item in controls], dim=0)
                 for site in range(2)]
    slots = torch.cat([item['memory_slots'] for item in controls], dim=0)
    return {'input_ids': batch_ids.to(device), 'attention_mask': batch_mask.to(device)}, addresses, slots, labels


@torch.inference_mode()
def _target_nll(model, rows, tokenizer, organism, state, device, chunk_size=4):
    scores = []
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start:start + chunk_size]
        inputs, addresses, slots, spans = _batch(chunk, tokenizer, state, organism, device)
        output = model(**inputs, memory_state=state, memory_addresses=addresses,
                       memory_slots=slots, current_only=False, use_cache=False).logits.float()
        for index, (begin, length) in enumerate(spans):
            prediction = output[index:index + 1, begin:begin + length]
            target = chunk[index]['answer_ids']
            scores.append(float(F.cross_entropy(prediction.reshape(-1, prediction.shape[-1]),
                                                 target.reshape(-1))))
    return {'mean_answer_nll': sum(scores) / len(scores), 'examples': len(scores)}


@torch.inference_mode()
def _generate(model, rows, tokenizer, organism, state, device, chunk_size=4):
    result = []
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start:start + chunk_size]
        inputs, addresses, slots, _ = _batch(chunk, tokenizer, state, organism, device, include_answers=False)
        generated = model.generate(**inputs, memory_state=state, memory_addresses=addresses,
                                   memory_slots=slots, current_only=False, max_new_tokens=8,
                                   do_sample=False, use_cache=False,
                                   pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        prompt_len = inputs['input_ids'].shape[1]
        for index, row in enumerate(chunk):
            answer = tokenizer.decode(generated[index, prompt_len:], skip_special_tokens=True).strip()
            expected, other = row['value'], row['other_value']
            expected_word = row['expected_word']
            other_word = row['other_word']
            result.append({'name': row['name'], 'intent': row['intent'],
                           'question': row['question'], 'expected': expected_word, 'answer': answer,
                           'correct': expected_word in answer and other_word not in answer})
    return {'correct': sum(row['correct'] for row in result), 'total': len(result),
            'accuracy': sum(row['correct'] for row in result) / len(result), 'rows': result}


def _attach_words(rows):
    from scripts.deepseek_semantic_value_pilot import WORDS
    for row in rows:
        row['expected_word'] = WORDS[row['value']]
        row['other_word'] = WORDS[row['other_value']]


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base model hash mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float16,
        attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    integrated, organism, _ = load_integrated(SOURCE_CHECKPOINT, base, MODEL, individual_device='cuda')
    integrated.reader.requires_grad_(False)
    state, facts = _facts(organism, fresh_controlled_state(organism), VALUE_PAIRS, TRAIN_INDICES)
    train_rows, held_rows = _rows(facts, tokenizer)
    _attach_words(train_rows)
    _attach_words(held_rows)
    pair = {layer: _load_candidate(ANSWER_V4/f'layer-{layer}.pt', layer, base) for layer in PAIR_LAYERS}
    payload = torch.load(LAYER18, map_location='cpu', weights_only=True)
    if payload.get('metadata', {}).get('base_weight_sha256') != WEIGHT_SHA or payload['metadata'].get('layer') != LAYER:
        raise ValueError('Layer-18 candidate provenance mismatch')
    block = SPORALayerReplacement(base.config.hidden_size, base.config.num_attention_heads,
                                  max_length=128, layer_idx=LAYER).to('cuda').eval()
    block.load_state_dict(payload['state'], strict=True)
    originals = {layer: base.model.layers[layer] for layer in (*PAIR_LAYERS, LAYER)}
    for layer in PAIR_LAYERS:
        base.model.layers[layer] = pair[layer]
    nll_results, held_results = {}, {}
    chosen_strength = None
    try:
        for strength in STRENGTHS:
            graft = SPORALayerBlend(originals[LAYER], block, strength).to('cuda').eval()
            base.model.layers[LAYER] = graft
            nll_results[str(strength)] = _target_nll(integrated, train_rows, tokenizer, organism, state, 'cuda')
            held_results[str(strength)] = _generate(integrated, held_rows, tokenizer, organism, state, 'cuda')
            print(f'strength {strength}: train_nll={nll_results[str(strength)]["mean_answer_nll"]:.4f}, '
                  f'held={held_results[str(strength)]["correct"]}/{held_results[str(strength)]["total"]}', flush=True)
        chosen_strength = min(STRENGTHS, key=lambda value: (nll_results[str(value)]['mean_answer_nll'], value))
    finally:
        for layer, module in originals.items():
            base.model.layers[layer] = module
    report = {
        'status': 'postmemory_layer18_progressive_graft_strength_sweep',
        'base_weight_sha256': WEIGHT_SHA,
        'source_integrated_checkpoint_sha256': sha(SOURCE_CHECKPOINT),
        'upstream_pair_sha256': {str(layer): sha(ANSWER_V4/f'layer-{layer}.pt') for layer in PAIR_LAYERS},
        'layer18_candidate_sha256': sha(LAYER18),
        'layers': [6, 12, 18],
        'strengths': list(STRENGTHS),
        'training_answer_nll': nll_results,
        'held_generated_answers': held_results,
        'selected_strength_by_train_nll': chosen_strength,
        'elapsed_seconds': time.perf_counter()-started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'device': torch.cuda.get_device_name(0),
        'limitations': [
            'One held entity split in one finite beverage domain.',
            'Blend strength is selected by teacher-forced NLL on the training entities.',
            'Cache-disabled generation is used because the graft evaluates both original and SPORA branches.',
            'This is a progressive structural interpolation, not a fully replaced third layer.',
        ],
    }
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'selected_strength': chosen_strength,
                      'held_accuracy': held_results[str(chosen_strength)]['accuracy']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
