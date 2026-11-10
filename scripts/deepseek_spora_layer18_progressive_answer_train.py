"""Answer-train layer 18 inside a low-strength graft after SPORA memory."""
import json
import random
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
from scripts.deepseek_named_binding_probe import KEYS
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_memory_answer_distill import (
    LAYERS as PAIR_LAYERS, OUTPUT as ANSWER_V4, SOURCE_CHECKPOINT,
    _facts, _load_candidate, _memory_args, _rows,
)
from scripts.deepseek_spora_memory_answer_distill_v4 import TRAIN_INDICES, VALUE_PAIRS
from scripts.deepseek_spora_postmemory_layer18_answer_distill import LAYER as POST_LAYER
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.integrated_checkpoint import load_integrated
from spora.structure_transfer import SPORALayerBlend, SPORALayerReplacement

OUTPUT = ROOT / 'runs/deepseek-spora-layer18-progressive-answer-20261111'
START = ROOT / 'runs/deepseek-spora-postmemory-layer18-answer-20261111/layer-18.pt'
GRAFT_STRENGTH = 0.05
SWEEP = (0.0, 0.025, 0.05, 0.075, 0.1, 0.2, 0.4, 1.0)
STEPS = 140
SEED = 11827


def _run_answers(integrated, base, organism, state, pair, candidate, original18, rows, tokenizer, strength):
    base.model.layers[6], base.model.layers[12] = pair[6], pair[12]
    base.model.layers[18] = SPORALayerBlend(original18, candidate, strength).to('cuda').eval()
    results = []
    try:
        with torch.inference_mode():
            for row in rows:
                prompt = row['prefix']
                generated = integrated.generate(**prompt, **_memory_args(organism, state, row),
                                                max_new_tokens=8, do_sample=False,
                                                pad_token_id=tokenizer.pad_token_id,
                                                eos_token_id=tokenizer.eos_token_id)
                answer = tokenizer.decode(generated[0, prompt['input_ids'].shape[1]:],
                                          skip_special_tokens=True).strip()
                correct = WORDS[row['value']] in answer and WORDS[row['other_value']] not in answer
                results.append({'name': row['name'], 'intent': row['intent'], 'question': row['question'],
                                'expected': WORDS[row['value']], 'answer': answer, 'correct': correct})
    finally:
        base.model.layers[18] = original18
    return {'correct': sum(row['correct'] for row in results), 'total': len(results),
            'accuracy': sum(row['correct'] for row in results) / len(results), 'rows': results}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected DeepSeek base hash mismatch')
    torch.manual_seed(SEED)
    random.seed(SEED)
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
    pair = {layer: _load_candidate(ANSWER_V4/f'layer-{layer}.pt', layer, base) for layer in PAIR_LAYERS}
    payload = torch.load(START, map_location='cpu', weights_only=True)
    if payload.get('metadata', {}).get('base_weight_sha256') != WEIGHT_SHA or payload['metadata'].get('layer') != POST_LAYER:
        raise ValueError('Layer-18 start candidate provenance mismatch')
    candidate = SPORALayerReplacement(base.config.hidden_size, base.config.num_attention_heads,
                                      max_length=128, layer_idx=18).to('cuda')
    candidate.load_state_dict(payload['state'], strict=True)
    original18 = base.model.layers[18]
    before = _run_answers(integrated, base, organism, state, pair, candidate,
                          original18, held_rows, tokenizer, GRAFT_STRENGTH)
    targets = []
    for row in train_rows:
        full_ids = torch.cat((row['prefix']['input_ids'], row['answer_ids']), dim=1)
        targets.append({'row': row, 'input_ids': full_ids,
                        'attention_mask': torch.ones_like(full_ids),
                        'prefix_len': row['prefix']['input_ids'].shape[1],
                        'answer_len': row['answer_ids'].shape[1]})
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=1e-6, weight_decay=0.01, foreach=False)
    order = list(range(len(targets)))
    rng = random.Random(SEED + 1)
    trace = []
    candidate.train()
    for step in range(STEPS):
        if step % len(order) == 0:
            rng.shuffle(order)
        item = targets[order[step % len(order)]]
        graft = SPORALayerBlend(original18, candidate, GRAFT_STRENGTH).to('cuda').train()
        base.model.layers[6], base.model.layers[12], base.model.layers[18] = pair[6], pair[12], graft
        output = integrated(input_ids=item['input_ids'], attention_mask=item['attention_mask'],
                            **_memory_args(organism, state, item['row']))
        start = item['prefix_len'] - 1
        logits = output.logits[:, start:start + item['answer_len']].float()
        labels = item['input_ids'][:, item['prefix_len']:item['prefix_len'] + item['answer_len']]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        if not bool(torch.isfinite(loss)):
            base.model.layers[18] = original18
            raise ValueError('Nonfinite progressive graft answer loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
        optimizer.step()
        base.model.layers[18] = original18
        if step in (0, 9, 19, 39, 79, 119, 139):
            trace.append({'step': step + 1, 'answer_cross_entropy': float(loss.detach())})
            print(json.dumps(trace[-1]), flush=True)
    candidate.eval()
    sweep = {}
    for strength in SWEEP:
        sweep[str(strength)] = _run_answers(integrated, base, organism, state, pair, candidate,
                                            original18, held_rows, tokenizer, strength)
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT / 'layer-18.pt'
    metadata = {'format': 'spora-qwen-layer-replacement-candidate-v1',
                'base_weight_sha256': WEIGHT_SHA, 'layer': 18, 'width': base.config.hidden_size,
                'heads': base.config.num_attention_heads, 'max_length': 128, 'seed': SEED,
                'train_steps': STEPS, 'training_graft_strength': GRAFT_STRENGTH,
                'objective': 'native_memory_answer_cross_entropy_progressive_graft'}
    torch.save({'metadata': metadata,
                'state': {name: tensor.detach().cpu() for name, tensor in candidate.state_dict().items()}}, checkpoint)
    report = {'status': 'answer_trained_progressive_postmemory_spora_graft',
              'metadata': metadata, 'source_integrated_checkpoint_sha256': sha(SOURCE_CHECKPOINT),
              'upstream_pair_sha256': {str(layer): sha(ANSWER_V4/f'layer-{layer}.pt') for layer in PAIR_LAYERS},
              'starting_layer18_sha256': sha(START), 'candidate_sha256': sha(checkpoint),
              'training_queries': len(train_rows), 'held_queries': len(held_rows),
              'held_before_at_training_strength': before, 'held_strength_sweep': sweep,
              'loss_trace': trace, 'elapsed_seconds': time.perf_counter()-started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'device': torch.cuda.get_device_name(0),
              'limitations': [
                  'Six training entities and two held entities from one finite beverage-memory task.',
                  'Fixed blend strength during optimization; strength sweep reuses the same held set.',
                  'Layer 18 keeps the original Qwen layer in the residual blend; this is a staged graft.',
                  'No autonomous language routing or broad open-chat evaluation.',
              ]}
    with (OUTPUT/'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({key: value['accuracy'] for key, value in sweep.items()}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
