"""Fit SPORA layers to teacher activations on natural and memory prefixes."""
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, SOURCE_CHECKPOINT
from scripts.deepseek_spora_memory_answer_distill import _facts, _load_candidate, _memory_args, _rows
from scripts.deepseek_spora_memory_answer_distill_v4 import TRAIN_INDICES, VALUE_PAIRS
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.integrated_checkpoint import load_integrated
from spora.language_training import atomic_save

DATA = ROOT/'runs/deepseek-natural-distill-data-20261111'
OUTPUT = ROOT/'runs/deepseek-spora-natural-hidden-distill-20261111'
LAYERS = (6, 12)
STEPS = 120
SEED = 18073
MAX_LENGTH = 768


def residual_metrics(block, cache):
    block.eval()
    sums = []
    with torch.inference_mode():
        for example in cache:
            x = example['x'].to('cuda', dtype=torch.float32).unsqueeze(0)
            y = example['y'].to('cuda', dtype=torch.float32).unsqueeze(0)
            out = block(x)
            target = y-x
            error = ((out-y).square().mean()/target.square().mean().clamp_min(1e-8)).item()
            cosine = F.cosine_similarity(out, y, dim=-1).mean().item()
            sums.append({'relative_residual_mse': error, 'hidden_cosine': cosine,
                         'length': x.shape[1], 'group': example['group']})
    return sums


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base weight hash mismatch')
    source = json.loads((DATA/'summary.json').read_text(encoding='utf-8'))
    if source.get('base_sha256') != WEIGHT_SHA or len(source.get('rows', [])) != 3:
        raise ValueError('Teacher trajectory provenance mismatch')
    trajectories = []
    for row in source['rows']:
        path = DATA/row['file']
        if sha(path) != row['sha256']:
            raise ValueError('Teacher trajectory digest mismatch')
        data = torch.load(path, map_location='cpu', weights_only=True)
        ids = data['input_ids']
        if ids.ndim != 1 or ids.dtype != torch.long or not 128 < ids.numel() <= MAX_LENGTH:
            raise ValueError('Unexpected natural trajectory shape or length')
        trajectories.append(ids)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    integrated, organism, _ = load_integrated(SOURCE_CHECKPOINT, base, MODEL, individual_device='cuda')
    state, facts = _facts(organism, fresh_controlled_state(organism), VALUE_PAIRS, TRAIN_INDICES)
    train_memory, held_memory = _rows(facts, tokenizer)
    # Temporal and current reads are mixed; the remaining held entities never train this run.
    train_memory = [row for row in train_memory if row['variant'] == 0]
    held_memory = [row for row in held_memory if row['variant'] == 0]
    caches = {i: {'train_natural': [], 'held_natural': [], 'train_memory': [], 'held_memory': []} for i in LAYERS}

    def collect(ids, group, memory_args):
        output = integrated(input_ids=ids, attention_mask=torch.ones_like(ids),
                            output_hidden_states=True, **memory_args)
        for i in LAYERS:
            caches[i][group].append({'x': output.hidden_states[i][0].detach().half().cpu(),
                                     'y': output.hidden_states[i+1][0].detach().half().cpu(),
                                     'group': group})

    with torch.inference_mode():
        for index, ids in enumerate(trajectories):
            collect(ids.unsqueeze(0).to('cuda'), 'train_natural' if index < 2 else 'held_natural',
                    {'use_cache': False})
            print(json.dumps({'cached_natural': index+1, 'tokens': ids.numel()}), flush=True)
        for group, rows in (('train_memory', train_memory), ('held_memory', held_memory)):
            for row in rows:
                ids = torch.cat((row['prefix']['input_ids'], row['answer_ids']), dim=1)
                collect(ids, group, _memory_args(organism, state, row))
            print(json.dumps({'cached_group': group, 'count': len(rows)}), flush=True)
    # Free the frozen teacher before allocating optimizer states for the SPORA blocks.
    base.to('cpu')
    integrated.reader.to('cpu')
    del integrated, organism, state
    torch.cuda.empty_cache()
    OUTPUT.mkdir(parents=False, exist_ok=False)
    summary = {'base_sha256': WEIGHT_SHA, 'source_checkpoint_sha256': sha(SOURCE_CHECKPOINT),
               'teacher_data_sha256': sha(DATA/'summary.json'),
               'teacher_trajectory_sha256': [row['sha256'] for row in source['rows']],
               'seed': SEED, 'steps_per_layer': STEPS, 'max_length': MAX_LENGTH, 'layers': {},
               'limitations': ['Two natural trajectories trained, one held; not an independent language benchmark.',
                               'Teacher hidden-state targets may not preserve generated answers.',
                               'Memory examples are controlled finite-domain reads with supplied addresses.',
                               'Each block is trained independently using teacher, not student, hidden inputs.',
                               'No online learning, autonomous routing, or final chat acceptance.']}
    for layer in LAYERS:
        block = _load_candidate(PAIR/f'layer-{layer}.pt', layer, base)
        block.extend_positions(MAX_LENGTH)
        validation = caches[layer]['held_natural'] + caches[layer]['held_memory']
        before = residual_metrics(block, validation)
        block.train()
        optimizer = torch.optim.AdamW(block.parameters(), lr=1e-5, weight_decay=0.0, foreach=False)
        rng = random.Random(SEED+layer)
        trace = []
        for step in range(STEPS):
            group = 'train_natural' if step % 2 == 0 else 'train_memory'
            item = rng.choice(caches[layer][group])
            x = item['x'].to('cuda', dtype=torch.float32).unsqueeze(0)
            y = item['y'].to('cuda', dtype=torch.float32).unsqueeze(0)
            prediction = block(x)
            residual_scale = (y-x).square().mean().clamp_min(1e-8)
            loss = (prediction-y).square().mean()/residual_scale
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite natural distillation loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0)
            optimizer.step()
            if step in (0, 19, 39, 79, 119):
                trace.append({'step': step+1, 'group': group, 'length': x.shape[1], 'relative_residual_mse': float(loss.detach())})
                print(json.dumps({'layer': layer, **trace[-1]}), flush=True)
        after = residual_metrics(block, validation)
        checkpoint = OUTPUT/f'layer-{layer}.pt'
        metadata = {'format': 'spora-qwen-layer-replacement-candidate-v1', 'base_weight_sha256': WEIGHT_SHA,
                    'layer': layer, 'width': base.config.hidden_size, 'heads': base.config.num_attention_heads,
                    'max_length': MAX_LENGTH, 'seed': SEED, 'train_steps': STEPS,
                    'objective': 'local_residual_hidden_distillation_natural_and_memory'}
        atomic_save({'metadata': metadata, 'state': {k: v.detach().cpu() for k, v in block.state_dict().items()}}, checkpoint)
        summary['layers'][str(layer)] = {'metadata': metadata, 'start_sha256': sha(PAIR/f'layer-{layer}.pt'),
                                          'candidate_sha256': sha(checkpoint), 'held_before': before,
                                          'held_after': after, 'loss_trace': trace}
        del block, optimizer
        torch.cuda.empty_cache()
    summary['elapsed_seconds'] = time.perf_counter()-started
    summary['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    with (OUTPUT/'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'completed_layers': list(summary['layers'])}), flush=True)


if __name__ == '__main__':
    run()
