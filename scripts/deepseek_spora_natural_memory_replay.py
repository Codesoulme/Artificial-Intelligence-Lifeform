"""Replay memory answers while retaining natural-prefix output distributions."""
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
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as ROOT_PAIR, SOURCE_CHECKPOINT
from scripts.deepseek_spora_memory_answer_distill import _facts, _load_candidate, _memory_args, _rows
from scripts.deepseek_spora_memory_answer_distill_v4 import TRAIN_INDICES, VALUE_PAIRS
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.integrated_checkpoint import load_integrated
from spora.language_training import atomic_save

START = ROOT/'runs/deepseek-spora-natural-hidden-distill-20261111'
DATA = ROOT/'runs/deepseek-natural-distill-data-20261111'
OUTPUT = ROOT/'runs/deepseek-spora-natural-memory-replay-20261111'
LAYERS = (6, 12)
STEPS = 90
SEED = 712911
VALUE_PAIR_SETS = (VALUE_PAIRS,)


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base model hash mismatch')
    source = json.loads((START/'result.json').read_text(encoding='utf-8'))
    sequences = json.loads((DATA/'summary.json').read_text(encoding='utf-8'))
    for i in LAYERS:
        info = source['layers'][str(i)]
        root_hash = info.get('root_sha256', info['start_sha256'])
        if sha(START/f'layer-{i}.pt') != info['candidate_sha256'] or sha(ROOT_PAIR/f'layer-{i}.pt') != root_hash:
            raise ValueError('Candidate provenance mismatch')
    if sha(DATA/'summary.json') != source.get('teacher_data_sha256', source.get('natural_data_sha256')):
        raise ValueError('Natural data provenance mismatch')
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    integrated, organism, _ = load_integrated(SOURCE_CHECKPOINT, base, MODEL, individual_device='cuda')
    integrated.reader.requires_grad_(False)
    memory_rows = []
    for pairs in VALUE_PAIR_SETS:
        state, facts = _facts(organism, fresh_controlled_state(organism), pairs, TRAIN_INDICES)
        examples, _ = _rows(facts, tokenizer)
        memory_rows.extend((row, state) for row in examples)
    blocks = {i: _load_candidate(START/f'layer-{i}.pt', i, base) for i in LAYERS}
    for i in LAYERS:
        base.model.layers[i] = blocks[i]
    natural = []
    with torch.inference_mode():
        for row in sequences['rows'][:2]:
            path = DATA/row['file']
            if sha(path) != row['sha256']:
                raise ValueError('Teacher sequence digest mismatch')
            ids = torch.load(path, map_location='cpu', weights_only=True)['input_ids']
            for length in (128, 192):
                if ids.numel() <= length:
                    raise ValueError('Natural prefix too short')
                prefix = ids[:length].clone().unsqueeze(0).to('cuda')
                logits = integrated(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                                    use_cache=False).logits[:, -1].float().detach().half().cpu()
                natural.append({'input_ids': prefix, 'teacher': logits, 'length': length})
    # inference-mode tensors cannot be used as labels in backwards; clone inputs here.
    for item in natural:
        item['input_ids'] = item['input_ids'].clone()
    print(json.dumps({'memory_questions': len(memory_rows), 'natural_prefixes': len(natural)}), flush=True)
    optimizer = torch.optim.AdamW([p for block in blocks.values() for p in block.parameters()],
                                  lr=8e-7, weight_decay=0.01, foreach=False)
    rng = random.Random(SEED)
    trace = []
    for step in range(STEPS):
        if step % 3 == 0:
            item = rng.choice(natural)
            ids = item['input_ids']
            logits = integrated(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[:, -1].float()
            teacher = item['teacher'].to('cuda', dtype=torch.float32)
            loss = F.kl_div(F.log_softmax(logits/2, dim=-1), F.softmax(teacher/2, dim=-1),
                            reduction='batchmean') * 4
            group = 'natural_kl'
        else:
            row, row_state = rng.choice(memory_rows)
            prefix, answer = row['prefix']['input_ids'], row['answer_ids']
            ids = torch.cat((prefix, answer), dim=1)
            output = integrated(input_ids=ids, attention_mask=torch.ones_like(ids), **_memory_args(organism, row_state, row))
            start = prefix.shape[1]-1
            logits = output.logits[:, start:start+answer.shape[1]].float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), answer.reshape(-1))
            group = 'memory_answer_ce'
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite replay loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for block in blocks.values() for p in block.parameters()], 1.0)
        optimizer.step()
        if step in (0, 9, 19, 39, 59, 89):
            entry = {'step': step+1, 'group': group, 'loss': float(loss.detach())}
            trace.append(entry)
            print(json.dumps(entry), flush=True)
    OUTPUT.mkdir(parents=False, exist_ok=False)
    report = {'base_sha256': WEIGHT_SHA, 'source_checkpoint_sha256': sha(SOURCE_CHECKPOINT),
              'natural_data_sha256': sha(DATA/'summary.json'), 'seed': SEED, 'steps': STEPS,
              'memory_training_questions': len(memory_rows), 'memory_value_pair_sets': VALUE_PAIR_SETS,
              'natural_training_prefixes': len(natural),
              'layers': {}, 'trace': trace,
              'limitations': ['Controlled memory questions and two incomplete teacher sequences only.',
                              'Natural KL is measured at the last token of four prefixes, not full trajectories.',
                              'No autonomous memory routing or independent open-chat acceptance.']}
    for i in LAYERS:
        checkpoint = OUTPUT/f'layer-{i}.pt'
        metadata = {'format': 'spora-qwen-layer-replacement-candidate-v1', 'base_weight_sha256': WEIGHT_SHA,
                    'layer': i, 'width': base.config.hidden_size, 'heads': base.config.num_attention_heads,
                    'max_length': blocks[i].max_length, 'seed': SEED, 'train_steps': STEPS,
                    'objective': 'memory_answer_replay_with_natural_prefix_kl'}
        atomic_save({'metadata': metadata, 'state': {k: v.detach().cpu() for k, v in blocks[i].state_dict().items()}}, checkpoint)
        report['layers'][str(i)] = {'metadata': metadata, 'start_sha256': sha(START/f'layer-{i}.pt'),
                                    'root_sha256': sha(ROOT_PAIR/f'layer-{i}.pt'), 'candidate_sha256': sha(checkpoint)}
    report['elapsed_seconds'] = time.perf_counter()-started
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    with (OUTPUT/'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': 'trained', 'elapsed_seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    run()
