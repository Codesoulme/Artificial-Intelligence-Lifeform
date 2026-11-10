"""Train a full layer-18 substitute with output distillation and memory answers."""
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

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, SOURCE_CHECKPOINT
from scripts.deepseek_spora_memory_answer_distill import _facts, _load_candidate, _memory_args, _rows
from scripts.deepseek_spora_memory_answer_distill_v4 import TRAIN_INDICES, VALUE_PAIRS
from scripts.deepseek_spora_postmemory_layer18_answer_distill import _evaluate
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.integrated_checkpoint import load_integrated
from spora.structure_transfer import SPORALayerBlend

START = ROOT / 'runs/deepseek-spora-layer18-progressive-answer-20261111/layer-18.pt'
OUTPUT = ROOT / 'runs/deepseek-spora-layer18-curriculum-distill-20261111'
GENERAL_TRAIN = ('请简短介绍一下春天。', '解释什么是诚信。', '一加一等于几？',
                 '用一句话描述一只小猫。', '朋友难过时该怎么安慰？', '写一句问候语。')
GENERAL_HELD = ('用一句话解释什么是友善。', '请简述雨是怎么形成的。')
STEPS = 160
SEED = 99183


def answer_logits(integrated, base, organism, state, item):
    if item['kind'] == 'memory':
        out = integrated(**item['inputs'], **_memory_args(organism, state, item['row']))
    else:
        out = base(**item['inputs'], use_cache=False)
    return out.logits[:, -1, :].float()


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base model hash mismatch')
    predecessor = json.loads((START.parent / 'result.json').read_text(encoding='utf-8'))
    if sha(START) != predecessor['candidate_sha256'] or any(
        sha(PAIR/f'layer-{i}.pt') != predecessor['upstream_pair_sha256'][str(i)] for i in (6, 12)
    ):
        raise ValueError('Candidate and upstream pair provenance mismatch')
    torch.manual_seed(SEED)
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
    original = {i: base.model.layers[i] for i in (6, 12, 18)}
    pair = {i: _load_candidate(PAIR/f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    candidate = _load_candidate(START, 18, base)
    for i in (6, 12):
        base.model.layers[i] = pair[i]

    items = [{'kind': 'memory', 'row': row, 'inputs': row['prefix'],
              'label': row['answer_ids'][0, 0]} for row in train_rows]
    items += [{'kind': 'general', 'inputs': closed_think_inputs(tokenizer, q, 'cuda'), 'question': q}
              for q in GENERAL_TRAIN]
    held_general = [{'kind': 'general', 'inputs': closed_think_inputs(tokenizer, q, 'cuda'),
                     'question': q} for q in GENERAL_HELD]
    with torch.inference_mode():
        for item in items + held_general:
            item['teacher'] = answer_logits(integrated, base, organism, state, item).detach().half().cpu()
    print(json.dumps({'teacher_examples': len(items), 'held_general': len(held_general)}), flush=True)
    before = _evaluate(integrated, base, organism, state, pair, candidate, original, tokenizer, held_rows)
    for i in (6, 12):
        base.model.layers[i] = pair[i]
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=2e-6, weight_decay=0.01, foreach=False)
    randomizer = random.Random(SEED)
    memory_items = [x for x in items if x['kind'] == 'memory']
    general_items = [x for x in items if x['kind'] == 'general']
    trace = []
    candidate.train()
    for step in range(STEPS):
        item = randomizer.choice(general_items if step % 3 == 0 else memory_items)
        strength = (0.1, 0.3, 0.6, 1.0)[min(step * 4 // STEPS, 3)]
        base.model.layers[18] = SPORALayerBlend(original[18], candidate, strength).to('cuda')
        logits = answer_logits(integrated, base, organism, state, item)
        teacher = item['teacher'].to('cuda', dtype=torch.float32)
        distill = F.kl_div(F.log_softmax(logits / 2.0, dim=-1),
                           F.softmax(teacher / 2.0, dim=-1), reduction='batchmean') * 4.0
        loss = distill
        if item['kind'] == 'memory':
            loss = loss + F.cross_entropy(logits, item['label'].reshape(1))
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
        optimizer.step()
        base.model.layers[18] = original[18]
        if step in (0, 39, 79, 119, 159):
            entry = {'step': step + 1, 'strength': strength, 'kind': item['kind'],
                     'loss': float(loss.detach()), 'teacher_kl': float(distill.detach())}
            trace.append(entry)
            print(json.dumps(entry), flush=True)
    candidate.eval()
    after = _evaluate(integrated, base, organism, state, pair, candidate, original, tokenizer, held_rows)
    for i in (6, 12):
        base.model.layers[i] = pair[i]
    general_metrics = []
    with torch.inference_mode():
        base.model.layers[18] = candidate
        for item in held_general:
            logits = answer_logits(integrated, base, organism, state, item)
            teacher = item['teacher'].to('cuda', dtype=torch.float32)
            kl = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(teacher, dim=-1), reduction='batchmean')
            general_metrics.append({'question': item['question'], 'teacher_kl': float(kl),
                                    'teacher_top_token': int(teacher.argmax()), 'candidate_top_token': int(logits.argmax())})
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT/'layer-18.pt'
    metadata = {'format': 'spora-qwen-layer-replacement-candidate-v1', 'base_weight_sha256': WEIGHT_SHA,
                'layer': 18, 'width': base.config.hidden_size, 'heads': base.config.num_attention_heads,
                'max_length': 128, 'seed': SEED, 'train_steps': STEPS,
                'objective': 'progressive_graft_first_token_kl_and_memory_ce'}
    torch.save({'metadata': metadata, 'state': {k: v.detach().cpu() for k, v in candidate.state_dict().items()}}, checkpoint)
    report = {'metadata': metadata, 'source_integrated_checkpoint_sha256': sha(SOURCE_CHECKPOINT),
              'upstream_pair_sha256': {str(i): sha(PAIR/f'layer-{i}.pt') for i in (6, 12)},
              'start_sha256': sha(START), 'candidate_sha256': sha(checkpoint),
              'train_memory_questions': len(train_rows), 'train_general_questions': list(GENERAL_TRAIN),
              'held_before': before, 'held_after': after, 'held_general_first_token': general_metrics,
              'trace': trace, 'elapsed_seconds': time.perf_counter()-started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Distills first output token only, not full continuations.',
                              'Held memory entities appeared in previous experiments.',
                              'Small fixed set of synthetic facts and general prompts.',
                              'Evaluator supplies memory address and temporal slot.']}
    with (OUTPUT/'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'before': before['correct'], 'after': after['correct'], 'total': after['total']}), flush=True)


if __name__ == '__main__':
    run()
