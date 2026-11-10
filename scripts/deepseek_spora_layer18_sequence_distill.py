"""Distill answer-token sequences into the full SPORA layer-18 replacement."""
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

START = ROOT / 'runs/deepseek-spora-layer18-curriculum-distill-20261111/layer-18.pt'
OUTPUT = ROOT / 'runs/deepseek-spora-layer18-sequence-distill-20261111'
TRAIN_GENERAL = ('简短介绍春天。', '如何理解诚信？', '一加一等于几？', '描述一只小猫。',
                 '朋友难过时怎么安慰？', '写一句问候语。', '太阳为什么会发光？', '介绍一次愉快的散步。')
HELD_GENERAL = ('给朋友说一句鼓励的话。', '请简单介绍雨的形成。', '什么是公正？')
STEPS = 120
SEED = 37162


def model_logits(integrated, base, organism, state, item):
    args = {'input_ids': item['ids'], 'attention_mask': torch.ones_like(item['ids'])}
    if item['kind'] == 'memory':
        output = integrated(**args, **_memory_args(organism, state, item['row']))
    else:
        output = base(**args, use_cache=False)
    start = item['prefix_length'] - 1
    return output.logits[:, start:start + item['target_length'], :].float()


def generate_general(base, tokenizer, questions):
    rows = []
    with torch.inference_mode():
        for question in questions:
            inputs = closed_think_inputs(tokenizer, question, 'cuda')
            ids = base.generate(**inputs, max_new_tokens=12, do_sample=False, use_cache=False,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            tail = ids[:, inputs['input_ids'].shape[1]:]
            rows.append({'question': question, 'answer': tokenizer.decode(tail[0], skip_special_tokens=True).strip()})
    return rows


def short_exact(result):
    return sum(row['answer'] in (row['expected'], row['expected'] + '.', row['expected'] + '。')
               for row in result['rows'])


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base model hash mismatch')
    previous = json.loads((START.parent/'result.json').read_text(encoding='utf-8'))
    if sha(START) != previous['candidate_sha256'] or any(
        sha(PAIR/f'layer-{i}.pt') != previous['upstream_pair_sha256'][str(i)] for i in (6, 12)
    ):
        raise ValueError('Candidate and upstream pair provenance mismatch')
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
    state, facts = _facts(organism, fresh_controlled_state(organism), VALUE_PAIRS, TRAIN_INDICES)
    train_rows, held_rows = _rows(facts, tokenizer)
    originals = {i: base.model.layers[i] for i in (6, 12, 18)}
    pair = {i: _load_candidate(PAIR/f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    candidate = _load_candidate(START, 18, base)
    for i in (6, 12):
        base.model.layers[i] = pair[i]
    items = []
    for row in train_rows:
        prefix, answer = row['prefix']['input_ids'], row['answer_ids']
        items.append({'kind': 'memory', 'row': row, 'ids': torch.cat((prefix, answer), dim=1),
                      'prefix_length': prefix.shape[1], 'target_length': answer.shape[1]})
    with torch.inference_mode():
        teacher_general = generate_general(base, tokenizer, HELD_GENERAL)
        for question in TRAIN_GENERAL:
            prefix = closed_think_inputs(tokenizer, question, 'cuda')
            ids = base.generate(**prefix, max_new_tokens=10, do_sample=False, use_cache=False,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            length = ids.shape[1] - prefix['input_ids'].shape[1]
            if not length:
                raise ValueError('Teacher generated an empty sequence')
            items.append({'kind': 'general', 'ids': ids, 'question': question,
                          'prefix_length': prefix['input_ids'].shape[1], 'target_length': length})
        for index, item in enumerate(items):
            item['teacher'] = model_logits(integrated, base, organism, state, item).detach().half().cpu()
            if (index + 1) % 12 == 0:
                print(json.dumps({'teacher_cached': index+1, 'total': len(items)}), flush=True)
    # Generated token IDs created under inference_mode cannot be saved by
    # autograd when cross entropy backpropagates through the student logits.
    for item in items:
        item['ids'] = item['ids'].clone()
    before = _evaluate(integrated, base, organism, state, pair, candidate, originals, tokenizer, held_rows)
    for i in (6, 12):
        base.model.layers[i] = pair[i]
    base.model.layers[18] = candidate.eval()
    general_before = generate_general(base, tokenizer, HELD_GENERAL)
    candidate.train()
    memory = [item for item in items if item['kind'] == 'memory']
    general = [item for item in items if item['kind'] == 'general']
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=1e-6, weight_decay=0.01, foreach=False)
    rng, trace = random.Random(SEED), []
    for step in range(STEPS):
        item = rng.choice(general if step % 3 == 0 else memory)
        logits = model_logits(integrated, base, organism, state, item)
        teacher = item['teacher'].to('cuda', dtype=torch.float32)
        temperature = 2.0
        kl = F.kl_div(F.log_softmax(logits/temperature, dim=-1),
                      F.softmax(teacher/temperature, dim=-1), reduction='batchmean')
        kl = kl * temperature**2 / item['target_length']
        labels = item['ids'][:, item['prefix_length']:item['prefix_length'] + item['target_length']]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        loss = kl + (ce if item['kind'] == 'memory' else 0.25*ce)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite sequence distillation loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
        optimizer.step()
        if step in (0, 19, 39, 79, 119):
            trace.append({'step': step+1, 'kind': item['kind'], 'target_tokens': item['target_length'],
                          'loss': float(loss.detach()), 'kl_per_token': float(kl.detach()),
                          'cross_entropy': float(ce.detach())})
            print(json.dumps(trace[-1]), flush=True)
    candidate.eval()
    general_after = generate_general(base, tokenizer, HELD_GENERAL)
    after = _evaluate(integrated, base, organism, state, pair, candidate, originals, tokenizer, held_rows)
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT/'layer-18.pt'
    metadata = {'format': 'spora-qwen-layer-replacement-candidate-v1', 'base_weight_sha256': WEIGHT_SHA,
                'layer': 18, 'width': base.config.hidden_size, 'heads': base.config.num_attention_heads,
                'max_length': 128, 'seed': SEED, 'train_steps': STEPS,
                'objective': 'full_replacement_multi_token_kl_and_ce'}
    torch.save({'metadata': metadata, 'state': {k: v.detach().cpu() for k, v in candidate.state_dict().items()}}, checkpoint)
    report = {'metadata': metadata, 'source_integrated_checkpoint_sha256': sha(SOURCE_CHECKPOINT),
              'upstream_pair_sha256': {str(i): sha(PAIR/f'layer-{i}.pt') for i in (6, 12)},
              'start_sha256': sha(START), 'candidate_sha256': sha(checkpoint),
              'training_memory_questions': len(memory), 'training_general_questions': list(TRAIN_GENERAL),
              'general_train_target_tokens': sum(x['target_length'] for x in general),
              'memory_train_target_tokens': sum(x['target_length'] for x in memory),
              'memory_before': before, 'memory_after': after,
              'exact_before': short_exact(before), 'exact_after': short_exact(after),
              'general_teacher': teacher_general, 'general_before': general_before, 'general_after': general_after,
              'trace': trace, 'elapsed_seconds': time.perf_counter()-started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['General teacher continuations limited to ten tokens, held to twelve.',
                              'Only eight generic training questions and three held general questions.',
                              'Held memory entities used in previous development; one finite beverage domain.',
                              'Memory address and time slot supplied by the evaluator.']}
    with (OUTPUT/'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'before_exact': report['exact_before'], 'after_exact': report['exact_after'],
                      'total': after['total']}), flush=True)


if __name__ == '__main__':
    run()
