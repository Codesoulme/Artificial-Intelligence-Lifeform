"""Jointly refine two replacement layers with language and memory replay."""
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
from transformers import AutoTokenizer
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_layer18_broad_pilot import trainer as broad
from scripts.deepseek_spora_layer18_general_long_audit import QUESTIONS
from scripts.deepseek_spora_memory_answer_distill import SOURCE_CHECKPOINT, _rows, _memory_args, _load_candidate, _evaluate
from scripts.deepseek_spora_memory_answer_distill_v4 import TRAIN_INDICES, VALUE_PAIRS
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.integrated_checkpoint import load_integrated
from spora.language_training import atomic_save, frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-pair-general-replay-20261112'
REFERENCE = ROOT / 'runs/deepseek-spora-layer18-broad-pilot-20261112/result.json'
PAIR = broad.PAIR
_facts = broad._facts
STEPS = 120
SEED = 88041
GENERATE = 32


def generate(base, tokenizer, questions):
    rows = []
    with torch.inference_mode():
        for question in questions:
            prefix = closed_think_inputs(tokenizer, question, 'cuda')
            ids = base.generate(**prefix, max_new_tokens=GENERATE, do_sample=False, use_cache=False,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            tail = ids[0, prefix['input_ids'].shape[1]:]
            rows.append({'question': question, 'answer': tokenizer.decode(tail, skip_special_tokens=True).strip()})
    return rows


def exact_memory(evaluation):
    return sum(row['answer'] in (row['expected'], row['expected'] + '.', row['expected'] + '。')
               for row in evaluation['rows'])


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if set(broad.TRAIN_GENERAL) & set(QUESTIONS):
        raise ValueError('General training and development question overlap')
    reference = json.loads(REFERENCE.read_text(encoding='utf-8'))
    for i in (6, 12):
        if sha(PAIR / f'layer-{i}.pt') != reference['upstream_pair_sha256'][str(i)]:
            raise ValueError('Upstream pair digest mismatch')
    source_sha = sha(SOURCE_CHECKPOINT)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed().eval()
    base.requires_grad_(False)
    integrated, organism, original_state = load_integrated(SOURCE_CHECKPOINT, base, MODEL, individual_device='cuda')
    integrated.reader.requires_grad_(False)
    original_digest = frozen_digest(organism, original_state)
    state, facts = _facts(organism, fresh_controlled_state(organism), VALUE_PAIRS, TRAIN_INDICES)
    train_rows, held_rows = _rows(facts, tokenizer)
    original_layers = {i: base.model.layers[i] for i in (6, 12)}
    replacements = {i: _load_candidate(PAIR / f'layer-{i}.pt', i, base) for i in (6, 12)}
    raw_general = generate(base, tokenizer, QUESTIONS)
    general = []
    with torch.inference_mode():
        for question in broad.TRAIN_GENERAL:
            prefix = closed_think_inputs(tokenizer, question, 'cuda')
            ids = base.generate(**prefix, max_new_tokens=16, do_sample=False, use_cache=False,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            count = ids.shape[1] - prefix['input_ids'].shape[1]
            if count < 1:
                raise ValueError('Empty teacher continuation')
            logits = base(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
            start = prefix['input_ids'].shape[1] - 1
            general.append({'ids': ids, 'prefix_length': prefix['input_ids'].shape[1], 'length': count,
                            'teacher': logits[:, start:start + count, :].detach().half().cpu()})
    for item in general:
        item['ids'] = item['ids'].clone()
    memory = [{'row': row, 'ids': torch.cat((row['prefix']['input_ids'], row['answer_ids']), dim=1),
               'prefix_length': row['prefix']['input_ids'].shape[1], 'length': row['answer_ids'].shape[1]}
              for row in train_rows]
    memory_before = _evaluate(integrated, base, organism, state, replacements, original_layers, tokenizer, held_rows)
    for i in (6, 12):
        base.model.layers[i] = replacements[i].eval()
    before_general = generate(base, tokenizer, QUESTIONS)
    for layer in original_layers.values():
        layer.to('cpu')
    params = [p for block in replacements.values() for p in block.parameters()]
    optimizer = torch.optim.AdamW(params, lr=1e-6, weight_decay=0.01, foreach=False)
    rng = random.Random(SEED)
    trace = []
    for step in range(STEPS):
        is_general = step % 3 == 0
        item = rng.choice(general if is_general else memory)
        ids = item['ids']
        args = {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}
        logits = (base(**args, use_cache=False).logits if is_general else
                  integrated(**args, **_memory_args(organism, state, item['row'])).logits)
        start = item['prefix_length'] - 1
        values = logits[:, start:start + item['length'], :].float()
        if is_general:
            teacher = item['teacher'].to('cuda', dtype=torch.float32)
            loss = F.kl_div(F.log_softmax(values / 2., dim=-1), F.softmax(teacher / 2., dim=-1),
                            reduction='batchmean') * 4. / item['length']
        else:
            target = ids[:, item['prefix_length']:item['prefix_length'] + item['length']]
            loss = F.cross_entropy(values.reshape(-1, values.shape[-1]), target.reshape(-1))
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if step in (0, 19, 39, 79, 119):
            trace.append({'step': step + 1, 'kind': 'general' if is_general else 'memory', 'loss': float(loss.detach())})
            print(json.dumps(trace[-1]), flush=True)
    for block in replacements.values():
        block.eval()
    memory_after = _evaluate(integrated, base, organism, state, replacements, original_layers, tokenizer, held_rows)
    for i in (6, 12):
        base.model.layers[i] = replacements[i]
    after_general = generate(base, tokenizer, QUESTIONS)
    if sha(SOURCE_CHECKPOINT) != source_sha or frozen_digest(organism, original_state) != original_digest:
        raise ValueError('Protected individual changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checks = {}
    for i, block in replacements.items():
        path = OUTPUT / f'layer-{i}.pt'
        metadata = {'format': 'spora-qwen-layer-replacement-candidate-v1', 'base_weight_sha256': WEIGHT_SHA,
                    'layer': i, 'width': base.config.hidden_size, 'heads': base.config.num_attention_heads,
                    'max_length': block.max_length, 'seed': SEED, 'train_steps': STEPS,
                    'objective': 'paired_memory_ce_and_general_raw_teacher_kl'}
        atomic_save({'metadata': metadata, 'state': {k: v.detach().cpu() for k, v in block.state_dict().items()}}, path)
        checks[str(i)] = sha(path)
    summary = {'held_memory_before_exact': exact_memory(memory_before), 'held_memory_after_exact': exact_memory(memory_after),
               'held_memory_total': memory_after['total'],
               'general_before_matches_raw': sum(a['answer'] == b['answer'] for a, b in zip(before_general, raw_general)),
               'general_after_matches_raw': sum(a['answer'] == b['answer'] for a, b in zip(after_general, raw_general)),
               'general_changed': sum(a['answer'] != b['answer'] for a, b in zip(before_general, after_general))}
    report = {'status': 'pair_general_memory_replay_pilot', 'summary': summary, 'trace': trace,
              'raw_general': raw_general, 'before_general': before_general, 'after_general': after_general,
              'memory_before': memory_before, 'memory_after': memory_after, 'candidate_sha256': checks,
              'source_pair_sha256': reference['upstream_pair_sha256'], 'source_checkpoint_sha256': source_sha,
              'base_sha256': WEIGHT_SHA, 'steps': STEPS, 'seed': SEED,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Authored general prompts and previously used memory questions, not independent holdout.',
                              'Raw teacher can be incorrect; exact output matching does not prove quality.',
                              'General targets capped at 16 tokens; held generation capped at 32.',
                              'Memory facts and addresses supplied programmatically; no natural language writing.',
                              'Only temporary individual; no default model or real individual modified.',
                              'Full research streamed loader; baseline parity assessed only on limited samples.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
