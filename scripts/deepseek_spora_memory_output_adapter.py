"""Train and audit a vocabulary output head without changing address features."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_composed_router_answer_eval import _load_candidate, _memory_args
from scripts.deepseek_spora_memory_answer_distill import SOURCE_CHECKPOINT
from scripts.deepseek_spora_recalibrate_name_router import SOURCE
from scripts.open_base_baseline import INDIVIDUAL_SHA
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import load, query
from spora.integrated_checkpoint import load_integrated
from spora.internal_memory import observe_sourced
from spora.language_training import atomic_save, frozen_digest
from spora.memory_output import MemoryOutputAdapter, MemoryOutputLogitsProcessor

OUTPUT = ROOT/'runs/deepseek-spora-memory-output-adapter-20261111'
STEPS = 160
SEED = 93011
TRAIN_VALUES = (1, 6)

PROMPTS = (
    '根据保存的偏好，{name}此刻愿意喝什么？',
    '请直接回答{name}当前最喜欢的饮料。',
    '记录显示{name}现在偏爱哪种饮品？',
    '只需说出{name}眼下爱喝的东西。',
)


def _make_state(organism, key, value):
    state = fresh_controlled_state(organism)
    state, decision = observe_sourced(organism, state, key, value, 'direct', 1.0)
    if not decision['current_write']:
        raise ValueError('Controlled memory write rejected')
    return state


def _memory_logits(organism, state, key):
    device = next(organism.parameters()).device
    keys = torch.tensor([key], dtype=torch.long, device=device)
    zeros = torch.zeros_like(keys)
    return query(organism, state, keys, zeros, zeros)


@torch.no_grad()
def _generate(integrated, tokenizer, organism, state, adapter, key, question, use_adapter):
    prefix = closed_think_inputs(tokenizer, question, 'cuda')
    args = _memory_args(organism, state, {'key': key, 'intent': 'current'})
    processors = []
    if use_adapter:
        processors = [MemoryOutputLogitsProcessor(adapter, _memory_logits(organism, state, key))]
    ids = integrated.generate(**prefix, **args, max_new_tokens=16, do_sample=False,
                              logits_processor=processors or None,
                              pad_token_id=tokenizer.pad_token_id,
                              eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(ids[0, prefix['input_ids'].shape[1]:], skip_special_tokens=True).strip()


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base mismatch')
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    integrated, organism, original = load_integrated(SOURCE_CHECKPOINT, base, MODEL, individual_device='cuda')
    original_digest = frozen_digest(organism, original)
    blocks = {i: _load_candidate(SOURCE/f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    for i, block in blocks.items():
        base.model.layers[i] = block
    adapter = MemoryOutputAdapter(organism.config.value_count, base.config.vocab_size, rank=32).to('cuda')
    adapter.train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.01, weight_decay=0.0001)
    rows = [(name, KEYS[i], value, question.format(name=name))
            for i, name in enumerate(NAMES[:2]) for value in TRAIN_VALUES
            for question in PROMPTS]
    losses = []
    for step in range(STEPS):
        name, key, value, question = rows[step % len(rows)]
        state = _make_state(organism, key, value)
        prefix = closed_think_inputs(tokenizer, question, 'cuda')
        args = _memory_args(organism, state, {'key': key, 'intent': 'current'})
        optimizer.zero_grad(set_to_none=True)
        output = integrated(**prefix, **args)
        target_ids = tokenizer(WORDS[value], add_special_tokens=False, return_tensors='pt').input_ids.to('cuda')
        target = target_ids[0, 0].expand(1)
        memory_logits = _memory_logits(organism, state, key)
        bias = adapter(memory_logits).float()
        logits = output.logits[:, -1, :].float() + bias
        loss = F.cross_entropy(logits, target)
        loss = loss + 0.0001*adapter.value_to_rank.weight.square().mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite output adapter loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step in (0, 9, 39, 79, 159, STEPS-1):
            losses.append({'step': step+1, 'loss': float(loss.detach())})
    adapter.eval()
    evaluation = []
    for i, name in enumerate(NAMES[:2]):
        key = KEYS[i]
        for value in TRAIN_VALUES:
            state = _make_state(organism, key, value)
            for prompt in PROMPTS:
                question = prompt.format(name=name)
                baseline = _generate(integrated, tokenizer, organism, state, adapter, key, question, False)
                adapted = _generate(integrated, tokenizer, organism, state, adapter, key, question, True)
                evaluation.append({'name': name, 'key': key, 'value': value, 'expected': WORDS[value],
                                   'question': question, 'baseline': baseline, 'adapted': adapted,
                                   'baseline_contains': WORDS[value] in baseline,
                                   'adapted_contains': WORDS[value] in adapted})
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA or frozen_digest(organism, original) != original_digest:
        raise ValueError('Protected input changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT/'adapter.pt'
    atomic_save({'format': 'spora-memory-output-adapter-v1', 'base_sha256': WEIGHT_SHA,
                 'individual_sha256': INDIVIDUAL_SHA, 'candidate_sha256': {
                     str(i): sha(SOURCE/f'layer-{i}.pt') for i in (6, 12)},
                 'config': adapter.config_dict(), 'state': adapter.state_dict()}, checkpoint)
    report = {'status': 'memory_output_adapter_development_only', 'base_sha256': WEIGHT_SHA,
              'individual_sha256': INDIVIDUAL_SHA, 'candidate_source': str(SOURCE),
              'candidate_sha256': {str(i): sha(SOURCE/f'layer-{i}.pt') for i in (6, 12)},
              'steps': STEPS, 'seed': SEED, 'train_values': list(TRAIN_VALUES), 'loss_trace': losses, 'evaluation': evaluation,
              'baseline_contains': sum(r['baseline_contains'] for r in evaluation),
              'adapted_contains': sum(r['adapted_contains'] for r in evaluation),
              'total': len(evaluation), 'adapter_sha256': sha(checkpoint),
              'elapsed_seconds': time.perf_counter()-started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Finite values and two bound entities; no open-world facts.',
                              'The adapter is trained on the same prompt family as this narrow probe.',
                              'It biases every generation step and is not yet integrated into durable runtime checkpoints.',
                              'String containment is not a semantic answer judge.',
                              'Address selection is supplied by the controlled experiment.']}
    with (OUTPUT/'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('baseline_contains', 'adapted_contains', 'total', 'adapter_sha256')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
