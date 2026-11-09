"""Pilot continued readout alignment for names with controlled numeric memory."""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_chat_interference import SELECTOR
from scripts.deepseek_full_answer import answer_batch
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_named_binding_probe import KEYS, NAMES, TRAIN_TEMPLATES, UNKNOWN
from scripts.deepseek_named_composite_probe import TEMPLATES as HELD_TEMPLATES
from scripts.deepseek_named_incremental_probe import NEW_KEYS
from scripts.deepseek_read_strength_probe import ScaledReadout
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.open_base_trajectory_comparison import numeric_correct
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import load, query
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest
from spora.open_base_memory import memory_hooks

OUTPUT = ROOT / 'runs/deepseek-named-readout-pilot-20261110'
TRAIN_VALUES = (0, 2, 4, 6, 8, 10, 12, 14)
HELD_VALUES = (5, 13)
STEPS = 80


def snapshot(organism, key, value):
    state = fresh_controlled_state(organism)
    for event_key, event_value in ((63, (value + 3) % 16), (key, value)):
        state, decision = observe_sourced(organism, state, event_key, event_value, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('Controlled observation not written')
    keys = torch.tensor([key, 63], dtype=torch.long)
    zero = torch.zeros_like(keys)
    with torch.no_grad():
        actual = query(organism, state, keys, zero, zero).argmax(-1).tolist()
    if actual != [value, (value + 3) % 16]:
        raise ValueError('Native snapshot values mismatch')
    return state


def combined_states(states):
    return SimpleNamespace(current=[torch.cat([state.current[site] for state in states], dim=0).detach()
                                    for site in range(2)], pages=[])


@torch.no_grad()
def evaluate(base, tokenizer, adapter, bases, states, names, keys):
    rows = []
    for group in ('named', 'numeric'):
        for name, key in zip(names, keys):
            templates = HELD_TEMPLATES if group == 'named' else ('对象{key}的值是多少？',)
            for template in templates:
                question = template.format(name=name, key=key)
                addresses = [basis[key:key+1] for basis in bases]
                inputs = closed_think_inputs(tokenizer, question, 'cuda')
                batch = {k: v.expand(len(HELD_VALUES), -1) for k, v in inputs.items()}
                memory = combined_states([states[(key, value)] for value in HELD_VALUES])
                with memory_hooks([base.model.layers[5], base.model.layers[17]], adapter, memory,
                                  current_only=True, addresses=addresses):
                    generated = base.generate(**batch, max_new_tokens=8, do_sample=False,
                                              pad_token_id=tokenizer.eos_token_id)
                outputs = []
                for item, value in zip(generated, HELD_VALUES):
                    tokens = item[inputs['input_ids'].shape[1]:].tolist()
                    text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    outputs.append({'expected': value, 'text': text, 'correct': numeric_correct(text, value),
                                    'ended_with_eos': tokenizer.eos_token_id in tokens})
                rows.append({'group': group, 'name': name, 'key': key, 'question': question, 'outputs': outputs})
    summary = {}
    for group in ('named', 'numeric'):
        selected = [r for r in rows if r['group'] == group]
        summary[group] = {'pairs': len(selected), 'examples': sum(len(r['outputs']) for r in selected),
                          'correct': sum(item['correct'] for r in selected for item in r['outputs']),
                          'both_correct': sum(all(item['correct'] for item in r['outputs']) for r in selected)}
    return {'summary': summary, 'rows': rows}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    protected = {p: sha(p) for p in (MODEL / 'model.safetensors', INDIVIDUAL, SELECTOR)}
    if protected[MODEL / 'model.safetensors'] != WEIGHT_SHA or protected[INDIVIDUAL] != INDIVIDUAL_SHA:
        raise ValueError('Protected model mismatch')
    parent = torch.load(SELECTOR, map_location='cpu', weights_only=True)
    if (parent.get('format') != 'deepseek-spora-question-pooled-finite-address-v1'
            or parent.get('weight_sha256') != WEIGHT_SHA or parent.get('individual_sha256') != INDIVIDUAL_SHA
            or parent.get('layers') != [5, 17]):
        raise ValueError('Readout provenance mismatch')
    torch.manual_seed(4633)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(organism, original)
    eval_names, eval_keys = (NAMES[0], NAMES[2], UNKNOWN[0], UNKNOWN[1]), (KEYS[0], KEYS[2], NEW_KEYS[0], NEW_KEYS[1])
    if set(NAMES) & set(UNKNOWN[:2]) or set(TRAIN_VALUES) & set(HELD_VALUES):
        raise ValueError('Train/evaluation split mismatch')
    keys = set(KEYS + eval_keys)
    states = {(key, value): snapshot(organism, key, value) for key in sorted(keys)
              for value in TRAIN_VALUES + HELD_VALUES}
    state_digests = {key: frozen_digest(organism, state) for key, state in states.items()}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    adapter = ScaledReadout(base.config.hidden_size, organism.config.memory_width, 0.5).to('cuda')
    adapter.load_state_dict(parent['readout'], strict=True)
    bases = [F.normalize(site.current_basis.to('cuda').float(), dim=-1) for site in organism.memories]
    print('baseline evaluation starting', flush=True)
    adapter.eval()
    baseline = evaluate(base, tokenizer, adapter, bases, states, eval_names, eval_keys)
    adapter.train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.0002, weight_decay=0.)
    tasks = [(name, key, template, value) for name, key in zip(NAMES, KEYS)
             for template in TRAIN_TEMPLATES for value in TRAIN_VALUES]
    generator = torch.Generator().manual_seed(4633)
    order = torch.randperm(len(tasks), generator=generator).tolist()
    log = []
    for step in range(STEPS):
        name, key, template, value = tasks[order[step]]
        other = (value + 2) % 16
        question = template.format(name=name)
        inputs, labels = answer_batch(tokenizer, question, [value, other], 'cuda')
        memory = combined_states([states[(key, value)], states[(key, other)]])
        addresses = [basis[key:key+1] for basis in bases]
        optimizer.zero_grad(set_to_none=True)
        with memory_hooks([base.model.layers[5], base.model.layers[17]], adapter, memory,
                          current_only=True, addresses=addresses):
            logits = base(**inputs, use_cache=False).logits[:, :-1, :].float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 20 == 0:
            entry = {'step': step + 1, 'loss': float(loss.detach()), 'gradient_norm': float(norm)}
            log.append(entry)
            print(json.dumps(entry), flush=True)
    adapter.eval()
    if (any(p.grad is not None for p in base.parameters()) or frozen_digest(organism, original) != before
            or any(frozen_digest(organism, state) != state_digests[key] for key, state in states.items())
            or any(sha(p) != d for p, d in protected.items())):
        raise ValueError('Protected input changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT / 'adapter.pt'
    torch.save({'format': 'deepseek-spora-named-readout-development-v1',
                'state': {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA,
                'parent_sha256': protected[SELECTOR], 'layers': [5, 17], 'strength': 0.5}, checkpoint)
    print('adapted checkpoint saved; evaluation starting', flush=True)
    adapted = evaluate(base, tokenizer, adapter, bases, states, eval_names, eval_keys)
    if (frozen_digest(organism, original) != before
            or any(frozen_digest(organism, state) != state_digests[key] for key, state in states.items())
            or any(sha(p) != d for p, d in protected.items())):
        raise ValueError('Protected input changed during evaluation')
    report = {'status': 'named_readout_alignment_development_only', 'weight_sha256': WEIGHT_SHA,
              'individual_sha256': INDIVIDUAL_SHA, 'parent_sha256': protected[SELECTOR],
              'adapter_sha256': sha(checkpoint), 'steps': STEPS, 'learning_rate': 0.0002,
              'training_names': list(NAMES), 'held_names': list(UNKNOWN[:2]),
              'training_values': TRAIN_VALUES, 'held_values': HELD_VALUES, 'log': log,
              'baseline': baseline, 'adapted': adapted, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Frozen base and original individual; only the readout adapter is updated.',
                              'Correct addresses supplied explicitly; locator, gate and unknown handling not evaluated.',
                              'Held names and templates occurred in earlier development; not independent final evaluation.',
                              'All finite values appeared in earlier adapter training, although held values were excluded from this pilot.',
                              'Snapshots contain two controlled direct events and share a model and basis.',
                              'Closed thinking, greedy output and eight-token cap; not free conversation or reasoning.',
                              'No ordinary-chat preservation, open entities, history or crash recovery tested.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'baseline': baseline['summary'], 'adapted': adapted['summary']}), flush=True)


if __name__ == '__main__':
    run()
