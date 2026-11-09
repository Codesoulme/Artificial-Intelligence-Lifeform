"""Controlled DeepSeek/SPORA current-memory alignment, with an explicitly closed think section.

The prefix is a diagnostic ablation, not evidence of reasoning or general dialogue.
Only the adapter is trained; the original individual is read through cloned states.
"""
import argparse
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

from scripts.deepseek_local_baseline import MODEL, REVISION, WEIGHT_SHA
from scripts.open_base_addressed_alignment import QUERY_PATH, QUERY_SHA, neural_addresses, TRAIN_PROMPTS, HELD_PROMPTS
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.open_base_real_memory_alignment import TRAIN_PAIRS, HELD_PAIRS, variants, pack_states
from scripts.open_base_synthetic_alignment import choice_ids
from spora.growing import load
from spora.language_training import frozen_digest
from spora.open_base_memory import OpenBaseMemory, memory_hooks


def closed_think_inputs(tokenizer, question, device):
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': question}], tokenize=False,
                                            add_generation_prompt=True)
    if '<think>' not in prompt or '</think>' in prompt:
        raise ValueError('Unexpected DeepSeek generation template')
    return tokenizer(prompt + '</think>\n\n', return_tensors='pt', add_special_tokens=False).to(device)


@torch.no_grad()
def evaluate(model, layers, adapter, tokenizer, choices, reader, organism, states, prompts, mode='normal'):
    rows = []
    for pair, state_pair in states.items():
        for pos, key in enumerate(pair):
            for template in prompts:
                question = template.format(key=key)
                addresses, parsed = neural_addresses(reader, organism, question)
                if parsed['entity'] != key:
                    raise ValueError('Wrong neural address')
                chosen = list(reversed(state_pair)) if mode == 'swapped' else state_pair
                batch = pack_states(chosen, empty=mode == 'empty')
                inputs = closed_think_inputs(tokenizer, question, 'cuda')
                inputs = {k: v.expand(2, -1) for k, v in inputs.items()}
                if mode == 'disabled':
                    logits = model(**inputs, use_cache=False).logits[:, -1, choices].float()
                else:
                    with memory_hooks(layers, adapter, batch, current_only=True, addresses=addresses):
                        logits = model(**inputs, use_cache=False).logits[:, -1, choices].float()
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError('Nonfinite evaluation logits')
                targets = [pos, 1-pos] if mode != 'swapped' else [1-pos, pos]
                predictions = logits.argmax(-1).tolist()
                rows.append({'pair': list(pair), 'key': key, 'question': question, 'expected': targets,
                             'predicted': predictions, 'logits': logits.tolist(),
                             'both_correct': predictions == targets})
    return {'pairs': len(rows), 'both_correct': sum(r['both_correct'] for r in rows), 'rows': rows}


@torch.no_grad()
def free_generate(model, layers, adapter, tokenizer, reader, organism, states):
    rows = []
    for pair, state_pair in states.items():
        for pos, key in enumerate(pair):
            question = HELD_PROMPTS[0].format(key=key)
            addresses, parsed = neural_addresses(reader, organism, question)
            if parsed['entity'] != key:
                raise ValueError('Wrong neural address')
            inputs = closed_think_inputs(tokenizer, question, 'cuda')
            inputs = {k: v.expand(2, -1) for k, v in inputs.items()}
            with memory_hooks(layers, adapter, pack_states(state_pair), current_only=True, addresses=addresses):
                output = model.generate(**inputs, max_new_tokens=6, do_sample=False,
                                        pad_token_id=tokenizer.eos_token_id)
            responses = [tokenizer.decode(row[inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                         for row in output]
            targets = ['1', '2'] if pos == 0 else ['2', '1']
            rows.append({'pair': list(pair), 'key': key, 'question': question, 'expected': targets,
                         'responses': responses, 'both_exact': responses == targets})
    return rows


def run(output, steps):
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT / 'runs').resolve()) or output == (ROOT / 'runs').resolve() or output.exists():
        raise ValueError('Output must be a new directory under runs')
    if type(steps) is not int or not 1 <= steps <= 2000:
        raise ValueError('Invalid step count')
    def verify():
        return (sha(MODEL / 'model.safetensors') == WEIGHT_SHA and sha(INDIVIDUAL) == INDIVIDUAL_SHA
                and sha(QUERY_PATH) == QUERY_SHA)
    if not verify():
        raise ValueError('Protected source mismatch')
    torch.manual_seed(4603)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    reader, reader_state, _ = load(QUERY_PATH, 'cpu')
    if not hasattr(reader, 'chinese_query') or any(not torch.equal(a.current_basis, b.current_basis)
            for a, b in zip(organism.memories, reader.memories)):
        raise ValueError('Incompatible neural addresses')
    before, reader_before = frozen_digest(organism, original), frozen_digest(reader, reader_state)
    states, events = variants(organism, original, TRAIN_PAIRS + HELD_PAIRS)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    choices = choice_ids(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    adapter = OpenBaseMemory(model.config.hidden_size, organism.config.memory_width).to('cuda')
    layers = [model.model.layers[5], model.model.layers[17]]
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.002, weight_decay=0.)
    tasks = [(pair, position, template) for pair in TRAIN_PAIRS for position in range(2) for template in TRAIN_PROMPTS]
    log = []
    for step in range(steps):
        pair, position, template = tasks[step % len(tasks)]
        question = template.format(key=pair[position])
        addresses, parsed = neural_addresses(reader, organism, question)
        if parsed['entity'] != pair[position]:
            raise ValueError('Wrong training address')
        inputs = closed_think_inputs(tokenizer, question, 'cuda')
        inputs = {k: v.expand(2, -1) for k, v in inputs.items()}
        targets = torch.tensor([choices[position], choices[1-position]], device='cuda')
        optimizer.zero_grad(set_to_none=True)
        with memory_hooks(layers, adapter, pack_states(states[pair]), current_only=True, addresses=addresses):
            logits = model(**inputs, use_cache=False).logits[:, -1, :].float()
            loss = F.cross_entropy(logits, targets)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == steps:
            entry = {'step': step + 1, 'loss': float(loss.detach()), 'gradient_norm': float(norm)}
            log.append(entry)
            print(json.dumps(entry), flush=True)
    adapter.eval()
    held = {pair: states[pair] for pair in HELD_PAIRS}
    options = (model, layers, adapter, tokenizer, choices, reader, organism, held, HELD_PROMPTS)
    results = {mode: evaluate(*options, mode) for mode in ('normal', 'swapped', 'empty', 'disabled')}
    generation = free_generate(model, layers, adapter, tokenizer, reader, organism, held)
    if (any(p.grad is not None for p in model.parameters()) or frozen_digest(organism, original) != before
            or frozen_digest(reader, reader_state) != reader_before or not verify()):
        raise ValueError('Protected inputs changed')
    output.mkdir(parents=False, exist_ok=False)
    snapshot = output / 'adapter.pt'
    torch.save({'format': 'deepseek-spora-closed-think-read-v1',
                'state': {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA,
                'query_sha256': QUERY_SHA, 'layers': [5, 17], 'current_only': True}, snapshot)
    report = {'status': 'controlled_next_token_diagnostic_only', 'revision': REVISION,
              'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA, 'query_sha256': QUERY_SHA,
              'adapter_sha256': sha(snapshot), 'steps': steps, 'events': events, 'log': log,
              'results': results, 'free_generation': generation, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['The think section is explicitly closed before the answer; no reasoning is tested.',
                              'P26 is a separately trained finite numbered-question reader.',
                              'All states share one original background and two controlled writes.',
                              'Only 1 and 2 are trained; candidate ranking is not unrestricted generation.',
                              'P26 may have seen the held keys and prompts.',
                              'Swapped conditions reuse the same two states in reverse order.']}
    with (output / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({mode: {k: v for k, v in result.items() if k != 'rows'}
                      for mode, result in results.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.steps)
