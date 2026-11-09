"""Train a frozen-base adapter on controlled SPORA-generated memory branches."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.open_base_baseline import MODEL, INDIVIDUAL, INDIVIDUAL_SHA, WEIGHT_SHA, REVISION, sha
from scripts.open_base_synthetic_alignment import choice_ids, encode_prompt
from spora.growing import load, query
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest
from spora.open_base_memory import OpenBaseMemory, memory_hooks
from spora.sleep_memory import history_site

TRAIN_PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7))
HELD_PAIRS = ((8, 9), (10, 11))
TRAIN_PROMPTS = ('对象{key}当前值是多少？只答1或2。', '读取对象{key}的数值，仅写1或2。',
                 '对象{key}记录的当前数字是1还是2？', '只回答数字，对象{key}现在是多少？')
HELD_PROMPTS = ('对象{key}目前记录的是哪个数字？只写1或2。', '从记忆中查询对象{key}，回答1或2。')
OUTPUT = ROOT/'runs/open-base-real-memory-alignment-20261109'


def native_read(model, state, keys):
    device = next(model.parameters()).device
    key_tensor = torch.tensor(keys, dtype=torch.long, device=device)
    zero = torch.zeros_like(key_tensor)
    with torch.no_grad():
        return query(model, state, key_tensor, zero, zero).argmax(-1).tolist()


def variants(model, original, pairs):
    """Write two opposite assignments to independent copies and verify both."""
    result, events = {}, []
    for pair in pairs:
        states = []
        for values in ((1, 2), (2, 1)):
            state = copy.deepcopy(original)
            for key, value in zip(pair, values):
                state, decision = observe_sourced(model, state, key, value, 'direct', 1.0)
                events.append({'pair': list(pair), 'key': key, 'value': value,
                               'action': decision['action'], 'current_write': decision['current_write']})
            actual = native_read(model, state, list(pair))
            if actual != list(values):
                raise ValueError(f'Native memory failed controlled assignment {pair}: {actual}')
            states.append(state)
        result[pair] = states
    return result, events


def pack_states(states, empty=False):
    if len(states) != 2 or len(states[0].pages) != len(states[1].pages):
        raise ValueError('Expected two states with equal page count')
    current = [torch.cat([state.current[site] for state in states], dim=0).detach() for site in range(2)]
    pages = [[[torch.cat([history_site(state.pages[p], site)[0] for state in states], dim=0).detach()]
              for site in range(2)] for p in range(len(states[0].pages))]
    if empty:
        current = [torch.zeros_like(x) for x in current]
        pages = [[[torch.zeros_like(x) for x in site] for site in page] for page in pages]
    return SimpleNamespace(current=current, pages=pages)


@torch.no_grad()
def evaluate(model, layers, adapter, tokenizer, choices, states, prompts, mode='normal', current_only=False):
    pairs = correct = both = 0
    margins = []
    for pair, variants_ in states.items():
        for position, key in enumerate(pair):
            for template in prompts:
                actual_states = list(reversed(variants_)) if mode == 'swapped' else variants_
                batch = pack_states(actual_states, empty=mode == 'empty')
                inputs = encode_prompt(tokenizer, template.format(key=key), 'cuda')
                inputs = {k: v.expand(2, -1) for k, v in inputs.items()}
                if mode == 'disabled':
                    logits = model(**inputs, use_cache=False).logits[:, -1, choices].float()
                else:
                    with memory_hooks(layers, adapter, batch, current_only=current_only):
                        logits = model(**inputs, use_cache=False).logits[:, -1, choices].float()
                targets = torch.tensor([position, 1-position] if mode != 'swapped' else [1-position, position], device='cuda')
                pred = logits.argmax(-1)
                correct += int((pred == targets).sum())
                both += int(bool((pred == targets).all()))
                pairs += 1
                margins.extend((logits.gather(1, targets[:, None]).squeeze(1) -
                                logits.gather(1, (1-targets)[:, None]).squeeze(1)).tolist())
    return {'pairs': pairs, 'both_correct': both, 'examples': pairs*2, 'correct': correct,
            'mean_correct_vs_other_logit': sum(margins)/len(margins)}


def run(output=OUTPUT, steps=400, address_weight=0., current_only=False):
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT/'runs').resolve()) or output == (ROOT/'runs').resolve():
        raise ValueError('Output must be a new directory under runs')
    if type(steps) is not int or not 1 <= steps <= 1000:
        raise ValueError('Steps must be between 1 and 1000')
    if not 0 <= address_weight <= 10 or type(current_only) is not bool:
        raise ValueError('Invalid address loss weight or reader mode')
    if output.exists():
        raise FileExistsError(output)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA or sha(INDIVIDUAL) != INDIVIDUAL_SHA:
        raise ValueError('Protected input mismatch')
    torch.manual_seed(23801)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(organism, original)
    states, events = variants(organism, original, TRAIN_PAIRS + HELD_PAIRS)
    if before != frozen_digest(organism, original):
        raise AssertionError('Original state changed during controlled writes')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    choices = choice_ids(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='eager').to('cuda').eval()
    model.requires_grad_(False)
    adapter = OpenBaseMemory(model.config.hidden_size, organism.config.memory_width).to('cuda')
    bases = [site.current_basis.detach().to('cuda') for site in organism.memories]
    layers = [model.model.layers[5], model.model.layers[17]]
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.002, weight_decay=0.)
    tasks = [(pair, position, template) for pair in TRAIN_PAIRS for position in range(2) for template in TRAIN_PROMPTS]
    log = []
    for step in range(steps):
        pair, position, template = tasks[step % len(tasks)]
        batch = pack_states(states[pair])
        inputs = encode_prompt(tokenizer, template.format(key=pair[position]), 'cuda')
        inputs = {k: v.expand(2, -1) for k, v in inputs.items()}
        targets = torch.tensor([choices[position], choices[1-position]], device='cuda')
        optimizer.zero_grad(set_to_none=True)
        captured, handles = {}, []
        try:
            if address_weight:
                for index, projection in enumerate(adapter.query):
                    def save_query(_module, _inputs, value, index=index):
                        captured[index] = value
                    handles.append(projection.register_forward_hook(save_query))
            with memory_hooks(layers, adapter, batch, current_only=current_only):
                logits = model(**inputs, use_cache=False).logits[:, -1, :].float()
                loss = F.cross_entropy(logits, targets)
                if address_weight:
                    if set(captured) != {0, 1}:
                        raise ValueError('Address projections were not both called')
                    key_labels = torch.full((2,), pair[position], dtype=torch.long, device='cuda')
                    address_loss = sum(F.cross_entropy(10*F.normalize(captured[i][:, -1, :], dim=-1) @ bases[i].T,
                                                       key_labels) for i in range(2))/2
                    loss = loss + address_weight*address_loss
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Nonfinite loss')
                loss.backward()
        finally:
            for handle in handles:
                handle.remove()
        grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step+1) % 50 == 0 or step+1 == steps:
            item = {'step': step+1, 'loss': float(loss.detach()), 'gradient_norm': float(grad_norm)}
            log.append(item)
            print(json.dumps(item), flush=True)
    adapter.eval()
    train = {pair: states[pair] for pair in TRAIN_PAIRS}
    held = {pair: states[pair] for pair in HELD_PAIRS}
    results = {'train': evaluate(model, layers, adapter, tokenizer, choices, train, TRAIN_PROMPTS, current_only=current_only),
               'new_keys_new_prompts': evaluate(model, layers, adapter, tokenizer, choices, held, HELD_PROMPTS, current_only=current_only),
               'swapped': evaluate(model, layers, adapter, tokenizer, choices, held, HELD_PROMPTS, 'swapped', current_only),
               'empty': evaluate(model, layers, adapter, tokenizer, choices, held, HELD_PROMPTS, 'empty', current_only),
               'disabled': evaluate(model, layers, adapter, tokenizer, choices, held, HELD_PROMPTS, 'disabled', current_only)}
    if (any(p.grad is not None for p in model.parameters()) or frozen_digest(organism, original) != before
            or sha(INDIVIDUAL) != INDIVIDUAL_SHA or sha(MODEL/'model.safetensors') != WEIGHT_SHA):
        raise ValueError('Protected input changed')
    output.mkdir(parents=False, exist_ok=False)
    snapshot = output/'adapter.pt'
    torch.save({'format': 'spora-open-base-real-memory-v1',
                'state': {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA, 'layers': [5, 17],
                'current_only': current_only}, snapshot)
    report = {'status': 'controlled_written_memory_not_general_semantic_alignment', 'revision': REVISION,
              'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA,
              'individual_frozen_digest': before, 'adapter_sha256': sha(snapshot), 'layers': [5, 17],
              'steps': steps, 'address_weight': address_weight, 'current_only': current_only,
              'train_pairs': TRAIN_PAIRS, 'held_pairs': HELD_PAIRS,
              'train_prompts': TRAIN_PROMPTS, 'held_prompts': HELD_PROMPTS, 'events': events,
              'log': log, 'results': results, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'seconds': time.perf_counter()-start,
              'limitations': ['Controlled events were written only to in-memory copies of the original state.',
                              'All cases use the same original background and exactly two new events.',
                              'Only values 1 and 2 are used; ranking two tokens is not free generation.',
                              'Historical time, unknown answers and unrelated real memories are not tested.']}
    with (output/'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report['results'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--address-weight', type=float, default=0.)
    parser.add_argument('--current-only', action='store_true')
    args = parser.parse_args()
    run(args.output, args.steps, args.address_weight, args.current_only)
