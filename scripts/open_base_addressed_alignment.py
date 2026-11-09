"""Use the P26 neural question reader to address controlled SPORA memory."""
import argparse
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

from scripts.open_base_baseline import MODEL, INDIVIDUAL, INDIVIDUAL_SHA, WEIGHT_SHA, REVISION, sha
from scripts.open_base_real_memory_alignment import TRAIN_PAIRS, HELD_PAIRS, variants, pack_states
from scripts.open_base_synthetic_alignment import choice_ids, encode_prompt
from spora.growing import load
from spora.language_training import frozen_digest
from spora.open_base_memory import OpenBaseMemory, memory_hooks

QUERY_PATH = ROOT/'runs/p26-memory-acceptance/individual-query.pt'
QUERY_SHA = '61732fea74cd38761b8247561436fc73d7b6c7047be5afd4989598954eeee3d9'
OUTPUT = ROOT/'runs/open-base-addressed-memory-20261109'
TRAIN_PROMPTS = ('对象{key}现在的值是多少？', '对象{key}的当前值是什么？',
                 '编号{key}现在的值是多少？', '对象{key}当前的值是？')
HELD_PROMPTS = ('对象{key}的值是多少？', '请问对象{key}目前的值是多少？')


def neural_addresses(query_model, memory_model, question):
    parsed = query_model.chinese_query.parse([question])[0]
    if not parsed['accepted'] or parsed['kind'] != 'current':
        raise ValueError('Neural query rejected or was not a current-value question')
    key = parsed['entity']
    return [site.current_basis[key:key+1].detach() for site in memory_model.memories], parsed


@torch.no_grad()
def evaluate(model, layers, adapter, tokenizer, choices, query_model, memory_model, states, prompts, mode='normal'):
    pairs = correct = both = 0
    for pair, versions in states.items():
        for position, key in enumerate(pair):
            for template in prompts:
                question = template.format(key=key)
                addresses, parsed = neural_addresses(query_model, memory_model, question)
                if parsed['entity'] != key:
                    raise ValueError('Neural address differs from evaluation key')
                chosen = list(reversed(versions)) if mode == 'swapped' else versions
                batch = pack_states(chosen, empty=mode == 'empty')
                inputs = encode_prompt(tokenizer, question, 'cuda')
                inputs = {k: v.expand(2, -1) for k, v in inputs.items()}
                if mode == 'disabled':
                    logits = model(**inputs, use_cache=False).logits[:, -1, choices].float()
                else:
                    with memory_hooks(layers, adapter, batch, current_only=True, addresses=addresses):
                        logits = model(**inputs, use_cache=False).logits[:, -1, choices].float()
                targets = torch.tensor([position, 1-position] if mode != 'swapped' else [1-position, position], device='cuda')
                correct += int((logits.argmax(-1) == targets).sum())
                both += int(bool((logits.argmax(-1) == targets).all()))
                pairs += 1
    return {'pairs': pairs, 'both_correct': both, 'examples': pairs*2, 'correct': correct}


def run(output=OUTPUT, steps=400):
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT/'runs').resolve()) or output == (ROOT/'runs').resolve():
        raise ValueError('Output must be a new directory under runs')
    if type(steps) is not int or not 1 <= steps <= 1000 or output.exists():
        raise ValueError('Invalid steps or existing output')
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA or sha(INDIVIDUAL) != INDIVIDUAL_SHA or sha(QUERY_PATH) != QUERY_SHA:
        raise ValueError('Protected source mismatch')
    torch.manual_seed(23802)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    query_model, query_state, _ = load(QUERY_PATH, 'cpu')
    if not hasattr(query_model, 'chinese_query') or any(not torch.equal(a.current_basis, b.current_basis)
            for a, b in zip(organism.memories, query_model.memories)):
        raise ValueError('Neural address model is incompatible with original individual')
    before = frozen_digest(organism, original)
    query_before = frozen_digest(query_model, query_state)
    states, events = variants(organism, original, TRAIN_PAIRS + HELD_PAIRS)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    choices = choice_ids(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='eager').to('cuda').eval()
    model.requires_grad_(False)
    adapter = OpenBaseMemory(model.config.hidden_size, organism.config.memory_width).to('cuda')
    layers = [model.model.layers[5], model.model.layers[17]]
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.002, weight_decay=0.)
    tasks = [(pair, position, template) for pair in TRAIN_PAIRS for position in range(2) for template in TRAIN_PROMPTS]
    log = []
    for step in range(steps):
        pair, position, template = tasks[step % len(tasks)]
        question = template.format(key=pair[position])
        addresses, parsed = neural_addresses(query_model, organism, question)
        if parsed['entity'] != pair[position]:
            raise ValueError('Neural address differs from training key')
        batch = pack_states(states[pair])
        inputs = encode_prompt(tokenizer, question, 'cuda')
        inputs = {k: v.expand(2, -1) for k, v in inputs.items()}
        targets = torch.tensor([choices[position], choices[1-position]], device='cuda')
        optimizer.zero_grad(set_to_none=True)
        with memory_hooks(layers, adapter, batch, current_only=True, addresses=addresses):
            logits = model(**inputs, use_cache=False).logits[:, -1, :].float()
            loss = F.cross_entropy(logits, targets)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step+1) % 50 == 0 or step+1 == steps:
            log.append({'step': step+1, 'loss': float(loss.detach()), 'gradient_norm': float(norm)})
            print(json.dumps(log[-1]), flush=True)
    adapter.eval()
    train = {pair: states[pair] for pair in TRAIN_PAIRS}
    held = {pair: states[pair] for pair in HELD_PAIRS}
    options = (model, layers, adapter, tokenizer, choices, query_model, organism)
    results = {'train': evaluate(*options, train, TRAIN_PROMPTS),
               'known_keys_new_prompts': evaluate(*options, train, HELD_PROMPTS),
               'new_keys_known_prompts': evaluate(*options, held, TRAIN_PROMPTS),
               'new_keys_new_prompts': evaluate(*options, held, HELD_PROMPTS),
               'swapped': evaluate(*options, held, HELD_PROMPTS, 'swapped'),
               'empty': evaluate(*options, held, HELD_PROMPTS, 'empty'),
               'disabled': evaluate(*options, held, HELD_PROMPTS, 'disabled')}
    if (any(p.grad is not None for p in model.parameters()) or frozen_digest(organism, original) != before
            or frozen_digest(query_model, query_state) != query_before or sha(INDIVIDUAL) != INDIVIDUAL_SHA
            or sha(QUERY_PATH) != QUERY_SHA or sha(MODEL/'model.safetensors') != WEIGHT_SHA):
        raise ValueError('Protected input changed')
    output.mkdir(parents=False, exist_ok=False)
    snapshot = output/'adapter.pt'
    torch.save({'format': 'spora-open-base-p26-addressed-v1',
                'state': {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA,
                'query_sha256': QUERY_SHA, 'layers': [5, 17], 'current_only': True}, snapshot)
    report = {'status': 'p26_addressed_controlled_memory_diagnostic', 'revision': REVISION,
              'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA,
              'query_sha256': QUERY_SHA, 'adapter_sha256': sha(snapshot), 'steps': steps,
              'train_pairs': TRAIN_PAIRS, 'held_pairs': HELD_PAIRS, 'train_prompts': TRAIN_PROMPTS,
              'held_prompts': HELD_PROMPTS, 'events': events, 'log': log, 'results': results,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'seconds': time.perf_counter()-start,
              'limitations': ['P26 was separately trained for finite numbered questions; this is not open semantic addressing.',
                              'P26 supplies a neural address, not the memory answer; current matrix is read within Qwen.',
                              'All states share one original background and contain two controlled writes.',
                              'Only values 1 and 2 are compared by logits; no free-generation or unknown handling is verified.',
                              'Held keys and prompts may have appeared during P26 training; no independent acceptance.']}
    with (output/'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output, args.steps)
