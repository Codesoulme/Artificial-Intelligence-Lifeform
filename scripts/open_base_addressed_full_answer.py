"""Train complete numeric replies from SPORA-written current memory snapshots."""
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

from scripts.open_base_addressed_alignment import QUERY_PATH, QUERY_SHA, neural_addresses, TRAIN_PROMPTS, HELD_PROMPTS
from scripts.open_base_baseline import MODEL, INDIVIDUAL, INDIVIDUAL_SHA, WEIGHT_SHA, REVISION, sha
from scripts.open_base_real_memory_alignment import TRAIN_PAIRS, HELD_PAIRS, native_read
from scripts.open_base_synthetic_alignment import encode_prompt
from spora.growing import load
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest
from spora.open_base_memory import OpenBaseMemory, memory_hooks

OUTPUT = ROOT/'runs/open-base-addressed-full-answer-20261109'
DEV_VALUES = (0, 1, 2, 3, 4, 7, 10, 15)


def create_snapshots(organism, original, pairs):
    """Two events per branch, with reversed assignments and native-read verification."""
    snapshots, actions = {}, []
    for pair in pairs:
        for value in range(16):
            assignments = ((value, (value+1)%16), ((value+1)%16, value))
            branches = []
            for assignment in assignments:
                state = copy.deepcopy(original)
                for key, target in zip(pair, assignment):
                    state, decision = observe_sourced(organism, state, key, target, 'direct', 1.0)
                    actions.append(decision['action'])
                actual = native_read(organism, state, list(pair))
                if actual != list(assignment):
                    raise ValueError(f'Native memory did not recover assignment {pair}: {actual}')
                branches.append(state)
            snapshots[(pair, value)] = SimpleNamespace(
                current=[torch.cat([state.current[site] for state in branches], dim=0).detach() for site in range(2)],
                pages=[])
    return snapshots, actions


def answer_batch(tokenizer, prompt, answers, device):
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None or not answers:
        raise ValueError('Missing answers, padding token or EOS')
    prefix = encode_prompt(tokenizer, prompt, device)['input_ids'][0].tolist()
    encoded = [tokenizer.encode(str(value), add_special_tokens=False) for value in answers]
    if any(not row for row in encoded):
        raise ValueError('Missing answer tokens')
    endings = [row + [tokenizer.eos_token_id] for row in encoded]
    sequences = [prefix+row for row in endings]
    length = max(len(row) for row in sequences)
    ids = torch.full((len(sequences), length), tokenizer.pad_token_id, dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    for i, seq in enumerate(sequences):
        ids[i, :len(seq)] = torch.tensor(seq, device=device)
        labels[i, len(prefix):len(seq)] = ids[i, len(prefix):len(seq)]
        attention[i, :len(seq)] = 1
    return {'input_ids': ids, 'attention_mask': attention}, labels


@torch.no_grad()
def generate_cases(model, layers, adapter, tokenizer, query_model, organism, snapshots, pairs, prompts, values, mode='normal'):
    rows = []
    for pair in pairs:
        for key_index, key in enumerate(pair):
            for template in prompts:
                question = template.format(key=key)
                addresses, parsed = neural_addresses(query_model, organism, question)
                if parsed['entity'] != key:
                    raise ValueError('Neural address mismatch')
                for value in values:
                    state = snapshots[(pair, value)]
                    if mode == 'swapped':
                        state = SimpleNamespace(current=[x.flip(0) for x in state.current], pages=[])
                    elif mode == 'empty':
                        state = SimpleNamespace(current=[torch.zeros_like(x) for x in state.current], pages=[])
                    expected = ([value, (value+1)%16] if key_index == 0 else [(value+1)%16, value])
                    if mode == 'swapped':
                        expected = list(reversed(expected))
                    inputs = encode_prompt(tokenizer, question, 'cuda')
                    inputs = {k: v.expand(2, -1) for k, v in inputs.items()}
                    with memory_hooks(layers, adapter, state, current_only=True, addresses=addresses):
                        generated = model.generate(**inputs, max_new_tokens=6, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                    responses = [tokenizer.decode(r[inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                                 for r in generated]
                    correct = [answer in (str(target), str(target)+'。', str(target)+'.')
                               for answer, target in zip(responses, expected)]
                    rows.append({'pair': list(pair), 'key': key, 'value_seed': value, 'question': question,
                                 'mode': mode, 'expected': expected, 'responses': responses, 'correct': correct})
    return rows


def summary(rows):
    return {'pairs': len(rows), 'both_correct': sum(all(r['correct']) for r in rows),
            'examples': 2*len(rows), 'correct': sum(sum(r['correct']) for r in rows)}


def run(output=OUTPUT, steps=1600):
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT/'runs').resolve()) or output == (ROOT/'runs').resolve():
        raise ValueError('Output must be a new directory under runs')
    if type(steps) is not int or not 1 <= steps <= 4000 or output.exists():
        raise ValueError('Invalid steps or existing output')
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA or sha(INDIVIDUAL) != INDIVIDUAL_SHA or sha(QUERY_PATH) != QUERY_SHA:
        raise ValueError('Protected source mismatch')
    torch.manual_seed(23803)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    query_model, query_state, _ = load(QUERY_PATH, 'cpu')
    if not hasattr(query_model, 'chinese_query') or any(not torch.equal(a.current_basis, b.current_basis)
            for a, b in zip(organism.memories, query_model.memories)):
        raise ValueError('Neural address model incompatible')
    before, query_before = frozen_digest(organism, original), frozen_digest(query_model, query_state)
    snapshots, actions = create_snapshots(organism, original, TRAIN_PAIRS + HELD_PAIRS)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='eager').to('cuda').eval()
    model.requires_grad_(False)
    adapter = OpenBaseMemory(model.config.hidden_size, organism.config.memory_width).to('cuda')
    layers = [model.model.layers[5], model.model.layers[17]]
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.001, weight_decay=0.)
    tasks = [(pair, position, template, value) for pair in TRAIN_PAIRS for position in range(2)
             for template in TRAIN_PROMPTS for value in range(16)]
    generator = torch.Generator().manual_seed(23803)
    order = torch.randperm(len(tasks), generator=generator).tolist()
    log = []
    for step in range(steps):
        if step and step % len(order) == 0:
            order = torch.randperm(len(tasks), generator=generator).tolist()
        pair, position, template, value = tasks[order[step % len(order)]]
        question = template.format(key=pair[position])
        addresses, parsed = neural_addresses(query_model, organism, question)
        if parsed['entity'] != pair[position]:
            raise ValueError('Neural address differs from training question')
        answers = [value, (value+1)%16] if position == 0 else [(value+1)%16, value]
        inputs, labels = answer_batch(tokenizer, question, answers, 'cuda')
        optimizer.zero_grad(set_to_none=True)
        with memory_hooks(layers, adapter, snapshots[(pair, value)], current_only=True, addresses=addresses):
            logits = model(**inputs, use_cache=False).logits[:, :-1, :].float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step+1) % 100 == 0 or step+1 == steps:
            log.append({'step': step+1, 'loss': float(loss.detach()), 'gradient_norm': float(norm)})
            print(json.dumps(log[-1]), flush=True)
    adapter.eval()
    options = (model, layers, adapter, tokenizer, query_model, organism, snapshots)
    rows = (generate_cases(*options, HELD_PAIRS, HELD_PROMPTS, DEV_VALUES)
            + generate_cases(*options, HELD_PAIRS, HELD_PROMPTS, DEV_VALUES, 'swapped')
            + generate_cases(*options, HELD_PAIRS, HELD_PROMPTS, DEV_VALUES, 'empty'))
    summaries = {mode: summary([r for r in rows if r['mode'] == mode]) for mode in ('normal', 'swapped', 'empty')}
    if (any(p.grad is not None for p in model.parameters()) or frozen_digest(organism, original) != before
            or frozen_digest(query_model, query_state) != query_before or sha(INDIVIDUAL) != INDIVIDUAL_SHA
            or sha(QUERY_PATH) != QUERY_SHA or sha(MODEL/'model.safetensors') != WEIGHT_SHA):
        raise ValueError('Protected input changed')
    output.mkdir(parents=False, exist_ok=False)
    snapshot = output/'adapter.pt'
    torch.save({'format': 'spora-open-base-p26-full-answer-v1',
                'state': {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA,
                'query_sha256': QUERY_SHA, 'layers': [5, 17], 'current_only': True}, snapshot)
    report = {'status': 'controlled_memory_generation_development_only', 'revision': REVISION,
              'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA,
              'query_sha256': QUERY_SHA, 'adapter_sha256': sha(snapshot), 'steps': steps,
              'train_pairs': TRAIN_PAIRS, 'held_pairs': HELD_PAIRS, 'train_prompts': TRAIN_PROMPTS,
              'held_prompts': HELD_PROMPTS, 'dev_values': DEV_VALUES, 'action_counts': {
                  action: actions.count(action) for action in sorted(set(actions))},
              'log': log, 'summary': summaries, 'rows': rows,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'seconds': time.perf_counter()-start,
              'limitations': ['P26 is a separately trained finite numbered-question reader, not open semantic addressing.',
                              'All states share the original background and two controlled writes; no independent trajectory.',
                              'All 16 numeric values were trained, even where object pairs were held out.',
                              'Swapped rows repeat the same two states in reverse order; not independent examples.',
                              'Exact numeric scoring may reject correct full-sentence answers; actual responses retained.',
                              'P26 may have seen held questions in its own training; not final acceptance.']}
    with (output/'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output, args.steps)
