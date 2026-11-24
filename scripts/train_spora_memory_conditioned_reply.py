"""Compare memory-dependent generation with correlated and counterfactual supervision."""

import argparse
from contextlib import ExitStack, nullcontext
from pathlib import Path
import random
import re
from string import Formatter
from time import perf_counter

from scripts.audit_spora_recurrent_field_composition import SCENARIOS, read
from scripts.benchmark_spora_recurrent_dialogue import exchange, grade, reply, settings_for
from scripts.train_spora_latent_readout import (
    DATA, HELD, MODEL, SOURCE, AutoTokenizer, alternate, capture, dialogue_data, evaluate, extend_positions,
    frozen_digest, install, load_full_streamed, load_integrated_runtime, load_recurrent_replacements, prompt_ids,
    recall_prompts, records, sha, store_json, torch,
)
from spora.language_training import atomic_save
from spora.research_dialogue_working_state import MixtureWorkingTurnReader, WorkingTurnState, slot_continuations


SEED = 26024
MODES = ('correlated', 'paired')
TRANSFER_FIELDS = {
    'clinic_booking': ('time', 'place'),
    'repair_pickup': ('place', 'person'),
    'pottery_inventory': ('quantity', 'person'),
    'choir_program': ('gear', 'person'),
    'garden_supply': ('gear', 'place'),
    'film_notice': ('place', 'person'),
}


def tokens(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def state_for(tokenizer, values):
    return WorkingTurnState().update({field: tuple(tokens(tokenizer, value)) for field, value in values.items()})


def answer_ids(tokenizer, template, values):
    target = []
    for literal, field, _, _ in Formatter().parse(template):
        target.extend(tokens(tokenizer, literal))
        if field:
            target.extend(tokens(tokenizer, values[field]))
    if tokenizer.decode(target) != template.format_map(values):
        raise ValueError('Field-aligned tokenization changed the reference text')
    return [*target, tokenizer.eos_token_id]


def visible_history(case):
    history = []
    exchange(history, case['first'][0], '好，我看到了。')
    exchange(history, case['update'][0], '好，收到了。')
    return history


def build_training(tokenizer, count):
    payload = read(DATA)
    rng = random.Random(SEED)
    plans = {mode: [] for mode in MODES}
    for index, original in enumerate(dialogue_data(payload, held=False, count=count)):
        variants = (original, alternate(original, payload, rng))
        histories = [visible_history(case) for case in variants]
        if index % 3 == 0:
            # Both arms see this mix of state-only and visible evidence examples.
            histories = [[], []]
        prompt = [prompt_ids(tokenizer, history, original['question']) for history in histories]
        for branch, case in enumerate(variants):
            template = payload['kinds'][case['kind']]['answer']
            labels = answer_ids(tokenizer, template, case['current'])
            state = state_for(tokenizer, case['current'])
            plans['correlated'].append((prompt[branch], labels, state))
            plans['paired'].append((prompt[index % 2], labels, state))
    return plans


@torch.no_grad()
def cache_training(decoder, layer, plans):
    requests = dict.fromkeys(
        (tuple(prompt + labels[:-1]), len(labels))
        for arm in plans.values() for prompt, labels, _ in arm
    )
    print(f'Capturing {len(requests)} unique frozen decoder requests', flush=True)
    encoded = dict(zip(requests, capture(decoder, layer, requests), strict=True))
    return {mode: [(encoded[tuple(prompt + labels[:-1]), len(labels)], prompt, labels, state)
                   for prompt, labels, state in rows] for mode, rows in plans.items()}


def forward_example(base, reader, example):
    hidden, prompt, labels, state = example
    device = base.device
    ids, owners, positions = state.tokens(device)
    embeddings = base.get_input_embeddings()(ids)
    sequence = torch.tensor([prompt + labels[:-1]], device=device)
    projected = torch.arange(len(prompt) - 1, sequence.shape[1], device=device)
    continuation = slot_continuations(sequence, projected, len(prompt), ids, positions)
    changed = reader(hidden[None].to(device), embeddings, owners, positions)
    normed = base.model.norm(changed)
    weights = reader.attention(normed, embeddings, owners, positions, continuation)
    logits = reader.copy_scores(normed, base.lm_head(normed), ids, weights)[0]
    target = torch.tensor(labels, device=device)
    matches = ids[None, :] == target[:, None]
    is_value = matches.any(dim=-1)
    if not bool(is_value.any()):
        raise ValueError('A target must require copying at least one state token')
    mass = (weights[0] * matches).sum(dim=-1)
    score = logits.gather(-1, target[:, None]).squeeze(-1)
    return -score.mean() - .3 * mass[is_value].clamp_min(1e-8).log().mean(), logits


def fit(base, reader, samples, steps):
    rng = random.Random(SEED)
    order = list(range(len(samples)))
    optimizer = torch.optim.AdamW(reader.parameters(), lr=6e-4)
    trace, window = [], []
    for step in range(steps):
        if step % len(samples) == 0:
            rng.shuffle(order)
        loss, _ = forward_example(base, reader, samples[order[step % len(samples)]])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reader.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        window.append(float(loss.detach()))
        if (step + 1) % 64 == 0 or step + 1 == steps:
            trace.append({'step': step + 1, 'loss': sum(window) / len(window)})
            print(trace[-1], flush=True)
            window.clear()
    return trace


@torch.no_grad()
def verify_training_path(decoder, reader, sample):
    _, prompt, labels, state = sample
    _, offline = forward_example(decoder.base, reader, sample)
    ids = torch.tensor([prompt + labels[:-1]], device=decoder.base.device)
    with reader.active(decoder, state, prompt_tokens=len(prompt)):
        live = decoder(input_ids=ids, use_cache=True, logits_to_keep=len(labels)).logits[0]
    torch.testing.assert_close(live, offline, rtol=1e-4, atol=2e-4)


def one_answer(decoder, tokenizer, reader, history, row, new, foil, settings):
    values = row['values']
    updated, retained = TRANSFER_FIELDS[row['case']]
    state = state_for(tokenizer, {updated: new, retained: values['keep']})
    scope = reader.active(decoder, state) if reader is not None else nullcontext()
    with scope:
        answer = reply(decoder, tokenizer, history, row['question'], **settings)
    answer.pop('seconds')
    forbidden = (values['old'], foil)
    return {**answer, 'new': new in answer['answer'], 'keep': values['keep'] in answer['answer'],
            'foil': any(value in answer['answer'] for value in forbidden),
            'passed': grade(answer, [re.escape(new), re.escape(values['keep'])],
                            [re.escape(value) for value in forbidden])}


def heldout(decoder, tokenizer, reader, held, *, include_both=True):
    settings = settings_for(decoder, tokenizer, 48)
    rows = []
    for i, row in enumerate(held):
        values = row['values']
        other = held[i ^ 1]
        replacement = other['values']['new']
        if other['case'] != row['case'] or replacement in values.values():
            raise ValueError('A held-out swap must change exactly the requested value')
        variants = (('matched', values['new'], replacement),)
        if include_both:
            variants += (('swapped', replacement, values['new']),)
        outputs = {name: {arm: one_answer(decoder, tokenizer, reader, history, row, new, foil, settings)
                          for arm, new, foil in variants}
                   for name, history in (('long', row['history']), ('state_only', []))}
        rows.append({'case': row['case'], 'variant': row['variant'], 'values': values,
                     'replacement': replacement, 'outputs': outputs})
    return rows


def summary(rows):
    result = {}
    for scope in ('long', 'state_only'):
        cases = [row['outputs'][scope] for row in rows]
        result[scope] = {condition: {metric: sum(case[condition][metric] for case in cases)
                                     for metric in ('passed', 'new', 'keep', 'foil', 'eos')}
                         for condition in cases[0]}
        if 'swapped' in cases[0]:
            result[scope]['both'] = sum(case['matched']['passed'] and case['swapped']['passed'] for case in cases)
    return result


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    paths = (args.checkpoint, DATA, SOURCE, SCENARIOS, *(folder / filename for folder in args.candidates
                                                       for filename in ('attention.pt', 'comparison.json')))
    protected = {path: sha(path) for path in paths}
    sources = {str(path): digest for path, digest in protected.items()}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    plans = None if args.verify else build_training(tokenizer, args.count)
    scenarios = {**read(SOURCE), 'recall': read(SCENARIOS)['recall']}
    held = [row for row in recall_prompts(tokenizer, scenarios, extended=True) if row['length'] == 'extended']
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    torch.cuda.empty_cache()
    base.lm_head.to(base.device)
    individual = frozen_digest(runtime.organism, runtime.state)
    candidates = load_recurrent_replacements(decoder, args.candidates, protected[args.checkpoint])
    memory_cases = records(runtime.organism, HELD)
    layer = base.model.layers[-1]
    if args.verify:
        result = read(args.output / 'result.json')
        if result['sources'] != sources or result['individual_sha256'] != individual:
            raise ValueError('Inputs changed before restart')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        result = {'sources': sources, 'individual_sha256': individual, 'steps': args.steps, 'arms': {},
                  'examples_per_arm': len(plans['paired']), 'test_count': len(held)}
    with ExitStack() as stack:
        install(stack, candidates)
        baseline_memory = evaluate(decoder, tokenizer, runtime.organism, memory_cases)
        if not args.verify:
            cached = cache_training(decoder, layer, plans)
            torch.cuda.empty_cache()
        for mode in ((args.verify,) if args.verify else MODES):
            torch.manual_seed(SEED)
            reader = MixtureWorkingTurnReader(base.config.hidden_size).to(base.device)
            path = args.output / f'{mode}.pt'
            if args.verify:
                saved = torch.load(path, map_location='cpu', weights_only=True)
                if sha(path) != result['arms'][mode]['sha256'] or saved['mode'] != mode:
                    raise ValueError('The candidate changed before restart')
                reader.load_state_dict(saved['reader'])
            else:
                verify_training_path(decoder, reader, cached[mode][0])
                trace = fit(base, reader, cached[mode], args.steps)
                verify_training_path(decoder, reader, cached[mode][0])
                atomic_save({'reader': reader.state_dict(), 'mode': mode}, path)
            reader.eval().requires_grad_(False)
            rows = heldout(decoder, tokenizer, reader, held)
            state = state_for(tokenizer, {'place': held[0]['values']['keep']})
            with reader.active(decoder, state):
                memory = evaluate(decoder, tokenizer, runtime.organism, memory_cases)
            if memory != baseline_memory:
                raise ValueError('The candidate altered durable memory generation')
            print(mode, summary(rows), memory['summary'], flush=True)
            if args.verify:
                if rows != result['arms'][mode]['rows'] or memory != result['memory']:
                    raise ValueError('A fresh process changed the recorded result')
            else:
                result['arms'][mode] = {'sha256': sha(path), 'trace': trace, 'summary': summary(rows), 'rows': rows}
        if not args.verify:
            result['baseline'] = summary(heldout(decoder, tokenizer, None, held, include_both=False))
    if (frozen_digest(runtime.organism, runtime.state) != individual
            or any(sha(path) != digest for path, digest in protected.items())):
        raise ValueError('The experiment changed a frozen input or individual')
    elapsed = perf_counter() - started
    if args.verify:
        store_json(args.output / f'{args.verify}-reload.json', {'mode': args.verify, 'seconds': elapsed,
                                                                'test_count': len(held), 'sources': sources})
    else:
        store_json(args.output / 'result.json', {**result, 'memory': baseline_memory, 'seconds': elapsed})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('candidates', nargs=2, type=Path)
    parser.add_argument('--steps', type=int, default=384)
    parser.add_argument('--count', type=int, default=24, help='Training pairs per task family')
    parser.add_argument('--verify', choices=MODES)
    args = parser.parse_args()
    if args.steps < 1 or args.count < 3 or (not args.verify and args.output.exists()):
        parser.error('Choose a positive budget and a new output path')
    main(args)
