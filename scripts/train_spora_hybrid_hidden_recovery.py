"""Measure and recover long-dialogue hidden states after adding recurrent layer 24."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import random
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_hybrid_language_likelihood import cases, score
from scripts.benchmark_spora_recurrent_dialogue import encode, using
from scripts.train_spora_recurrent_counterfactual import (
    CORPUS, DATA, HELD, MANIFEST, MODEL, SOURCE, AutoTokenizer, checkpoint_layers, dialogue_data,
    evaluate, extend_positions, frozen_digest, load_full_streamed, load_integrated_runtime,
    load_recurrent_replacements, recall_prompts, records, run_dialogue, save_training, sha, store_json,
    training_pairs, SEED,
)


LAYERS = (19, 22, 24)
PROBES = (19, 22, 23, 24, 25, 27)


def forward(decoder, ids, answer_size, layers):
    captures = {}

    def hook(index):
        def capture(_module, _inputs, output):
            captures[index] = output[0] if isinstance(output, tuple) else output
        return capture

    tokens = torch.tensor([ids], device=decoder.base.device)
    with ExitStack() as handles:
        for index in layers:
            handle = decoder.base.model.layers[index].register_forward_hook(hook(index))
            handles.callback(handle.remove)
        logits = decoder(input_ids=tokens, use_cache=False, logits_to_keep=answer_size).logits[0]
    return logits, captures


def relative_error(actual, expected):
    expected = expected.to(device=actual.device, dtype=torch.float32)
    return (actual.float() - expected).square().mean() / expected.square().mean().clamp_min(1e-8)


def fit(decoder, replacements, batches, teacher, args, weight):
    params = []
    for _, replacement in replacements.values():
        replacement.batched_training = True
        replacement.core.train().requires_grad_(True)
        params.extend(replacement.core.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.)
    trace = []
    try:
        with using(replacements, 'recurrent'), checkpoint_layers(decoder, min(replacements) + 1):
            for step, (pair, targets) in enumerate(zip(batches, teacher, strict=True), 1):
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for (ids, answer, difference, foil, _), reference in zip(pair, targets, strict=True):
                    logits, captures = forward(decoder, ids, len(answer), LAYERS if weight else ())
                    logits = logits.float()
                    gold = torch.tensor(answer, device=logits.device)
                    margin = logits[difference, gold[difference]] - logits[difference, foil]
                    language = F.cross_entropy(logits, gold) + .3 * F.softplus(1. - margin)
                    hidden = (torch.stack([relative_error(captures[index], reference[index])
                                           for index in LAYERS]).mean() if weight else logits.new_zeros(()))
                    ((language + weight * hidden) * .5).backward()
                    losses.append((float(language.detach()), float(hidden.detach())))
                norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                trace.append(dict(step=step, lengths=[len(arm[0]) for arm in pair], gradient=float(norm),
                                  language=sum(loss[0] for loss in losses) / 2,
                                  hidden=sum(loss[1] for loss in losses) / 2))
                optimizer.step()
                if step == 1 or step % 4 == 0 or step == len(batches):
                    print('Training', weight, trace[-1], flush=True)
    finally:
        for _, replacement in replacements.values():
            replacement.core.eval().requires_grad_(False)
    return trace


@torch.no_grad()
def activation_targets(decoder, ids, layers):
    _, captures = forward(decoder, ids, 1, layers)
    return {index: value.detach().cpu() for index, value in captures.items()}


@torch.no_grad()
def activation_errors(decoder, probes, targets):
    rows = []
    for prompt, reference in zip(probes, targets, strict=True):
        _, captures = forward(decoder, prompt, 1, PROBES)
        rows.append({index: {'all': float(relative_error(captures[index], reference[index])),
                             'last_16': float(relative_error(captures[index][:, -16:], reference[index][:, -16:]))}
                     for index in PROBES})
    return rows


def main(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = (args.checkpoint, args.crosswoz, SOURCE, DATA, CORPUS, MANIFEST,
              *(directory / name for directory in args.candidates for name in ('attention.pt', 'comparison.json')))
    protected = {path: sha(path) for path in inputs}
    if json.loads(MANIFEST.read_text(encoding='utf-8'))['files'][CORPUS.name] != protected[CORPUS]:
        raise ValueError('Training corpus digest mismatch')
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    training = json.loads(DATA.read_text(encoding='utf-8'))
    prompts = recall_prompts(tokenizer, source, extended=True)
    examples = dialogue_data(training, held=False, count=16)
    held_questions = {row['question'] for row in prompts}
    if held_questions & {row['question'] for row in examples}:
        raise ValueError('A held question occurs in training')
    pairs = training_pairs(tokenizer, training, examples)
    order = list(range(len(pairs)))
    random.Random(SEED).shuffle(order)
    longest = max(order, key=lambda index: max(len(arm[0]) for arm in pairs[index]))
    order.remove(longest)
    order.insert(0, longest)
    batches = [pairs[index] for index in order[:args.steps]]
    if len(batches) != args.steps:
        raise ValueError('Not enough distinct training pairs')
    probes = [encode(tokenizer, [*row['history'], {'role': 'user', 'content': row['question']}])
              for row in prompts if row['length'] == 'extended']
    held = cases(args.crosswoz, tokenizer, 48)
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    base.lm_head.to(base.device)
    identity = frozen_digest(runtime.organism, runtime.state)
    teacher = [[activation_targets(decoder, arm[0], LAYERS) for arm in pair] for pair in batches]
    held_targets = [activation_targets(decoder, ids, PROBES) for ids in probes]
    memory = records(runtime.organism, HELD)
    sources = {str(path): digest for path, digest in protected.items()}
    results = {}
    for label, weight in (('answer', 0.), ('hidden', args.hidden_weight)):
        replacements = load_recurrent_replacements(decoder, args.candidates, protected[args.checkpoint])
        if set(replacements) != set(LAYERS):
            raise ValueError('Expected recurrent layers 19, 22, and 24')
        if not results:
            with using({index: replacements[index] for index in (19, 22)}, 'recurrent'):
                results['two_layer'] = activation_errors(decoder, probes, held_targets)
            with using(replacements, 'recurrent'):
                results['initial'] = activation_errors(decoder, probes, held_targets)
        torch.cuda.reset_peak_memory_stats()
        trace = fit(decoder, replacements, batches, teacher, args, weight)
        peak = torch.cuda.max_memory_allocated()
        output = args.output / label
        output.mkdir()
        training_args = argparse.Namespace(output=output, steps=args.steps, learning_rate=args.learning_rate,
                                           field_weight=0., segment_supervision=False)
        saved = save_training(training_args, replacements, sources, protected[args.checkpoint], identity,
                              trace, len(batches), peak)
        directories = [output / f'layer-{index}' for index in LAYERS]
        reloaded = load_recurrent_replacements(decoder, directories, protected[args.checkpoint])
        with using(reloaded, 'recurrent'):
            hidden = activation_errors(decoder, probes, held_targets)
            language = score(decoder, held)
            dialogue = run_dialogue(decoder, tokenizer, prompts, source['open'])
            persistent = evaluate(decoder, tokenizer, runtime.organism, memory)
        row = dict(training=saved, hidden=hidden, hidden_weight=weight, language=language,
                   dialogue=dialogue, memory=persistent)
        store_json(output / 'result.json', row)
        results[label] = row
        print(label, language['mean_nll'], dialogue['recall_passed'], dialogue['open_turns_passed'],
              persistent['summary'], flush=True)
        del reloaded, replacements
        torch.cuda.empty_cache()
    if identity != frozen_digest(runtime.organism, runtime.state) or any(
            sha(path) != digest for path, digest in protected.items()):
        raise RuntimeError('Source or individual changed')
    store_json(args.output / 'result.json', dict(results=results, sources=sources, individual_sha256=identity,
               order=order[:args.steps], probes=PROBES, seconds=perf_counter() - started))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('crosswoz', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('candidates', type=Path, nargs=3)
    cli.add_argument('--steps', type=int, default=16)
    cli.add_argument('--learning-rate', type=float, default=.0005)
    cli.add_argument('--hidden-weight', type=float, default=1.)
    args = cli.parse_args()
    if args.steps < 1 or not 0 < args.learning_rate <= .01 or not 0 < args.hidden_weight <= 10:
        cli.error('Expected positive steps, learning rate through .01, and hidden weight through 10')
    main(args)
