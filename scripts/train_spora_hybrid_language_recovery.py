"""Recover real discrete attention layouts with matched Chinese response supervision."""

import argparse
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_hybrid_language_likelihood import cases, score
from scripts.benchmark_spora_recurrent_dialogue import (
    MODEL, AutoTokenizer, extend_positions, frozen_digest, load_full_streamed,
    load_integrated_runtime, sha, using,
)
from scripts.train_spora_recurrent_counterfactual import checkpoint_layers, store_json
from spora.language_training import atomic_save
from spora.research_recurrent_decode import load_recurrent_replacements


def response_logits(decoder, row, *, cache):
    tokens = torch.tensor([row['prompt'] + row['response'][:-1]], device=decoder.base.device)
    return decoder(input_ids=tokens, use_cache=cache, logits_to_keep=len(row['response'])).logits[0]


def train(decoder, replacements, rows, teacher, learning_rate):
    params = []
    for _, replacement in replacements.values():
        replacement.batched_training = True
        replacement.core.train().requires_grad_(True)
        params.extend(replacement.core.parameters())
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.)
    trace = []
    try:
        with using(replacements, 'recurrent'), checkpoint_layers(decoder, min(replacements) + 1):
            for step, (row, reference) in enumerate(zip(rows, teacher, strict=True), 1):
                optimizer.zero_grad(set_to_none=True)
                logits = response_logits(decoder, row, cache=False).float()
                target = torch.tensor(row['response'], device=logits.device)
                probability = reference.to(logits.device).float().softmax(-1)
                ce = F.cross_entropy(logits, target)
                kl = F.kl_div(logits.log_softmax(-1), probability, reduction='batchmean')
                loss = .5 * (ce + kl)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                optimizer.step()
                trace.append(dict(step=step, dialogue=row['dialogue'], ce=float(ce.detach()),
                                  kl=float(kl.detach()), gradient=float(norm)))
                if step == 1 or step % 4 == 0 or step == len(rows):
                    print('Training', trace[-1], flush=True)
    finally:
        for _, replacement in replacements.values():
            replacement.core.eval().requires_grad_(False)
    return trace


def main(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    protected = {path: sha(path) for path in (
        args.checkpoint, args.train, args.test, *(directory / name for directory in args.layers
                                                 for name in ('attention.pt', 'comparison.json')))}
    torch.manual_seed(50721)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    training = cases(args.train, tokenizer, args.steps, split='train')
    held = cases(args.test, tokenizer, args.count)
    if {row['dialogue'] for row in training} & {row['dialogue'] for row in held}:
        raise ValueError('Training and test dialogue identifiers overlap')
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    base.lm_head.to(base.device)
    identity = frozen_digest(runtime.organism, runtime.state)
    sources = {str(path): digest for path, digest in protected.items()}
    with torch.no_grad():
        teacher = [response_logits(decoder, row, cache=False).cpu() for row in training]
    results = {'parent': score(decoder, held)}
    for layout in ((19, 22), (19, 21, 22), (19, 22, 24)):
        name = '-'.join(map(str, layout))
        output = args.output / name
        output.mkdir()
        loaded = load_recurrent_replacements(decoder, args.layers, protected[args.checkpoint])
        replacements = {index: loaded[index] for index in layout}
        with using(replacements, 'recurrent'):
            before = score(decoder, held)
        torch.cuda.reset_peak_memory_stats()
        trace = train(decoder, replacements, training, teacher, args.learning_rate)
        peak = torch.cuda.max_memory_allocated()
        with using(replacements, 'recurrent'):
            after = score(decoder, held)
        for index, (_, replacement) in replacements.items():
            directory = output / f'layer-{index}'
            directory.mkdir()
            core = replacement.core
            path = directory / 'attention.pt'
            atomic_save(dict(format='spora-recurrent-attention-v1', layer=index, block=core.block,
                             checkpoint_sha256=protected[args.checkpoint], update_rule=core.update_rule,
                             history_projection=replacement.history_projection, core=core.state_dict()), path)
            store_json(directory / 'comparison.json', dict(layer=index, block=core.block, sources=sources,
                       update_rule=core.update_rule, history_projection=replacement.history_projection,
                       individual_sha256=identity, candidate_sha256=sha(path)))
        result = dict(before=before, after=after, trace=trace, training_peak_bytes=peak)
        if identity != frozen_digest(runtime.organism, runtime.state):
            raise RuntimeError('Individual changed during recovery')
        store_json(output / 'result.json', result)
        results[name] = result
        print(name, before['mean_nll'], after['mean_nll'], before['accuracy'], after['accuracy'], flush=True)
        del loaded, replacements
        torch.cuda.empty_cache()
    if any(sha(path) != digest for path, digest in protected.items()):
        raise RuntimeError('A protected source changed')
    store_json(args.output / 'result.json', dict(results=results, sources=sources, individual_sha256=identity,
               training=training, held=held, learning_rate=args.learning_rate, seconds=perf_counter() - started))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('train', type=Path)
    cli.add_argument('test', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('layers', nargs=4, type=Path)
    cli.add_argument('--steps', type=int, default=24)
    cli.add_argument('--count', type=int, default=48)
    cli.add_argument('--learning-rate', type=float, default=.0005)
    args = cli.parse_args()
    if args.steps < 1 or args.count < 1 or not 0 < args.learning_rate <= .01:
        cli.error('Expected positive sample counts and learning rate through .01')
    main(args)
