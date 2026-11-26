"""Recover real discrete attention layouts with matched Chinese response supervision."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_hybrid_language_likelihood import cases, score
from scripts.benchmark_spora_recurrent_dialogue import (
    MODEL, AutoTokenizer, encode, exchange, extend_positions, frozen_digest, load_full_streamed,
    load_integrated_runtime, sha, using,
)
from scripts.train_spora_recurrent_counterfactual import checkpoint_layers, store_json
from spora.language_training import atomic_save
from spora.research_recurrent_decode import load_recurrent_replacements


DEFAULT_LAYOUTS = ((19, 22), (19, 21, 22), (19, 22, 24))


def layout_name(layout):
    return '-'.join(map(str, layout))


def parse_layout(value):
    try:
        layers = tuple(sorted(map(int, value.split(','))))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use comma-separated layer numbers') from exc
    if len(set(layers)) != len(layers) or min(layers) < 0:
        raise argparse.ArgumentTypeError('A layout needs distinct nonnegative layer numbers')
    return layers


def response_logits(decoder, row, *, cache):
    tokens = torch.tensor([row['prompt'] + row['response'][:-1]], device=decoder.base.device)
    return decoder(input_ids=tokens, use_cache=cache, logits_to_keep=len(row['response'])).logits[0]


def revision_cases(path, tokenizer):
    source = json.loads(path.read_text(encoding='utf-8'))
    distractors = []
    for user, assistant in source['distractors']:
        exchange(distractors, user, assistant)
    rows = []
    for index, case in enumerate(source['cases']):
        changed, retained = case['fields']
        for variant, (old, new, keep) in enumerate(case['values']):
            initial, update = [], []
            explicit = (index // 4 + variant) % 2 == 0
            far = (index // 2 + variant) % 2 == 0
            exchange(initial, case['first'].format(old=old, keep=keep), '我记下了。')
            exchange(update, case['update'].format(field=changed if explicit else '这项', new=new), '收到。')
            if variant == 0:
                question = case['question']
                answer = f'{changed}是{new}，{retained}是{keep}。'
                target = 'both'
            else:
                field, answer, target = (changed, new, 'new') if index % 2 == 0 else (retained, keep, 'keep')
                question = f'现在的{field}是什么？请只写这一项的值。'
            history = initial + (update + distractors if far else distractors + update)
            prompt = encode(tokenizer, [*history, {'role': 'user', 'content': question}])
            response = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
            if len(prompt) + len(response) > 2048 or len({old, new, keep}) != 3:
                raise ValueError(f'Invalid revision training example: {case["name"]}')
            rows.append(dict(dialogue=f'revision/{case["name"]}/{variant}', prompt=prompt, response=response,
                             target=target, explicit=explicit, far=far))
    return rows


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
    extra_sources = (args.revision_train,) if args.revision_train else ()
    protected = {path: sha(path) for path in (
        args.checkpoint, args.train, args.test, *extra_sources, *(directory / name for directory in args.layers
                                                 for name in ('attention.pt', 'comparison.json')))}
    torch.manual_seed(50721)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    training = cases(args.train, tokenizer, args.steps, split='train')
    if args.revision_train:
        revisions = revision_cases(args.revision_train, tokenizer)
        if not revisions or 2 * len(revisions) > args.steps:
            raise ValueError('Allow at least one replay step per revision example')
        training = training[:-len(revisions)]
        for index, revision in enumerate(revisions):
            training.insert(2 * index + 1, revision)
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
    layouts = args.layout or DEFAULT_LAYOUTS
    required_layers = {layer for layout in layouts for layer in layout}
    results = {'parent': score(decoder, held)}
    for layout in layouts:
        name = layout_name(layout)
        output = args.output / name
        output.mkdir()
        loaded = load_recurrent_replacements(decoder, args.layers, protected[args.checkpoint])
        if set(loaded) != required_layers:
            raise ValueError('Input candidates must cover the requested layouts exactly')
        replacements = {index: loaded[index] for index in layout}
        with using(replacements, 'recurrent'):
            before = score(decoder, held)
        torch.cuda.reset_peak_memory_stats()
        trace = train(decoder, replacements, training, teacher, args.learning_rate)
        peak = torch.cuda.max_memory_allocated()
        with using(replacements, 'recurrent'):
            after = score(decoder, held)
        candidate_hashes = {}
        for index, (_, replacement) in replacements.items():
            directory = output / f'layer-{index}'
            directory.mkdir()
            core = replacement.core
            path = directory / 'attention.pt'
            atomic_save(dict(format='spora-recurrent-attention-v1', layer=index, block=core.block,
                             checkpoint_sha256=protected[args.checkpoint], update_rule=core.update_rule,
                             history_projection=replacement.history_projection, core=core.state_dict()), path)
            candidate_hashes[index] = sha(path)
            store_json(directory / 'comparison.json', dict(layer=index, block=core.block, sources=sources,
                       update_rule=core.update_rule, history_projection=replacement.history_projection,
                       individual_sha256=identity, candidate_sha256=candidate_hashes[index]))
        result = dict(before=before, after=after, trace=trace, training_peak_bytes=peak,
                      candidate_hashes=candidate_hashes)
        if identity != frozen_digest(runtime.organism, runtime.state):
            raise RuntimeError('Individual changed during recovery')
        store_json(output / 'result.json', result)
        results[name] = result
        print(name, before['mean_nll'], after['mean_nll'], before['accuracy'], after['accuracy'], flush=True)
        del loaded, replacements
        torch.cuda.empty_cache()
    if any(sha(path) != digest for path, digest in protected.items()):
        raise RuntimeError('A protected source changed')
    store_json(args.output / 'result.json', dict(results=results, layouts=layouts, sources=sources, individual_sha256=identity,
               training=training, held=held, learning_rate=args.learning_rate, seconds=perf_counter() - started))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('train', type=Path)
    cli.add_argument('test', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('layers', nargs='+', type=Path)
    cli.add_argument('--layout', type=parse_layout, action='append', help='Repeat with comma-separated layer numbers')
    cli.add_argument('--steps', type=int, default=24)
    cli.add_argument('--revision-train', type=Path, help='Replace replay steps with paired-field revision supervision')
    cli.add_argument('--count', type=int, default=48)
    cli.add_argument('--learning-rate', type=float, default=.0005)
    args = cli.parse_args()
    if args.steps < 1 or args.count < 1 or not 0 < args.learning_rate <= .01:
        cli.error('Expected positive sample counts and learning rate through .01')
    if args.layout and len(set(args.layout)) != len(args.layout):
        cli.error('Use distinct layouts')
    main(args)
