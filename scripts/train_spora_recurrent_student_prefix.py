"""Compare matched answer-prefix and student-prefix distillation in the same recurrent layout."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from scripts.benchmark_spora_recurrent_dialogue import DynamicCache, using
from scripts.train_spora_hybrid_hidden_recovery import (
    CORPUS, DATA, HELD, LAYERS, LOSS_SCALE, MANIFEST, MODEL, SOURCE, AutoTokenizer, checkpoint_layers,
    dialogue_data, evaluate, extend_positions, frozen_digest, load_full_streamed, load_integrated_runtime,
    load_recurrent_replacements, recall_prompts, records, run_dialogue, save_training, sha, store_json,
    training_pairs,
)
from scripts.train_spora_hybrid_language_recovery import response_logits


@torch.no_grad()
def rollout(decoder, prompt, horizon, eos):
    cache = DynamicCache(config=decoder.base.config)
    tokens = torch.tensor([prompt], device=decoder.base.device)
    generated = []
    for _ in range(horizon):
        logits = decoder(input_ids=tokens, past_key_values=cache, use_cache=True, logits_to_keep=1).logits
        tokens = logits[:, -1].argmax(dim=-1, keepdim=True)
        token = int(tokens.item())
        generated.append(token)
        if token == eos:
            break
    return generated


def distill(decoder, replacements, pairs, args, eos, *, schedule=None):
    student_prefix = schedule is None
    parameters = [parameter for _, module in replacements.values() for parameter in module.core.parameters()]
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise ValueError('Scaled backward requires FP32 recurrent weights')
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.)
    traces, trajectories = [], []
    for step, pair in enumerate(pairs, 1):
        prompts = [ids[:len(ids) - len(answer) + 1] for ids, answer, *_ in pair]
        if student_prefix:
            with using(replacements, 'recurrent'):
                responses = [rollout(decoder, prompt, min(len(arm[1]), args.horizon), eos)
                             for prompt, arm in zip(prompts, pair, strict=True)]
        else:
            responses = [arm[1][:len(generated)] for arm, generated in zip(pair, schedule[step - 1], strict=True)]
        trajectories.append(responses)
        rows = [dict(prompt=prompt, response=response) for prompt, response in zip(prompts, responses, strict=True)]
        with torch.no_grad():
            reference = [response_logits(decoder, row, cache=False).cpu() for row in rows]
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _, replacement in replacements.values():
            replacement.batched_training = True
            replacement.core.train().requires_grad_(True)
        try:
            with using(replacements, 'recurrent'), checkpoint_layers(decoder, min(replacements) + 1):
                for row, expected in zip(rows, reference, strict=True):
                    logits = response_logits(decoder, row, cache=False).float().log_softmax(-1)
                    target = expected.to(logits.device).float().log_softmax(-1)
                    loss = (logits.exp() * (logits - target)).sum(-1).mean()
                    (loss * (.5 * LOSS_SCALE)).backward()
                    losses.append(float(loss.detach()))
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.div_(LOSS_SCALE)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
        finally:
            for _, replacement in replacements.values():
                replacement.core.eval().requires_grad_(False)
        trace = dict(step=step, kl=sum(losses) / len(losses), gradient=float(norm),
                     lengths=[len(row['prompt']) + len(row['response']) - 1 for row in rows],
                     completion_tokens=[len(response) for response in responses])
        traces.append(trace)
        print('Student' if student_prefix else 'Answer', trace, flush=True)
    return traces, trajectories


def main(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = (args.checkpoint, args.reference, CORPUS, DATA, SOURCE, MANIFEST,
              *(path / name for path in args.candidates for name in ('attention.pt', 'comparison.json')))
    protected = {path: sha(path) for path in inputs}
    sources = {str(path): digest for path, digest in protected.items()}
    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    expected = reference['results']['targeted']['training']['candidate_hashes']
    if (json.loads(MANIFEST.read_text(encoding='utf-8'))['files'][CORPUS.name] != protected[CORPUS]
            or any(protected[directory / 'attention.pt'] != expected[str(index)]
                   for index, directory in zip(LAYERS, args.candidates, strict=True))):
        raise ValueError('Source corpus or trained recurrent weights differ from the experiment reference')
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    training = json.loads(DATA.read_text(encoding='utf-8'))
    prompts = recall_prompts(tokenizer, source, extended=True)
    examples = dialogue_data(training, held=False, count=16)
    if {row['question'] for row in prompts} & {row['question'] for row in examples}:
        raise ValueError('The training and test questions overlap')
    order = reference['order'][:args.steps]
    if len(order) != args.steps:
        raise ValueError('The reference did not use the requested number of distinct pairs')
    pairs = training_pairs(tokenizer, training, examples)
    pairs = [pairs[index] for index in order]
    if [[len(arm[0]) for arm in pair] for pair in pairs] != [
            row['lengths'] for row in reference['results']['targeted']['training']['trace'][:args.steps]]:
        raise ValueError('The reference used a different training schedule')
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    base.lm_head.to(base.device)
    identity = frozen_digest(runtime.organism, runtime.state)
    if reference['individual_sha256'] != identity:
        raise ValueError('The reference used a different individual')
    memory = records(runtime.organism, HELD)
    schedule, results = None, {}
    for label in ('student', 'answer'):
        replacements = load_recurrent_replacements(decoder, args.candidates, protected[args.checkpoint])
        if set(replacements) != set(LAYERS):
            raise ValueError('Expected recurrent layers 19, 22 and 24')
        torch.cuda.reset_peak_memory_stats()
        trace, trajectories = distill(decoder, replacements, pairs, args, tokenizer.eos_token_id, schedule=schedule)
        if schedule is None:
            schedule = trajectories
        elif any(old['lengths'] != new['lengths'] for old, new in zip(results['student']['training']['trace'],
                                                                      trace, strict=True)):
            raise RuntimeError('The two arms used different training token budgets')
        peak = torch.cuda.max_memory_allocated()
        output = args.output / label
        output.mkdir()
        training_args = argparse.Namespace(output=output, steps=args.steps, learning_rate=args.learning_rate,
                                           field_weight=0., segment_supervision=False)
        saved = save_training(training_args, replacements, sources, protected[args.checkpoint], identity,
                              trace, len(pairs), peak)
        del replacements
        directories = [output / f'layer-{index}' for index in LAYERS]
        reloaded = load_recurrent_replacements(decoder, directories, protected[args.checkpoint])
        with using(reloaded, 'recurrent'):
            dialogue = run_dialogue(decoder, tokenizer, prompts, source['open'])
            persistent = evaluate(decoder, tokenizer, runtime.organism, memory)
        results[label] = dict(training=saved, trajectories=trajectories, dialogue=dialogue, memory=persistent)
        store_json(output / 'result.json', results[label])
        print(label, dialogue['recall_passed'], dialogue['open_turns_passed'], persistent['summary'], flush=True)
        del reloaded
        torch.cuda.empty_cache()
    if identity != frozen_digest(runtime.organism, runtime.state) or any(
            sha(path) != digest for path, digest in protected.items()):
        raise RuntimeError('A source or the individual changed')
    store_json(args.output / 'result.json', dict(results=results, order=order, horizon=args.horizon,
                                                loss_scale=LOSS_SCALE, sources=sources, individual_sha256=identity,
                                                seconds=perf_counter() - started))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('reference', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('candidates', type=Path, nargs=3)
    cli.add_argument('--steps', type=int, default=16)
    cli.add_argument('--horizon', type=int, default=24)
    cli.add_argument('--learning-rate', type=float, default=.0005)
    args = cli.parse_args()
    if not 1 <= args.steps <= 16 or not 2 <= args.horizon <= 48 or not 0 < args.learning_rate <= .01:
        cli.error('Expected 1–16 updates, a horizon of 2–48 tokens and a learning rate through .01')
    main(args)
