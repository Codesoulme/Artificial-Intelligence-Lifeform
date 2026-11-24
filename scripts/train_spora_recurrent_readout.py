"""Tune the final dialogue residual against frozen recurrent attention."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import random
from time import perf_counter

from torch.nn import functional as F

from scripts.audit_spora_recurrent_field_composition import SCENARIOS
from scripts.train_spora_recurrent_counterfactual import (
    CORPUS, DATA, HELD, MANIFEST, MODEL, SOURCE, AutoTokenizer, dialogue_data, encode,
    evaluate, extend_positions, frozen_digest, install, load_full_streamed, load_integrated_runtime,
    load_recurrent_replacements, records, recall_prompts, run_dialogue, sha, store_json, torch, training_pairs,
)
from spora.language_training import atomic_save
from spora.pretrained_attention import SPORAResidualBlock


FOUNDATIONS = SOURCE.parent / 'chinese-foundations.json'
SEED = 25113


class _ReadoutCaptured(Exception):
    pass


def project(base, hidden, down, up):
    residual = F.linear(F.gelu(F.linear(hidden.float(), down)), up)
    hidden = hidden + residual.to(hidden.dtype)
    return base.lm_head(base.model.norm(hidden)).float()


@torch.no_grad()
def capture(decoder, block, examples):
    rows = []
    device = decoder.base.device
    width = 1
    verify = True

    def collect(_module, _args, output):
        rows.append(output[0, -width:].detach().cpu())
        if not verify:
            raise _ReadoutCaptured

    handle = block.base.register_forward_hook(collect)
    try:
        for ids, labels, _ in examples:
            width = len(labels)
            inputs = torch.tensor([ids], device=device)
            try:
                outputs = decoder(input_ids=inputs, use_cache=True, logits_to_keep=width)
            except _ReadoutCaptured:
                continue
            if not verify:
                raise RuntimeError('Final layer did not return the captured features')
            expected = outputs.logits[0].float()
            actual = project(decoder.base, rows[-1].to(device), block.down.weight, block.up.weight)
            torch.testing.assert_close(actual, expected, atol=.03, rtol=.003)
            verify = False
    finally:
        handle.remove()
    return [(hidden, labels, fields) for hidden, (_, labels, fields) in zip(rows, examples, strict=True)]


def replay_examples(tokenizer, path, excluded):
    scenarios = json.loads(path.read_text(encoding='utf-8'))['dialogues']
    examples = []
    for scene in scenarios:
        question, answer = scene['turns'][0]
        if question in excluded:
            continue
        prompt = encode(tokenizer, [{'role': 'user', 'content': question}])
        tokens = tokenizer.encode(answer, add_special_tokens=False)
        if 2 <= len(tokens) <= 72:
            examples.append((prompt + tokens, [*tokens, tokenizer.eos_token_id], ()))
        if len(examples) == 20:
            return examples
    raise ValueError('Too few independent replay examples')


def fit(base, block, requests, replay, steps):
    parameters = (block.down.weight.requires_grad_(True), block.up.weight.requires_grad_(True))
    anchor = tuple(parameter.detach().clone() for parameter in parameters)
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.)
    rng = random.Random(SEED)
    schedule = list(range(len(requests)))
    rng.shuffle(schedule)
    train_steps, trace = 0, []
    losses = {'answer': [], 'replay': []}
    for step in range(steps):
        kind = 'answer' if step % 4 != 3 else 'replay'
        if kind == 'answer':
            hidden, labels, fields = requests[schedule[train_steps % len(schedule)]]
            train_steps += 1
            logits = project(base, hidden.to(base.device), *parameters)
            nll = F.cross_entropy(logits, torch.tensor(labels, device=base.device), reduction='none')
            per_field = torch.stack([nll[list(positions)].mean() for positions in fields])
            loss = (nll.mean() + per_field.mean()) * .5
        else:
            hidden, _, _ = replay[rng.randrange(len(replay))]
            hidden = hidden[[0, -1]].to(base.device)
            with torch.no_grad():
                reference = project(base, hidden, *anchor).softmax(-1)
            logits = project(base, hidden, *parameters)
            loss = F.kl_div(logits.log_softmax(-1), reference, reduction='batchmean')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        losses[kind].append(float(loss.detach()))
        if (step + 1) % 32 == 0 or step + 1 == steps:
            trace.append({'step': step + 1, **{name: sum(values) / len(values) if values else None
                                                for name, values in losses.items()}})
            print(trace[-1], flush=True)
            for values in losses.values():
                values.clear()
    block.requires_grad_(False)
    return trace


def main(args):
    started = perf_counter()
    inputs = [args.checkpoint, SOURCE, SCENARIOS, FOUNDATIONS, DATA, CORPUS, MANIFEST]
    inputs.extend(folder / name for folder in args.candidates for name in ('attention.pt', 'comparison.json'))
    protected = {path: sha(path) for path in inputs}
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if manifest['files'][CORPUS.name] != protected[CORPUS]:
        raise ValueError('The training corpus changed')
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    scenarios = json.loads(SOURCE.read_text(encoding='utf-8'))
    transfer = {**scenarios, 'recall': json.loads(SCENARIOS.read_text(encoding='utf-8'))['recall']}
    held = [row for row in recall_prompts(tokenizer, transfer, extended=True) if row['length'] == 'extended']
    legacy = [row for row in recall_prompts(tokenizer, scenarios, extended=True) if row['length'] != 'short']
    excluded = {row['question'] for row in (*held, *legacy)}
    excluded.update(question for case in scenarios['open'] for question, _, _ in case['turns'])
    training = json.loads(DATA.read_text(encoding='utf-8'))
    examples = dialogue_data(training, held=False, count=12)
    if excluded & {question for item in examples for question in
                   (item['first'][0], item['update'][0], item['question'])}:
        raise ValueError('Training text overlaps an evaluation question')
    requests = [(ids, labels, fields) for pair in training_pairs(tokenizer, training, examples, field_weight=1.)
                for ids, labels, _, _, fields in pair]
    replay = replay_examples(tokenizer, FOUNDATIONS, excluded)

    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    torch.cuda.empty_cache()
    base.lm_head.to(base.device)
    individual = frozen_digest(runtime.organism, runtime.state)
    candidates = load_recurrent_replacements(decoder, args.candidates, protected[args.checkpoint])
    block = base.model.layers[-1]
    if type(block) is not SPORAResidualBlock or not block.ordinary_only:
        raise ValueError('Expected an ordinary dialogue residual on the final decoder layer')
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {str(path): digest for path, digest in protected.items()}
    with ExitStack() as installed:
        install(installed, candidates)
        print(f'Capturing final-layer features for {len(requests)} answers and {len(replay)} replay rows', flush=True)
        features = capture(decoder, block, requests)
        replay_features = capture(decoder, block, replay)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        trace = fit(base, block, features, replay_features, args.steps)
        weights = args.output / 'readout.pt'
        atomic_save({'sources': sources, 'individual_sha256': individual, 'steps': args.steps,
                     'down': block.down.weight.detach().cpu(), 'up': block.up.weight.detach().cpu()}, weights)
        print('Saved final-layer dialogue residual', flush=True)
        report = {'training_peak_bytes': torch.cuda.max_memory_allocated(), 'trace': trace,
                  'answer_requests': len(features), 'replay_requests': len(replay_features),
                  'weight_sha256': sha(weights)}
        report['transfer'] = run_dialogue(decoder, tokenizer, held, [])
        report['legacy'] = run_dialogue(decoder, tokenizer, legacy, scenarios['open'])
        report['memory'] = evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))

    if (individual != frozen_digest(runtime.organism, runtime.state)
            or any(sha(path) != digest for path, digest in protected.items())):
        raise ValueError('Readout training changed a protected input or the individual')
    store_json(args.output / 'result.json', {**report, 'sources': sources,
                                            'individual_sha256': individual, 'seconds': perf_counter() - started})
    print('Transfer', report['transfer']['recall_passed'], 'legacy', report['legacy']['recall_passed'],
          'memory', report['memory']['summary'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('candidates', nargs=2, type=Path)
    parser.add_argument('--steps', type=int, default=192)
    options = parser.parse_args()
    if options.output.exists() or options.steps < 4:
        parser.error('Choose a new output directory and at least four updates')
    main(options)
