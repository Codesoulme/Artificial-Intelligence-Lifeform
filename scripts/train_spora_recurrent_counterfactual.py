"""Train two recurrent layers to follow a changed value through long dialogue."""

import argparse
from contextlib import ExitStack, contextmanager, nullcontext
from functools import partial
import json
import math
import os
from pathlib import Path
import random
from tempfile import NamedTemporaryFile
from time import perf_counter
from unittest.mock import patch

from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from scripts.train_spora_recurrent_long_alignment import (
    CORPUS, DATA, HELD, MANIFEST, MODEL, SOURCE, AutoTokenizer, dialogue_data, encode, evaluate, extend_positions,
    frozen_digest, load_full_streamed, load_integrated_runtime, load_recurrent_replacements, material, recall_prompts,
    records, replace, run_dialogue, sha, torch, training_history, training_sequences,
)
from scripts.train_spora_dialogue_working_state import annotated
from spora.language_training import atomic_save
from spora.research_dialogue_working_state import FIELDS
from spora.research_recurrent_decode import restrict_full_attention


SEED = 50721


def store_json(path, value):
    if path.exists():
        raise FileExistsError(path)
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.rename(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def alternate(example, source, rng):
    text, bounds = example['update']
    edits = example['changes']
    swaps = {field: rng.choice([value for value in source['values']['train'][field]
                                if value not in (example['initial'][field], current)])
             for field, current in edits.items()}
    spans = sorted((bounds[FIELDS.index(field)], field) for field in swaps)
    for (start, end), field in reversed(spans):
        if not start or text[start - 1:end] != edits[field]:
            raise ValueError('Update annotation did not identify the corrected value')
        text = text[:start - 1] + swaps[field] + text[end:]
    current = example['initial'] | swaps
    answer = source['kinds'][example['kind']]['answer'].format(**current)
    return {**example, 'update': (text, None), 'answer': answer, 'current': current, 'changes': swaps}


def field_token_groups(tokenizer, example, template, expected_ids):
    answer, boundaries = annotated(template, example['current'])
    if answer != example['answer']:
        raise ValueError('Field annotations differ from the supervised answer')
    encoded = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
    if encoded.input_ids != expected_ids:
        raise ValueError('Field annotations use a different answer tokenization')
    groups = tuple(tuple(i for i, (left, right) in enumerate(encoded.offset_mapping)
                         if left < right and left < stop and right >= start)
                   for start, stop in boundaries if start)
    if not groups or not all(groups):
        raise ValueError('The target answer lacks a token span for a requested field')
    return groups


def training_pairs(tokenizer, source, examples, *, field_weight=0.):
    rng = random.Random(SEED)
    alternate_examples = [alternate(row, source, rng) for row in examples]
    fillers = material(tokenizer, CORPUS, 40)
    originals = training_sequences(tokenizer, examples, fillers)
    swaps = training_sequences(tokenizer, alternate_examples, fillers)
    pairs = []
    for example, other, (first, first_end), (second, second_end) in zip(
        examples, alternate_examples, originals, swaps, strict=True
    ):
        first_answer = [*first[first_end + 1:], tokenizer.eos_token_id]
        second_answer = [*second[second_end + 1:], tokenizer.eos_token_id]
        difference = next((i for i, (a, b) in enumerate(zip(first_answer, second_answer)) if a != b), None)
        if difference is None or first[:first_end + 1] == second[:second_end + 1]:
            raise ValueError('Counterfactual histories must change the correct answer at a shared prefix')
        template = source['kinds'][example['kind']]['answer']
        first_fields = field_token_groups(tokenizer, example, template, first_answer[:-1]) if field_weight else ()
        second_fields = field_token_groups(tokenizer, other, template, second_answer[:-1]) if field_weight else ()
        pairs.append(((first, first_answer, difference, second_answer[difference], first_fields),
                      (second, second_answer, difference, first_answer[difference], second_fields)))
    return pairs


def segment_boundaries(tokenizer, source, examples, pairs, block):
    rng = random.Random(SEED)
    fillers = material(tokenizer, CORPUS, 40)
    cuts = []
    for index, (example, pair) in enumerate(zip(examples, pairs, strict=True)):
        counterfactual = alternate(example, source, rng)
        ends = []
        for scenario, (ids, *_) in zip((example, counterfactual), pair, strict=True):
            history = training_history(scenario, fillers, index)
            last_event = next(i for i, turn in enumerate(history)
                              if turn['role'] == 'user' and turn['content'] == scenario['update'][0])
            rendered = tokenizer.apply_chat_template(history[:last_event + 2], tokenize=False,
                                                     add_generation_prompt=False)
            prefix = tokenizer.encode(rendered, add_special_tokens=False)
            if ids[:len(prefix)] != prefix:
                raise ValueError('The separated history must preserve the original prompt prefix')
            ends.append(len(prefix))
        padded_end = max(ends) + 2 * block
        cut = ((padded_end + block - 1) // block) * block
        for scenario, (ids, answer, *_) in zip((example, counterfactual), pair, strict=True):
            prompt_end = len(ids) - len(answer) + 1
            suffix = tokenizer.decode(ids[cut:prompt_end], skip_special_tokens=True)
            if prompt_end - cut < 96 or any(str(value) in suffix for value in scenario['current'].values()):
                cut = None
                break
        cuts.append(cut)
    if not any(cut is not None for cut in cuts):
        raise ValueError('Expected long histories with both facts before a segment boundary')
    return cuts


def evaluation_pairs(tokenizer, source):
    base = [row for row in recall_prompts(tokenizer, source, extended=True) if row['length'] != 'short']
    pairs = []
    for row in base:
        if row['variant'] != 0:
            continue
        alternate_row = next(other for other in base if other['case'] == row['case']
                             and other['length'] == row['length'] and other['variant'] == 1)
        updated = row['values']['new']
        target = alternate_row['values']['new']
        history = [turn.copy() for turn in row['history']]
        turn = history[2]
        if turn['content'].count(updated) != 1:
            raise ValueError('The held counterfactual must change just one update value')
        turn['content'] = turn['content'].replace(updated, target)
        ids = encode(tokenizer, [*history, {'role': 'user', 'content': row['question']}])
        counterfactual = {**row, 'variant': 2, 'history': history, 'input_tokens': len(ids),
                          'values': {**row['values'], 'new': target}, 'forbidden_update': updated}
        pairs.extend(({**row, 'forbidden_update': target}, counterfactual))
    return base, pairs


def loss_for(decoder, request, field_weight):
    ids, answer, difference, foil, fields = request
    device = decoder.base.device
    tokens = torch.tensor([ids], device=device)
    logits = decoder(input_ids=tokens, use_cache=False, logits_to_keep=len(answer)).logits[0].float()
    target = torch.tensor(answer, device=device)
    if field_weight:
        losses = F.cross_entropy(logits, target, reduction='none')
        balanced = torch.stack([losses[list(positions)].mean() for positions in fields]).mean()
        exact = (losses.mean() + field_weight * balanced) / (1 + field_weight)
    else:
        exact = F.cross_entropy(logits, target)
    margin = logits[difference, target[difference]] - logits[difference, foil]
    return exact + .3 * F.softplus(1. - margin), margin


@contextmanager
def checkpoint_layers(decoder, first):
    with ExitStack() as stack:
        for layer in decoder.base.model.layers[first:]:
            wrapped = partial(checkpoint, layer.forward, use_reentrant=False, preserve_rng_state=False)
            stack.enter_context(patch.object(layer, 'forward', wrapped))
        yield


def fit(decoder, candidates, pairs, steps, learning_rate, field_weight=0., *, boundaries=None):
    params = [param for _, replacement in candidates.values() for param in replacement.core.parameters()]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.)
    rng = random.Random(SEED)
    order = [max(range(len(pairs)), key=lambda i: max(len(arm[0]) for arm in pairs[i]))]
    trace = []
    eligible = 0
    for step in range(steps):
        if not order:
            order = list(range(len(pairs)))
            rng.shuffle(order)
        pair_index = order.pop()
        batch = pairs[pair_index]
        optimizer.zero_grad(set_to_none=True)
        observations = []
        boundary = boundaries[pair_index] if boundaries is not None else None
        eligible += int(boundary is not None)
        masked = boundary is not None and eligible % 2 == 1
        context = restrict_full_attention(decoder, boundary, candidates) if masked else nullcontext()
        with context:
            for arm in batch:
                loss, margin = loss_for(decoder, arm, field_weight)
                (loss * .5).backward()
                observations.append((float(loss.detach()), float(margin.detach())))
        gradient = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        row = {'step': step + 1, 'pair_index': pair_index, 'tokens': [len(arm[0]) for arm in batch],
               'loss': sum(obs[0] for obs in observations) / len(observations), 'margins': [obs[1] for obs in observations],
               'gradient': float(gradient)}
        if masked:
            row['segment_boundary'] = boundary
        trace.append(row)
        if step == 0 or (step + 1) % 8 == 0 or step + 1 == steps:
            print('Training', row, flush=True)
    return trace


def assess(decoder, tokenizer, pairs, open_cases):
    scored = run_dialogue(decoder, tokenizer, pairs, open_cases)
    values = {}
    for length in ('long', 'extended'):
        rows = [row for row in scored['recall'] if row['length'] == length]
        for row in rows:
            row['paired_pass'] = row['passed'] and row['forbidden_update'] not in row['answer']
        values[length] = {'answers': sum(row['paired_pass'] for row in rows),
                          'pairs': sum(first['paired_pass'] and second['paired_pass']
                                       for first, second in zip(rows[::2], rows[1::2], strict=True)),
                          'count': len(rows)}
    scored['counterfactual'] = values
    return scored


def install(stack, candidates):
    for block, replacement in candidates.values():
        stack.enter_context(replace(block, replacement))


def evaluate_candidate(runtime, tokenizer, held, regular, open_cases, transfer=()):
    decoder = runtime.decoder
    after = assess(decoder, tokenizer, held, open_cases)
    original_cases = run_dialogue(decoder, tokenizer, regular, [])
    memory = records(runtime.organism, HELD)
    memory_after = evaluate(decoder, tokenizer, runtime.organism, memory)
    result = {'paired_after': after, 'original_cases_after': original_cases, 'memory_after': memory_after}
    if transfer:
        result['transfer_after'] = run_dialogue(decoder, tokenizer, transfer, [])
    return result


def save_training(args, candidates, sources, checkpoint_sha, individual, trace, pair_count, peak):
    saved = {}
    for index, (_, replacement) in candidates.items():
        directory = args.output / f'layer-{index}'
        directory.mkdir()
        path = directory / 'attention.pt'
        core = replacement.core
        atomic_save({'format': 'spora-recurrent-attention-v1', 'layer': index, 'block': core.block,
                     'checkpoint_sha256': checkpoint_sha, 'update_rule': core.update_rule, 'core': core.state_dict(),
                     'history_projection': replacement.history_projection}, path)
        saved[str(index)] = sha(path)
        store_json(directory / 'comparison.json', {'layer': index, 'block': core.block, 'sources': sources,
                                                   'update_rule': core.update_rule,
                                                   'history_projection': replacement.history_projection,
                                                   'individual_sha256': individual, 'candidate_sha256': saved[str(index)]})
    training = {'sources': sources, 'individual_sha256': individual, 'training_pairs': pair_count, 'trace': trace,
                'steps': args.steps, 'learning_rate': args.learning_rate,
                'training_peak_bytes': peak, 'candidate_hashes': saved}
    if args.field_weight:
        training['field_weight'] = args.field_weight
    if args.segment_supervision:
        training['segment_supervision'] = {'masked_steps': sum('segment_boundary' in row for row in trace)}
    store_json(args.output / 'training.json', training)
    print('Saved recurrent weights for evaluation', flush=True)
    return training


def main(args):
    started = perf_counter()
    if not (args.verify or args.evaluate):
        args.output.mkdir(parents=True, exist_ok=False)
    if not args.verify and (args.output / 'result.json').exists():
        raise FileExistsError(args.output / 'result.json')
    inputs = [args.checkpoint, SOURCE, CORPUS, MANIFEST, DATA, args.reference]
    if args.transfer:
        inputs.append(args.transfer)
    inputs.extend(directory / name for directory in args.candidates for name in ('attention.pt', 'comparison.json'))
    protected = {path: sha(path) for path in inputs}
    if json.loads(MANIFEST.read_text(encoding='utf-8'))['files'][CORPUS.name] != protected[CORPUS]:
        raise ValueError('The training corpus changed')
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    training_data = json.loads(DATA.read_text(encoding='utf-8'))
    regular, held = evaluation_pairs(tokenizer, source)
    transfer = []
    if args.transfer:
        transfer_source = {**source, 'recall': json.loads(args.transfer.read_text(encoding='utf-8'))['recall']}
        transfer = [row for row in recall_prompts(tokenizer, transfer_source, extended=True)
                    if row['length'] == 'extended']
    examples = dialogue_data(training_data, held=False, count=16)
    questions = {case['question'] for case in examples}
    if questions & {row['question'] for row in (*held, *transfer)}:
        raise ValueError('A held question occurs in the training partition')
    base = load_full_streamed()
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    extend_positions(decoder, rotary, 2048)
    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    if (reference['individual_sha256'] != individual or reference['sources'][str(SOURCE)] != protected[SOURCE]
            or reference['sources'][str(args.checkpoint)] != protected[args.checkpoint]):
        raise ValueError('The comparison used a different checkpoint, individual or data')
    if any(reference['sources'][str(directory / 'attention.pt')] != protected[directory / 'attention.pt']
           for directory in args.candidates):
        raise ValueError('The comparison used different starting recurrent weights')
    candidates = load_recurrent_replacements(decoder, args.candidates, protected[args.checkpoint])
    sources = {str(path): digest for path, digest in protected.items()}
    if args.verify or args.evaluate:
        training = json.loads((args.output / 'training.json').read_text(encoding='utf-8'))
        if (training['sources'] != sources or training['individual_sha256'] != individual
                or training.get('field_weight', 0.) != args.field_weight
                or ('segment_supervision' in training) != args.segment_supervision):
            raise ValueError('The saved training used different inputs')
        directories = [args.output / f'layer-{index}' for index in candidates]
        for index, directory in zip(candidates, directories, strict=True):
            if sha(directory / 'attention.pt') != training['candidate_hashes'][str(index)]:
                raise ValueError('The saved recurrent weights changed')
        trained = load_recurrent_replacements(decoder, directories, protected[args.checkpoint])
    else:
        pairs = training_pairs(tokenizer, training_data, examples, field_weight=args.field_weight)
        boundaries = None
        if args.segment_supervision:
            blocks = {replacement.core.block for _, replacement in candidates.values()}
            if len(blocks) != 1:
                raise ValueError('Segment supervision requires the same recent block size in both branches')
            boundaries = segment_boundaries(tokenizer, training_data, examples, pairs, blocks.pop())
        trained = candidates
        with ExitStack() as installed:
            install(installed, trained)
            for _, replacement in trained.values():
                replacement.batched_training = True
                replacement.core.train().requires_grad_(True)
            torch.cuda.reset_peak_memory_stats()
            with checkpoint_layers(decoder, min(candidates) + 1):
                trace = fit(decoder, candidates, pairs, args.steps, args.learning_rate, args.field_weight,
                            boundaries=boundaries)
            peak = torch.cuda.max_memory_allocated()
        for _, replacement in trained.values():
            replacement.core.eval().requires_grad_(False)
        if (individual != frozen_digest(runtime.organism, runtime.state)
                or any(sha(path) != digest for path, digest in protected.items())):
            raise ValueError('A counterfactual input or individual changed during training')
        training = save_training(args, trained, sources, protected[args.checkpoint], individual, trace, len(pairs), peak)
    if args.verify:
        expected = json.loads((args.output / 'result.json').read_text(encoding='utf-8'))
        if any(expected[key] != value for key, value in training.items()):
            raise ValueError('The saved experiment used different training weights or inputs')
        with ExitStack() as installed:
            install(installed, trained)
            result = evaluate_candidate(runtime, tokenizer, held, regular, source['open'], transfer)
        if (any(expected[key] != value for key, value in result.items())
                or individual != frozen_digest(runtime.organism, runtime.state)
                or any(sha(path) != digest for path, digest in protected.items())):
            raise ValueError('Reload changed a generated answer, individual or input')
        store_json(args.output / 'reload.json', {'identical': True, 'paired': len(held), 'original': len(regular),
                                                'memory': result['memory_after']['summary']})
        print('Reloaded', len(held), 'paired and', len(regular), 'original answers', flush=True)
        return
    evaluation_path = args.output / 'candidate-evaluation.json'
    if evaluation_path.exists():
        evaluated = json.loads(evaluation_path.read_text(encoding='utf-8'))
        if evaluated['candidate_hashes'] != training['candidate_hashes']:
            raise ValueError('Partial evaluation used other weights')
        result = evaluated['result']
    else:
        with ExitStack() as installed:
            install(installed, trained)
            result = evaluate_candidate(runtime, tokenizer, held, regular, source['open'], transfer)
        store_json(evaluation_path, {'candidate_hashes': training['candidate_hashes'], 'result': result})
    print('Trained', result['paired_after']['counterfactual'], result['original_cases_after']['recall_passed'],
          result['memory_after']['summary'], flush=True)
    original = candidates if args.evaluate else load_recurrent_replacements(decoder, args.candidates, protected[args.checkpoint])
    with ExitStack() as installed:
        install(installed, original)
        before = assess(decoder, tokenizer, held, source['open'])
    print('Before', before['counterfactual'], flush=True)
    if (individual != frozen_digest(runtime.organism, runtime.state)
            or any(sha(path) != digest for path, digest in protected.items())):
        raise ValueError('A counterfactual input or individual changed')
    payload = {**training, 'seconds': perf_counter() - started, 'paired_before': before, **result}
    store_json(args.output / 'result.json', payload)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('candidates', nargs=2, type=Path)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=.0005)
    parser.add_argument('--field-weight', type=float, default=0.,
                        help='Mix equal weight per annotated answer field into the full-answer loss')
    parser.add_argument('--segment-supervision', action='store_true',
                        help='Restrict full attention on alternate eligible long training pairs')
    parser.add_argument('--transfer', type=Path, help='Optional additional long-dialogue evaluation scenarios')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--verify', action='store_true')
    modes.add_argument('--evaluate', action='store_true', help='Finish evaluation of already saved weights')
    args = parser.parse_args()
    if args.steps < 1 or not 0 < args.learning_rate <= .01:
        parser.error('Choose positive updates and a learning rate through .01')
    if args.field_weight < 0 or not math.isfinite(args.field_weight):
        parser.error('--field-weight must be finite and nonnegative')
    main(args)
