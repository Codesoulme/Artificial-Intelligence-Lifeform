"""Fit or compare native blocks with the existing migrated layers frozen."""

import argparse
import math
import time
from pathlib import Path

from audit_awake_block_components import fit_norm
from audit_awake_gelu_initialization import fit_readout, initialize_from_projections
from audit_awake_multiturn_layers import ceval_questions, conversation, encode, score_ceval, score_memory
from audit_awake_native_block import assemble, capture, fit, local_error
from audit_awake_native_holdout import CASES, FRESH_MEMORY, MEMORY, SUBJECTS
from audit_awake_native_interpolation import count_switches, paired_states
from deepseek_output_aligned_memory import MODEL, SOURCE, read, sha, torch, write
from spora.language_training import atomic_save
from spora.pretrained_attention import AbsoluteAttention
from spora.structure_transfer import load_native_block
from train_awake_absolute_attention import capture as capture_attention
from train_awake_absolute_attention import fit_attention, relative_error as attention_error, replay_requests
from train_awake_dialogue_recovery import CORPUS, dialogue_cases
from train_awake_feedforward_replay import collect_replay, fit as fit_feedforward, paired_targets
from train_awake_native_outputs import cache_language_targets, train
from train_awake_partial_layer import TRAIN_FILES, collect, corpus, relative_mse
from train_awake_replacement_replay import generate, load_layer_pair


INDEX = 24
SIXTH_BLOCK_LAYERS = (15, 25, 27)
SEVENTH_BLOCK_LAYERS = (25, 26)
ADDITIONAL_MEMORY = ((29, '许洲', 3, 0), (30, '苏苒', 0, 5), (31, '夏宜', 6, 2))
TRAIN_HISTORY = ((5, '明珂', 3, 6), (6, '茹川', 2, 0), (7, '洛源', 3, 7), (8, '青洵', 2, 5))


def history_training_pairs(organism, tokenizer):
    states = paired_states(organism, TRAIN_HISTORY, offset=1)
    prompts = [encode(tokenizer, [dict(role='user', content=f'更正以前，{name}选的饮品是什么？')])
               for _, name, _, _ in TRAIN_HISTORY]
    return tuple([(prompt, kwargs, value)
                  for prompt, (_, value, kwargs, _) in zip(prompts, state[1::2], strict=True)]
                 for state in states)


def initialize(decoder, tokenizer, teacher, texts, replay, feedforward_steps):
    training, held = texts[8:], texts[:8]
    norms = []
    for reference in (teacher.input_layernorm, teacher.post_attention_layernorm):
        features, targets = collect(decoder, tokenizer, training, reference)
        norms.append(fit_norm(reference, features, targets))
    features, targets = collect(decoder, tokenizer, training, teacher.mlp)
    replay_rows = [(ids, [tokenizer.eos_token_id], kwargs, None) for ids, kwargs in replay]
    replay_features, replay_targets = collect_replay(decoder, teacher.mlp, replay_rows)
    feedforward = initialize_from_projections(teacher.mlp, features)
    fit_readout(feedforward, torch.cat((features, replay_features)), torch.cat((targets, replay_targets)),
                None, None, regularization=.08)
    errors = {}
    if feedforward_steps:
        held_x, held_y = collect(decoder, tokenizer, held, teacher.mlp)
        errors['feedforward_initial'] = relative_mse(feedforward, held_x, held_y)
        fit_feedforward(feedforward, (features, targets), (replay_features, replay_targets), feedforward_steps)
        errors['feedforward_fitted'] = relative_mse(feedforward, held_x, held_y)
        print('Held feed-forward errors', errors, flush=True)
    return norms, feedforward.requires_grad_(False).eval(), errors


def align_block(decoder, teacher, block, requests, replay, steps):
    samples = capture(decoder, teacher, requests)
    held, ordinary = samples[:8], samples[8:]
    remembered = capture(decoder, teacher, replay)
    before = local_error(block, held)
    fit(block, ordinary, remembered, steps)
    after = local_error(block, held)
    print('Held block residual error', before, after, flush=True)
    return dict(initial=before, aligned=after)


def main(args):
    fixed = {12: args.layer12, 18: args.layer18}
    if args.layer24 is not None:
        fixed[24] = args.layer24
    if args.layer21 is not None:
        fixed[21] = args.layer21
    if args.layer27 is not None:
        fixed[27] = args.layer27
    paths = (SOURCE, MODEL / 'model.safetensors', MODEL / 'tokenizer.json', MODEL / 'tokenizer_config.json',
             *fixed.values(), *args.compare_candidates,
             *(args.ceval / f'{subject}.parquet' for subject in args.subjects))
    if not args.compare_candidates:
        paths = (*paths, CORPUS, *TRAIN_FILES)
    if args.resume_block is not None:
        paths = (*paths, args.resume_block)
    protected = {path: sha(path) for path in paths}
    provenance = dict(checkpoint_digest=protected[SOURCE], base_digest=protected[MODEL / 'model.safetensors'])
    started = time.perf_counter()
    torch.manual_seed(42117)
    decoder, tokenizer, _, _, organism, _ = load_layer_pair()
    layers = decoder.base.model.layers
    loaded = {}
    for index, path in fixed.items():
        report = read(path.parent / 'result.json')
        if index in (21, 27) and report.get('fixed_layers') != loaded:
            raise ValueError(f'Layer {index} requires its training backbone')
        digest = report['candidate_sha256']
        layers[index] = load_native_block(path, decoder.base, index, candidate_digest=digest, **provenance)
        loaded[str(index)] = digest
    limit = min(layers[index].max_length for index in (6, *fixed))
    questions = list(ceval_questions(args.ceval, tokenizer, args.subjects))
    memory_suites = {'earlier': paired_states(organism, MEMORY),
                     'fresh': paired_states(organism, FRESH_MEMORY, offset=5)}
    if args.resume_block is not None or args.layer27 is not None:
        memory_suites['additional'] = paired_states(organism, ADDITIONAL_MEMORY, offset=3)
    results = {}

    def score(name):
        choices = score_ceval(decoder, questions, limit)
        dialogues = [conversation(decoder, tokenizer, case, limit, decoding=dict(limit=48)) for case in CASES]
        summary = dict(ceval_correct=sum(row['choice'] == row['expected'] for row in choices),
                       ceval_total=sum(row['choice'] is not None for row in choices),
                       dialogue_correct=sum(turn['correct'] for group in dialogues for turn in group),
                       dialogue_eos=sum(turn['eos'] for group in dialogues for turn in group))
        memory = {}
        for suite, (first, shifted) in memory_suites.items():
            earlier = score_memory(decoder, tokenizer, first)
            alternate = score_memory(decoder, tokenizer, shifted)
            memory[suite] = dict(first=earlier, alternate=alternate)
            summary[f'{suite}_read'] = sum(row['correct'] for row in earlier)
            summary[f'{suite}_switches'] = count_switches(earlier, alternate)
        results[name] = dict(summary=summary, ceval=choices, dialogue=dialogues, memory=memory)
        print(name, summary, flush=True)

    score({2: 'three_blocks', 3: 'four_blocks', 4: 'five_blocks', 5: 'six_blocks'}[len(fixed)])
    if args.compare_candidates:
        candidates = {}
        fixed_digests = {str(index): protected[path] for index, path in fixed.items()}
        if args.layer27 is not None:
            allowed_layers = SEVENTH_BLOCK_LAYERS
        elif args.layer21 is not None:
            allowed_layers = SIXTH_BLOCK_LAYERS
        else:
            allowed_layers = (9, 21)
        for path in args.compare_candidates:
            report = read(path.parent / 'result.json')
            index = report['layer']
            if index not in allowed_layers or index in candidates or report['fixed_layers'] != fixed_digests:
                raise ValueError('Expected distinct replacements trained on the selected backbone')
            candidate = load_native_block(path, decoder.base, index, candidate_digest=report['candidate_sha256'],
                                          **provenance)
            if candidate.max_length < limit:
                raise ValueError('Candidate does not cover the baseline position range')
            original = layers[index]
            layers[index] = candidate
            try:
                score(f'layer_{index}')
            finally:
                layers[index] = original
            candidates[index] = protected[path]
            del candidate
        if any(sha(path) != digest for path, digest in protected.items()):
            raise ValueError('A protected input changed')
        write(args.output / 'result.json', dict(results=results, candidates=candidates,
                                                fixed_layers=fixed_digests, subjects=args.subjects,
                                                position_limit=limit, seconds=time.perf_counter() - started,
                                                peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                                                sources={str(path): digest for path, digest in protected.items()}))
        return
    texts = corpus()
    requests = [(tokenizer.encode(text, add_special_tokens=False)[:192], {}) for text in texts]
    replay = replay_requests(organism, tokenizer) + replay_requests(organism, tokenizer, value_offset=2)
    if args.screen_layers:
        errors = {}
        for index in args.screen_layers:
            torch.manual_seed(42117)
            teacher = layers[index]
            norms, feedforward, _ = initialize(decoder, tokenizer, teacher, texts, replay, 0)
            attention = AbsoluteAttention(teacher.self_attn, limit).requires_grad_(False).eval()
            block = assemble(decoder.base, attention, norms, feedforward, index)
            errors[index] = align_block(decoder, teacher, block, requests, replay, args.local_steps)
            layers[index] = block
            try:
                score(f'layer_{index}')
            finally:
                layers[index] = teacher
        if any(sha(path) != digest for path, digest in protected.items()):
            raise ValueError('A protected input changed')
        write(args.output / 'result.json', dict(results=results, local_error=errors, local_steps=args.local_steps,
                                                fixed_layers=loaded, subjects=args.subjects, position_limit=limit,
                                                seconds=time.perf_counter() - started,
                                                peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                                                sources={str(path): digest for path, digest in protected.items()}))
        return
    teacher = layers[args.layer]
    component_errors = {}
    if args.resume_block is None:
        norms, feedforward, component_errors = initialize(decoder, tokenizer, teacher, texts, replay,
                                                          args.feedforward_steps)
        attention = AbsoluteAttention(teacher.self_attn, limit).requires_grad_(False).eval()
    if args.attention_steps:
        source_attention = teacher.self_attn
        attention_rows = capture_attention(decoder, source_attention, requests)
        replay_rows = capture_attention(decoder, source_attention, replay)
        attention.float()
        untrained = attention_error(attention, attention_rows[:8])
        fit_attention(attention, attention_rows[8:], replay_rows, args.attention_steps)
        component_errors.update(attention_initial=untrained,
                                attention_fitted=attention_error(attention, attention_rows[:8]))
        if not args.feedforward_steps:
            held_x, held_y = collect(decoder, tokenizer, texts[:8], teacher.mlp)
            component_errors['feedforward'] = relative_mse(feedforward, held_x, held_y)
        print('Held component errors', component_errors, flush=True)
        teacher.self_attn = attention
        try:
            score('attention_only')
        finally:
            teacher.self_attn = source_attention
    if args.attention_steps or args.feedforward_steps:
        source_mlp = teacher.mlp
        teacher.mlp = feedforward
        try:
            score('feedforward_only')
        finally:
            teacher.mlp = source_mlp
    block_error = None
    if args.resume_block is None:
        block = assemble(decoder.base, attention, norms, feedforward, args.layer)
        block_error = align_block(decoder, teacher, block, requests, replay, args.local_steps)
    else:
        report = read(args.resume_block.parent / 'result.json')
        if report['fixed_layers'] != loaded:
            raise ValueError('The starting block requires its training backbone')
        block = load_native_block(args.resume_block, decoder.base, args.layer,
                                  candidate_digest=report['candidate_sha256'], **provenance)
        if block.max_length != limit:
            raise ValueError('Starting block position range differs from the backbone limit')

    examples = dialogue_cases(tokenizer)[::2]
    rollout_stats = None
    if args.student_rollout_tokens:
        layers[args.layer] = block
        continuations = []
        for question, prompt, reference in examples:
            budget = min(args.student_rollout_tokens, len(reference), limit - len(prompt))
            tail = generate(decoder, tokenizer, prompt, budget)
            continuations.append((question, prompt, tail))
        layers[args.layer] = teacher
        rollout_stats = dict(tokens=sum(len(tail) for _, _, tail in continuations),
                             eos=sum(tail[-1] == tokenizer.eos_token_id for _, _, tail in continuations))
        print('Student continuations', rollout_stats, flush=True)
        examples = continuations
    language = cache_language_targets(decoder, examples)
    history_pairs = history_training_pairs(organism, tokenizer) if args.history_replay else ()
    memory = paired_targets(decoder, tokenizer, organism, history_pairs=history_pairs)
    layers[args.layer] = block
    score('local_alignment' if args.resume_block is None else 'starting_candidate')
    losses, trained_parameters = train(decoder, block, language, memory, args.output_steps, contrast_weight=.5,
                                      reference_weight=args.reference_weight,
                                      first_token_weight=args.first_token_weight)
    score('output_alignment')

    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError('A protected input changed')
    path = args.output / f'layer-{args.layer}.pt'
    metadata = dict(format='spora-qwen-layer-replacement-candidate-v1', layer=args.layer, width=block.width,
                    heads=block.block.heads, max_length=limit, base_weight_sha256=provenance['base_digest'],
                    source_sha256=provenance['checkpoint_digest'])
    atomic_save(dict(metadata=metadata, state={key: value.cpu() for key, value in block.state_dict().items()}), path)
    digest = sha(path)
    layers[args.layer] = load_native_block(path, decoder.base, args.layer, candidate_digest=digest, **provenance)
    score('reloaded')
    if results['reloaded'] != results['output_alignment']:
        raise ValueError('Reloaded block changed evaluation results')
    write(args.output / 'result.json', dict(results=results, candidate_sha256=digest, layer=args.layer,
                                            fixed_layers={index: protected[path] for index, path in fixed.items()},
                                            subjects=args.subjects, position_limit=limit,
                                            local_steps=args.local_steps if args.resume_block is None else 0,
                                            output_steps=args.output_steps, trained_parameters=trained_parameters,
                                            reference_weight=args.reference_weight,
                                            resume_block_sha256=protected.get(args.resume_block),
                                            first_token_weight=args.first_token_weight,
                                            history_replay_pairs=len(history_pairs[0]) if history_pairs else 0,
                                            attention_steps=args.attention_steps, component_errors=component_errors,
                                            feedforward_steps=args.feedforward_steps,
                                            student_rollout_tokens=args.student_rollout_tokens,
                                            rollout_stats=rollout_stats,
                                            local_error=block_error, losses=losses,
                                            seconds=time.perf_counter() - started,
                                            peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                                            sources={str(path): digest for path, digest in protected.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layer12', type=Path, required=True)
    parser.add_argument('--layer18', type=Path, required=True)
    parser.add_argument('--layer24', type=Path, help='Freeze the verified native block at layer 24')
    parser.add_argument('--layer21', type=Path, help='Freeze layer 21 when fitting a sixth block')
    parser.add_argument('--layer27', type=Path, help='Freeze layer 27 when fitting a seventh block')
    parser.add_argument('--layer', type=int, choices=(9, 15, 21, INDEX, 25, 26, 27), default=INDEX)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--compare-candidates', type=Path, nargs='+', default=(),
                        help='Score frozen checkpoints independently against the selected backbone')
    mode.add_argument('--screen-layers', type=int, choices=(15, 25, 26, 27), nargs='+',
                      help='Compare these layers after local alignment without output distillation')
    mode.add_argument('--resume-block', type=Path, help='Continue output alignment from an existing native block')
    parser.add_argument('--ceval', type=Path, required=True)
    parser.add_argument('--subjects', nargs='+', default=SUBJECTS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--local-steps', type=int, default=64)
    parser.add_argument('--output-steps', type=int, default=96)
    parser.add_argument('--first-token-weight', type=float, default=0.,
                        help='Weight on the memory first-token margin during resumed training')
    parser.add_argument('--history-replay', action='store_true',
                        help='Add paired historical values on training-only addresses')
    parser.add_argument('--attention-steps', type=int, default=0,
                        help='Fit attention before whole-block alignment and score individual transplants')
    parser.add_argument('--feedforward-steps', type=int, default=0,
                        help='Refine the initialized GELU weights on ordinary and memory activations')
    parser.add_argument('--reference-weight', type=float, default=.25,
                        help='Weight on reference answers in addition to teacher distribution matching')
    parser.add_argument('--student-rollout-tokens', type=int, default=0,
                        help='Collect fixed student continuations up to this many tokens instead of reference answers')
    args = parser.parse_args()
    if min(args.local_steps, args.output_steps) < 1:
        parser.error('Training steps must be positive')
    if args.attention_steps < 0 or args.attention_steps == 1:
        parser.error('Use zero or at least two attention steps')
    if args.feedforward_steps < 0:
        parser.error('Feed-forward steps must be nonnegative')
    if not math.isfinite(args.reference_weight) or args.reference_weight < 0.:
        parser.error('Reference weight must be finite and nonnegative')
    if not math.isfinite(args.first_token_weight) or args.first_token_weight < 0.:
        parser.error('First-token weight must be finite and nonnegative')
    if (args.first_token_weight or args.history_replay) and (
            args.compare_candidates or args.screen_layers or (args.resume_block is None and args.layer27 is None)):
        parser.error('--first-token-weight and --history-replay require resumed or seventh-block training')
    if args.resume_block is not None and (args.layer != 27 or args.attention_steps or args.feedforward_steps):
        parser.error('--resume-block requires layer 27 without component pretraining')
    if args.student_rollout_tokens < 0 or (args.student_rollout_tokens and args.reference_weight):
        parser.error('Student continuations require a positive token cap and --reference-weight 0')
    if args.layer27 is not None and (args.layer21 is None or args.layer24 is None):
        parser.error('--layer27 requires --layer21 and --layer24')
    if args.screen_layers:
        screen_layers = SEVENTH_BLOCK_LAYERS if args.layer27 is not None else SIXTH_BLOCK_LAYERS
        if (args.layer21 is None or args.layer24 is None
                or len(set(args.screen_layers)) != len(args.screen_layers)
                or any(index not in screen_layers for index in args.screen_layers)):
            parser.error('--screen-layers requires distinct replacement layers for the selected backbone')
    elif args.compare_candidates:
        if args.layer24 is None:
            parser.error('--compare-candidates requires --layer24')
    else:
        if ((args.layer != INDEX) != (args.layer24 is not None)
                or (args.layer in (15, 25, 26, 27)) != (args.layer21 is not None)):
            parser.error('Fifth and later blocks require --layer24. Sixth and later blocks also require --layer21.')
        if (args.layer27 is not None and args.layer not in SEVENTH_BLOCK_LAYERS
                or args.layer == 26 and args.layer27 is None):
            parser.error('The seven-block experiment fits layer 25 or 26 with --layer27.')
    args.output.mkdir()
    main(args)
