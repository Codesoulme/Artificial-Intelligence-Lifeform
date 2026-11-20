"""Fit or compare native blocks with the existing migrated layers frozen."""

import argparse
import math
import time
from pathlib import Path

from audit_awake_block_components import fit_norm
from audit_awake_gelu_initialization import fit_readout, initialize_from_projections
from audit_awake_multiturn_layers import ceval_questions, conversation, score_ceval, score_memory
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


def main(args):
    fixed = {12: args.layer12, 18: args.layer18}
    if args.layer24 is not None:
        fixed[24] = args.layer24
    if args.layer21 is not None:
        fixed[21] = args.layer21
    paths = (SOURCE, MODEL / 'model.safetensors', MODEL / 'tokenizer.json', MODEL / 'tokenizer_config.json',
             *fixed.values(), *args.compare_candidates,
             *(args.ceval / f'{subject}.parquet' for subject in args.subjects))
    if not args.compare_candidates:
        paths = (*paths, CORPUS, *TRAIN_FILES)
    protected = {path: sha(path) for path in paths}
    provenance = dict(checkpoint_digest=protected[SOURCE], base_digest=protected[MODEL / 'model.safetensors'])
    started = time.perf_counter()
    torch.manual_seed(42117)
    decoder, tokenizer, _, _, organism, _ = load_layer_pair()
    layers = decoder.base.model.layers
    loaded = {}
    for index, path in fixed.items():
        report = read(path.parent / 'result.json')
        if index == 21 and report.get('fixed_layers') != loaded:
            raise ValueError('Layer 21 requires its training backbone')
        digest = report['candidate_sha256']
        layers[index] = load_native_block(path, decoder.base, index, candidate_digest=digest, **provenance)
        loaded[str(index)] = digest
    limit = min(layers[index].max_length for index in (6, *fixed))
    questions = list(ceval_questions(args.ceval, tokenizer, args.subjects))
    memory_suites = {'earlier': paired_states(organism, MEMORY),
                     'fresh': paired_states(organism, FRESH_MEMORY, offset=5)}
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

    score({2: 'three_blocks', 3: 'four_blocks', 4: 'five_blocks'}[len(fixed)])
    if args.compare_candidates:
        candidates = {}
        fixed_digests = {str(index): protected[path] for index, path in fixed.items()}
        allowed_layers = (15,) if args.layer21 is not None else (9, 21)
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
    teacher = layers[args.layer]
    texts = corpus()
    requests = [(tokenizer.encode(text, add_special_tokens=False)[:192], {}) for text in texts]
    replay = replay_requests(organism, tokenizer) + replay_requests(organism, tokenizer, value_offset=2)
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
    block = assemble(decoder.base, attention, norms, feedforward, args.layer)
    samples = capture(decoder, teacher, requests)
    held, ordinary = samples[:8], samples[8:]
    remembered = capture(decoder, teacher, replay)
    before = local_error(block, held)
    fit(block, ordinary, remembered, args.local_steps)
    after = local_error(block, held)
    print('Held block residual error', before, after, flush=True)

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
    memory = paired_targets(decoder, tokenizer, organism)
    layers[args.layer] = block
    score('local_alignment')
    losses, trained_parameters = train(decoder, block, language, memory, args.output_steps, contrast_weight=.5,
                                      reference_weight=args.reference_weight)
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
                                            subjects=args.subjects, position_limit=limit, local_steps=args.local_steps,
                                            output_steps=args.output_steps, trained_parameters=trained_parameters,
                                            reference_weight=args.reference_weight,
                                            attention_steps=args.attention_steps, component_errors=component_errors,
                                            feedforward_steps=args.feedforward_steps,
                                            student_rollout_tokens=args.student_rollout_tokens,
                                            rollout_stats=rollout_stats,
                                            local_error=dict(initial=before, aligned=after), losses=losses,
                                            seconds=time.perf_counter() - started,
                                            peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                                            sources={str(path): digest for path, digest in protected.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layer12', type=Path, required=True)
    parser.add_argument('--layer18', type=Path, required=True)
    parser.add_argument('--layer24', type=Path, help='Freeze the verified native block at layer 24')
    parser.add_argument('--layer21', type=Path, help='Freeze the verified native block at layer 21 for layer 15 training')
    parser.add_argument('--layer', type=int, choices=(9, 15, 21, INDEX), default=INDEX)
    parser.add_argument('--compare-candidates', type=Path, nargs='+', default=(),
                        help='Score frozen checkpoints independently against the selected backbone')
    parser.add_argument('--ceval', type=Path, required=True)
    parser.add_argument('--subjects', nargs='+', default=SUBJECTS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--local-steps', type=int, default=64)
    parser.add_argument('--output-steps', type=int, default=96)
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
    if args.student_rollout_tokens < 0 or (args.student_rollout_tokens and args.reference_weight):
        parser.error('Student continuations require a positive token cap and --reference-weight 0')
    if args.compare_candidates:
        if args.layer24 is None:
            parser.error('--compare-candidates requires --layer24')
    elif ((args.layer != INDEX) != (args.layer24 is not None)
          or (args.layer == 15) != (args.layer21 is not None)):
        parser.error('Layers 9 and 21 require --layer24. Layer 15 requires --layer24 and --layer21.')
    args.output.mkdir()
    main(args)
