"""Fit a native block with the existing migrated layers frozen."""

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
from train_awake_absolute_attention import replay_requests
from train_awake_dialogue_recovery import CORPUS, dialogue_cases
from train_awake_feedforward_replay import collect_replay, paired_targets
from train_awake_native_outputs import cache_language_targets, train
from train_awake_partial_layer import TRAIN_FILES, collect, corpus
from train_awake_replacement_replay import load_layer_pair


INDEX = 24


def initialize(decoder, tokenizer, teacher, texts, replay, limit, index):
    norms = []
    for reference in (teacher.input_layernorm, teacher.post_attention_layernorm):
        features, targets = collect(decoder, tokenizer, texts, reference)
        norms.append(fit_norm(reference, features, targets))
    features, targets = collect(decoder, tokenizer, texts, teacher.mlp)
    replay_rows = [(ids, [tokenizer.eos_token_id], kwargs, None) for ids, kwargs in replay]
    replay_features, replay_targets = collect_replay(decoder, teacher.mlp, replay_rows)
    feedforward = initialize_from_projections(teacher.mlp, features)
    fit_readout(feedforward, torch.cat((features, replay_features)), torch.cat((targets, replay_targets)),
                None, None, regularization=.08)
    attention = AbsoluteAttention(teacher.self_attn, limit)
    return assemble(decoder.base, attention, norms, feedforward, index)


def main(args):
    fixed = {12: args.layer12, 18: args.layer18}
    if args.layer24 is not None:
        fixed[24] = args.layer24
    paths = (SOURCE, MODEL / 'model.safetensors', MODEL / 'tokenizer.json', MODEL / 'tokenizer_config.json',
             *fixed.values(), CORPUS, *TRAIN_FILES,
             *(args.ceval / f'{subject}.parquet' for subject in args.subjects))
    protected = {path: sha(path) for path in paths}
    provenance = dict(checkpoint_digest=protected[SOURCE], base_digest=protected[MODEL / 'model.safetensors'])
    started = time.perf_counter()
    torch.manual_seed(42117)
    decoder, tokenizer, _, _, organism, _ = load_layer_pair()
    layers = decoder.base.model.layers
    for index, path in fixed.items():
        digest = read(path.parent / 'result.json')['candidate_sha256']
        layers[index] = load_native_block(path, decoder.base, index, candidate_digest=digest, **provenance)
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

    score('four_blocks' if args.layer24 is not None else 'three_blocks')
    teacher = layers[args.layer]
    texts = corpus()
    requests = [(tokenizer.encode(text, add_special_tokens=False)[:192], {}) for text in texts]
    replay = replay_requests(organism, tokenizer) + replay_requests(organism, tokenizer, value_offset=2)
    block = initialize(decoder, tokenizer, teacher, texts[8:], replay, limit, args.layer)
    samples = capture(decoder, teacher, requests)
    held, ordinary = samples[:8], samples[8:]
    remembered = capture(decoder, teacher, replay)
    before = local_error(block, held)
    fit(block, ordinary, remembered, args.local_steps)
    after = local_error(block, held)
    print('Held block residual error', before, after, flush=True)

    language = cache_language_targets(decoder, dialogue_cases(tokenizer)[::2])
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
                                            local_error=dict(initial=before, aligned=after), losses=losses,
                                            seconds=time.perf_counter() - started,
                                            peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                                            sources={str(path): digest for path, digest in protected.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layer12', type=Path, required=True)
    parser.add_argument('--layer18', type=Path, required=True)
    parser.add_argument('--layer24', type=Path, help='Freeze the verified native block at layer 24')
    parser.add_argument('--layer', type=int, choices=(9, INDEX), default=INDEX)
    parser.add_argument('--ceval', type=Path, required=True)
    parser.add_argument('--subjects', nargs='+', default=SUBJECTS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--local-steps', type=int, default=64)
    parser.add_argument('--output-steps', type=int, default=96)
    parser.add_argument('--reference-weight', type=float, default=.25,
                        help='Weight on reference answers in addition to teacher distribution matching')
    args = parser.parse_args()
    if min(args.local_steps, args.output_steps) < 1:
        parser.error('Training steps must be positive')
    if not math.isfinite(args.reference_weight) or args.reference_weight < 0.:
        parser.error('Reference weight must be finite and nonnegative')
    if (args.layer == 9) != (args.layer24 is not None):
        parser.error('Layer 9 requires --layer24, which is the training target in the default mode')
    args.output.mkdir()
    main(args)
