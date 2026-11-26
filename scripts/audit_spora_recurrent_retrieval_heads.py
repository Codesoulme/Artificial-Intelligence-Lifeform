"""Intervene on selected layer-24 heads to test precise retrieval in a recurrent layout."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_recurrent_attention import causal_teacher_mix
from scripts.benchmark_spora_recurrent_dialogue import DynamicCache, using
from scripts.train_spora_hybrid_hidden_recovery import (
    CORPUS, DATA, LAYERS, MANIFEST, MODEL, SOURCE, TARGET_LAYER, AutoTokenizer,
    dialogue_data, extend_positions, frozen_digest, gradient_sample_indices, load_full_streamed,
    load_integrated_runtime, load_recurrent_replacements, recall_prompts, run_dialogue, sha,
    store_json, training_pairs,
)


@contextmanager
def exact_heads(replacement, heads):
    """Experimental single-stream intervention with exact keys for selected heads only."""
    keys = values = None
    groups = replacement.source.num_key_value_groups
    kv_heads = sorted({head // groups for head in heads})
    head_to_kv = [kv_heads.index(head // groups) for head in heads]

    def restore(_core, inputs, kwargs, result):
        nonlocal keys, values
        query, key, value = inputs
        query, key, value = query[:, heads], key[:, kv_heads], value[:, kv_heads]
        prior = kwargs.get('state')
        if prior is None:
            keys, values = key, value
            exact = causal_teacher_mix(query, keys[:, head_to_kv], values[:, head_to_kv])
        else:
            if keys is None or keys.shape[-2] != prior.length:
                raise ValueError('Exact heads require the same append-only stream as the recurrent layer')
            keys = torch.cat((keys, key), dim=-2)
            values = torch.cat((values, value), dim=-2)
            exact = F.scaled_dot_product_attention(query, keys[:, head_to_kv], values[:, head_to_kv],
                                                   is_causal=False)
            if query.shape[-2] != 1:
                raise ValueError('The head intervention only supports single-token decoding after prefill')
        mixed, state = result
        mixed[:, heads] = exact.to(mixed.dtype)
        return mixed, state

    handle = replacement.core.register_forward_hook(restore, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def margins(decoder, pairs):
    rows = []
    for pair in pairs:
        margins_for_pair = []
        for ids, answer, difference, foil, _ in pair:
            # At the first fork, both answers share the same prefix and neither value is teacher-forced.
            prefix = ids[:len(ids) - len(answer) + 1 + difference]
            tokens = torch.tensor([prefix], device=decoder.base.device)
            cache = DynamicCache(config=decoder.base.config)
            logits = decoder(input_ids=tokens, past_key_values=cache, use_cache=True,
                             logits_to_keep=1).logits[0, -1].float()
            margins_for_pair.append(float(logits[answer[difference]] - logits[foil]))
        rows.append(margins_for_pair)
    return rows


def margin_gain(rows, baseline):
    return mean(found - reference for row, control in zip(rows, baseline, strict=True)
                for found, reference in zip(row, control, strict=True))


@torch.inference_mode()
def main(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = (args.checkpoint, args.reference, CORPUS, MANIFEST, SOURCE, DATA,
              *(path / name for path in args.candidates for name in ('attention.pt', 'comparison.json')))
    sources = {str(path): sha(path) for path in inputs}
    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    saved_hashes = reference['results']['targeted']['training']['candidate_hashes']
    if json.loads(MANIFEST.read_text(encoding='utf-8'))['files'][CORPUS.name] != sources[str(CORPUS)]:
        raise ValueError('Training corpus digest mismatch')
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    training = json.loads(DATA.read_text(encoding='utf-8'))
    examples = dialogue_data(training, held=False, count=16)
    prompts = recall_prompts(tokenizer, source, extended=True)
    if {row['question'] for row in prompts} & {row['question'] for row in examples}:
        raise ValueError('Head selection overlaps held questions')
    calibration = training_pairs(tokenizer, training, examples)
    order = gradient_sample_indices(calibration)
    pairs = [calibration[index] for index in order]
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    base.lm_head.to(base.device)
    identity = frozen_digest(runtime.organism, runtime.state)
    replacements = load_recurrent_replacements(decoder, args.candidates, sources[str(args.checkpoint)])
    if (set(replacements) != set(LAYERS) or identity != reference['individual_sha256']
            or any(sources[str(directory / 'attention.pt')] != saved_hashes[str(index)]
                   for index, directory in zip(LAYERS, args.candidates, strict=True))):
        raise ValueError('The intervention must use the same model, individual and recovered kernels')
    _, replacement = replacements[TARGET_LAYER]
    head_count = replacement.source.sizes[0] // replacement.source.head_dim
    two_layers = {key: value for key, value in replacements.items() if key != TARGET_LAYER}
    with using(two_layers, 'recurrent'):
        native = margins(decoder, pairs)
    with using(replacements, 'recurrent'):
        recurrent = margins(decoder, pairs)
        with exact_heads(replacement, tuple(range(head_count))):
            exact = margins(decoder, pairs)
        # The all-head intervention should match the native layer up to kernel rounding.
        max_delta = max(abs(a - b) for left, right in zip(exact, native, strict=True)
                        for a, b in zip(left, right, strict=True))
        if max_delta > .1:
            raise ValueError(f'All-head restoration disagrees with native attention by {max_delta:.4f} logits')
        rows = []
        for head in range(head_count):
            with exact_heads(replacement, (head,)):
                scores = margins(decoder, pairs)
            row = dict(head=head, gain=margin_gain(scores, recurrent), margins=scores)
            print('Head', row, flush=True)
            rows.append(row)
        ranked = sorted(rows, key=lambda row: (-row['gain'], row['head']))
        groups = replacement.source.num_key_value_groups
        chosen = tuple(row['head'] for row in ranked[:2])
        # Same query-head count. Match KV group use when two disjoint heads permit it.
        chosen_groups = {head // groups for head in chosen}
        alternatives = [row['head'] for row in reversed(ranked) if row['head'] not in chosen]
        matched = [head for head in alternatives if head // groups in chosen_groups]
        if len(chosen_groups) == 1 and len(matched) >= 2:
            control = tuple(matched[:2])
        elif len(chosen_groups) == 2 and chosen_groups <= {head // groups for head in matched}:
            control = tuple(next(head for head in matched if head // groups == group)
                            for group in sorted(chosen_groups))
        else:
            control = tuple(alternatives[:2])
        calibration_result = dict(pairs=order, native=native, recurrent=recurrent, all_exact=exact,
                                  max_native_delta=max_delta, scores=rows, chosen=chosen, control=control,
                                  kv_group_size=groups)
        store_json(args.output / 'calibration.json', calibration_result)
        held = [row for row in prompts if row['length'] != 'short']
        results = {}
        for label, heads in (('chosen', chosen), ('control', control)):
            with exact_heads(replacement, heads):
                results[label] = run_dialogue(decoder, tokenizer, held, source['open'])
            result = results[label]
            print(label, heads, result['recall_passed'], result['open_turns_passed'], flush=True)
    if (identity != frozen_digest(runtime.organism, runtime.state)
            or any(sha(path) != sources[str(path)] for path in inputs)):
        raise RuntimeError('Source or individual changed')
    store_json(args.output / 'result.json', dict(calibration=calibration_result, held=results, sources=sources,
                                                individual_sha256=identity, seconds=perf_counter() - started))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('reference', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('candidates', type=Path, nargs=3)
    main(cli.parse_args())
