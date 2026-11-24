"""Localize long-dialogue regressions in the two recurrent decoder layers."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from time import perf_counter

from scripts.benchmark_spora_recurrent_dialogue import (
    MODEL, SOURCE, AutoTokenizer, extend_positions, frozen_digest, load_full_streamed, load_integrated_runtime,
    recall_prompts, run_dialogue, sha, torch,
)
from spora.research_recurrent_decode import load_recurrent_replacements, replace


ARMS = {
    'layer19': ((19,), True, None),
    'layer22': ((22,), True, None),
    'both_recent_only': ((19, 22), False, None),
    'both_wider_recent': ((19, 22), True, 32),
}


def summarize(rows):
    totals = {}
    for length in ('long', 'extended'):
        group = [row for row in rows if row['length'] == length]
        totals[length] = {'count': len(group), 'pass': sum(row['passed'] for row in group),
                          'old_value': sum(row['values']['old'] in row['answer'] for row in group),
                          'updated_value': sum(row['values']['new'] in row['answer'] for row in group),
                          'retained_value': sum(row['values']['keep'] in row['answer'] for row in group),
                          'ended': sum(row['eos'] for row in group)}
    return totals


def main(args):
    inputs = {path: sha(path) for path in (args.checkpoint, args.reference, SOURCE)}
    for directory in args.candidates:
        inputs.update({directory / name: sha(directory / name) for name in ('attention.pt', 'comparison.json')})
    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    if any(reference['sources'][str(path)] != digest for path, digest in inputs.items() if path != args.reference):
        raise ValueError('Long dialogue reference was measured on different inputs')
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    prompts = [row for row in recall_prompts(tokenizer, source, extended=True) if row['length'] != 'short']
    base = load_full_streamed()
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    if reference['individual_sha256'] != individual or reference['position_table'] != 2048:
        raise ValueError('Reference used a different individual or position table')
    extend_positions(decoder, rotary, 2048)
    layers = load_recurrent_replacements(decoder, args.candidates, inputs[args.checkpoint])
    if set(layers) != {19, 22}:
        raise ValueError('This ablation expects layers 19 and 22')
    baseline = {name: [row for row in reference['dialogue'][mode]['recall'] if row['length'] != 'short']
                for name, mode in (('parent', 'parent'), ('both_recurrent', 'recurrent'))}
    for rows in baseline.values():
        if [{key: answer[key] for key in prompt} for answer, prompt in zip(rows, prompts, strict=True)] != prompts:
            raise ValueError('Recall prompts differ from the reference')
    arms = {name: {'summary': summarize(rows), 'rows': rows} for name, rows in baseline.items()}
    for name in args.arms:
        selected, include_far, width = ARMS[name]
        started = perf_counter()
        with ExitStack() as installed:
            for index in selected:
                block, candidate = layers[index]
                installed.callback(setattr, candidate, 'include_far', candidate.include_far)
                candidate.include_far = include_far
                if width is not None:
                    installed.callback(setattr, candidate.core, 'block', candidate.core.block)
                    candidate.core.block = width
                installed.enter_context(replace(block, candidate))
            rows = run_dialogue(decoder, tokenizer, prompts, [])['recall']
        arms[name] = {'summary': summarize(rows), 'rows': rows, 'seconds': perf_counter() - started,
                      'layers': selected, 'include_far': include_far, 'block_override': width}
        print(name, arms[name]['summary'], flush=True)
    if (individual != frozen_digest(runtime.organism, runtime.state)
            or any(sha(path) != digest for path, digest in inputs.items())):
        raise ValueError('An ablation input changed')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump({'individual_sha256': individual, 'inputs': {str(path): digest for path, digest in inputs.items()},
                   'arms': arms}, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('candidates', nargs=2, type=Path)
    parser.add_argument('--arms', nargs='+', choices=tuple(ARMS), default=tuple(ARMS))
    args = parser.parse_args()
    if args.output.exists() or len(args.arms) != len(set(args.arms)):
        parser.error('Use a new output path and distinct conditions')
    main(args)
