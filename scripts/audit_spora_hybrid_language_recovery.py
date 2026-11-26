"""Reload recovered attention layouts and compare generation and persistent reads."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from scripts.benchmark_spora_recurrent_dialogue import (
    MODEL, SOURCE, AutoTokenizer, extend_positions, frozen_digest, load_full_streamed,
    load_integrated_runtime, recall_prompts, run_dialogue, sha, using,
)
from scripts.train_spora_recurrent_counterfactual import store_json
from scripts.train_qwen_counterfactual_reply_20261122 import HELD, evaluate, records
from spora.research_recurrent_decode import load_recurrent_replacements


def main(args):
    started = perf_counter()
    manifest = args.recovery / 'result.json'
    reference = json.loads(manifest.read_text(encoding='utf-8'))
    protected = {path: sha(path) for path in (args.checkpoint, SOURCE, manifest)}
    if reference['sources'][str(args.checkpoint)] != protected[args.checkpoint]:
        raise ValueError('Recovery used a different parent checkpoint')
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    prompts = recall_prompts(tokenizer, source, extended=True)
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    base.lm_head.to(base.device)
    identity = frozen_digest(runtime.organism, runtime.state)
    if identity != reference['individual_sha256']:
        raise ValueError('Recovery used a different individual')
    memory = records(runtime.organism, HELD)
    results = {}
    for layout in ((19, 22), (19, 21, 22), (19, 22, 24)):
        name = '-'.join(map(str, layout))
        directories = [args.recovery / name / f'layer-{index}' for index in layout]
        for directory in directories:
            for filename in ('attention.pt', 'comparison.json'):
                path = directory / filename
                protected[path] = sha(path)
        replacements = load_recurrent_replacements(decoder, directories, protected[args.checkpoint])
        with using(replacements, 'recurrent'):
            dialogue = run_dialogue(decoder, tokenizer, prompts, source['open'])
            persistent = evaluate(decoder, tokenizer, runtime.organism, memory)
        results[name] = dict(dialogue=dialogue, memory=persistent)
        print(name, dialogue['recall_passed'], dialogue['open_turns_passed'], persistent['summary'], flush=True)
        del replacements
        torch.cuda.empty_cache()
    if identity != frozen_digest(runtime.organism, runtime.state) or any(
            sha(path) != digest for path, digest in protected.items()):
        raise RuntimeError('Source or individual changed in evaluation')
    store_json(args.output, dict(results=results, individual_sha256=identity,
               sources={str(path): digest for path, digest in protected.items()}, seconds=perf_counter() - started))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('recovery', type=Path)
    cli.add_argument('output', type=Path)
    args = cli.parse_args()
    if args.output.exists():
        cli.error('Choose a new output path')
    main(args)
