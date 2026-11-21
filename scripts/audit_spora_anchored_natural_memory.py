"""Compare natural memory routing with the bounded 64-token prefix."""

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoTokenizer

from scripts.audit_qwen_compatible_natural_memory_20261121 import fresh_version_cases, load_head
from scripts.audit_spora_local_attention import bounded_generation
from scripts.audit_spora_native_query_key import assess, write
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_checkpoint import load_integrated_runtime
from spora.language_training import frozen_digest
from spora.research_local_attention import anchor_attention, anchored_cache


def benchmark_questions(filler, *, heldout=False):
    cases = fresh_version_cases()
    pools = {}
    for question, index, mode in cases:
        pools.setdefault((index, mode), []).append(question)
    people = (1, 3, 5, 7) if heldout else (0, 2, 4, 6)
    template = 1 if heldout else 0
    subset = [(pools[index, mode][template], index, mode) for index in people for mode in (0, 1, 2)]
    negative = pools[None, None]
    subset.extend((question, None, None) for question in negative[template::len(negative) // 8][:8])
    return subset, [(f'{filler}\n\n{question}', index, mode) for question, index, mode in subset]


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    temporal_source = args.temporal_source or args.route_source
    paths = (args.checkpoint, args.route_source, args.router / 'router.pt', args.temporal / 'temporal.pt',
             temporal_source, ROOT / 'data/chinese-pilot/source.json')
    if args.version_selector is not None:
        paths = (args.version_selector, args.version_selector.with_name('result.json'), *paths)
    sources = {str(path): sha(path) for path in paths}
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(args.checkpoint, load_full_streamed(), MODEL)
    decoder = runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    layer_ids = decoder.compatible_block_indices[-4:]
    if len(layer_ids) != 4 or min(layer_ids) <= max(decoder.layer_indices):
        raise ValueError('The diagnostic needs four compatible layers after memory reading')
    layout = {'layers': list(layer_ids), 'window': 256, 'prefix': 64}
    if args.reload_only and decoder.anchored_attention_layout != layout:
        raise ValueError('The loaded checkpoint does not contain the expected bounded attention layout')
    training_base = sources[str(args.route_source)]
    router, _ = load_head(args.router, 'router.pt', 'router_sha256', training_base)
    temporal, time_sha = load_head(args.temporal, 'temporal.pt', 'temporal_sha256', sources[str(temporal_source)])
    if args.version_selector is not None:
        candidate = torch.load(args.version_selector, map_location='cpu', weights_only=True)
        manifest = json.loads(args.version_selector.with_name('result.json').read_text(encoding='utf-8'))
        if (candidate['format'] != 'spora-guarded-version-selector-v1'
                or candidate['checkpoint_sha256'] != sources[str(args.checkpoint)]
                or candidate['temporal_sha256'] != time_sha
                or manifest['selector_sha256'] != sources[str(args.version_selector)]):
            raise ValueError('Version selector belongs to a different model or report')
        temporal = dict(temporal, version_selector={**candidate, 'char_width': candidate.get('char_width', 0)})
    prose = json.loads(paths[-1].read_text(encoding='utf-8'))
    filler = '\n'.join(prose['development'] + prose['test'])
    if args.heldout:
        tokens = tokenizer.encode('\n'.join(prose['train']), add_special_tokens=False)
        filler = tokenizer.decode(tokens[:480], skip_special_tokens=True)
    short, long = benchmark_questions(filler, heldout=args.heldout)
    suites = [('short', short, None), ('long', long, None)]
    if args.route_current_turn:
        suites = [('long_with_separate_question', long, [question for question, _, _ in short])]
    if args.broad_versions:
        from scripts.train_spora_revision_contrast import held_questions

        suites = [('previous_contrast_holdout', held_questions(), None)]
    results = {}
    for name in (('bounded',) if args.reload_only else ('full', 'bounded')):
        scope, generation = nullcontext(), nullcontext()
        if name == 'bounded' and not args.reload_only and args.pack_to is None:
            scope = anchor_attention(decoder, layer_ids, 256, 64)
            generation = bounded_generation(decoder, lambda: anchored_cache(decoder.base.config, layer_ids, 256, 64))
        if name == 'bounded' and args.pack_to is not None:
            decoder.configure_anchored_attention(layout)
            pack(runtime, args.checkpoint, args.pack_to, layout)
        with scope, generation:
            for label, subset, routing in suites:
                for gain in args.output_gains:
                    result = assess(runtime, tokenizer, router, temporal, questions=subset, use_cache=True,
                                    routing_questions=routing, output_gain=gain)
                    key = label if args.output_gains == [1.] else f'{label}_gain_{gain:g}'
                    results.setdefault(name, {})[key] = result
                    print(name, key, result['summary'], flush=True)
    if individual != frozen_digest(runtime.organism, runtime.state) or any(
        sha(Path(path)) != digest for path, digest in sources.items()
    ):
        raise ValueError('The natural memory benchmark changed its source or individual')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, {'results': results, 'sources': sources, 'layers': layer_ids, 'use_cache': True,
                        'output_gains': args.output_gains, 'heldout': args.heldout,
                        'seconds': time.perf_counter() - started, 'peak_gpu_bytes': torch.cuda.max_memory_allocated()})


def pack(runtime, source, destination, layout):
    if destination.exists() or source.resolve() == destination.resolve():
        raise FileExistsError(destination)
    parent = torch.load(source, map_location='cpu', weights_only=True)
    if 'anchored_attention' in parent['metadata']:
        raise ValueError('A fresh source checkpoint is required')
    destination.parent.mkdir(parents=True, exist_ok=True)
    runtime.save(destination)
    packed = torch.load(destination, map_location='cpu', weights_only=True)
    before, after = parent['tensors'], packed['tensors']
    if (packed['metadata'] != dict(parent['metadata'], anchored_attention=layout) or before.keys() != after.keys()
            or any(not torch.equal(tensor, after[name]) for name, tensor in before.items())):
        raise ValueError('The candidate changed data outside the anchored attention layout')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'route-source', 'router', 'temporal', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--temporal-source', type=Path, help='Training checkpoint for an independently trained time head')
    parser.add_argument('--version-selector', type=Path, help='Evaluate an independently fitted version selector')
    parser.add_argument('--route-current-turn', action='store_true',
                        help='Use a separately provided user turn for query recognition and the full text for answering')
    parser.add_argument('--output-gains', type=float, nargs='+', choices=(1., 1.5, 2.), default=[1.],
                        help='Compare the strength of the existing neural memory output adapter')
    parser.add_argument('--heldout', action='store_true', help='Change names, query phrasing, and prose filler')
    parser.add_argument('--broad-versions', action='store_true', help='Evaluate the 128-query version contrast suite')
    io_mode = parser.add_mutually_exclusive_group()
    io_mode.add_argument('--pack-to', type=Path, help='Write an isolated anchored checkpoint and compare its layout')
    io_mode.add_argument('--reload-only', action='store_true', help='Evaluate the bounded layout restored from the checkpoint')
    main(parser.parse_args())
