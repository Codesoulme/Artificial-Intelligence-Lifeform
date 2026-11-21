"""Test a frequency-ordered query/key migration on the native value checkpoint."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from audit_awake_multiturn_layers import CEVAL_SUBJECTS, ceval_questions, score_ceval
from audit_awake_reasoning_residual_holdout_20261121 import DATA, load_scenarios
from audit_spora_dialogue_migration import CEVAL_GROUPS
from audit_spora_split_position_attention import compare_generation_cache, last_token_distributions
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_checkpoint import load_integrated_runtime, value_position_modules
from spora.language_training import atomic_save, frozen_digest
from spora.research_partial_query_key import PartialQueryKeyPositionAttention, select_query_key_rows
from train_spora_grouped_absolute_attention import LAYER, MAX_POSITION, evaluate_candidate


def full_result(runtime, tokenizer, args, reference, extra_questions, ordinary):
    result = evaluate_candidate(runtime, tokenizer, args.ceval, reference['result'])
    more = score_ceval(runtime.decoder, extra_questions, MAX_POSITION)
    cache = compare_generation_cache(runtime, tokenizer, ordinary)
    additional_before = sum(row['choice'] == row['expected'] for row in reference['additional_ceval'])
    additional_after = sum(row['choice'] == row['expected'] for row in more)
    eos_before = sum(turn['eos'] for rows in reference['result']['dialogue'].values() for turn in rows)
    eos_after = sum(turn['eos'] for rows in result['dialogue'].values() for turn in rows)
    counts = {'first_ceval_exact': result['ceval'] == reference['result']['ceval'],
              'additional_ceval_exact': more == reference['additional_ceval'],
              'additional_before': additional_before, 'additional_after': additional_after,
              'dialogue_exact': result['summary']['dialogue_unchanged'],
              'eos_before': eos_before, 'eos_after': eos_after,
              'memory_exact': result['memory'] == reference['result']['memory'],
              'generation_cache': cache}
    summary = result['summary']
    counts['passed'] = (summary['ceval_after'] >= summary['ceval_before']
                        and additional_after >= additional_before and eos_after >= eos_before
                        and counts['memory_exact'] and all(cache.values()))
    return {'checks': counts, 'result': result, 'additional_ceval': more}


def scan(args, runtime, layer, parent, tokenizer, reference, extra_questions, protected):
    previous = args.attention.with_name('result.json')
    recorded = json.loads(previous.read_text(encoding='utf-8'))
    if recorded['candidate_sha256'] != protected[args.attention]:
        raise ValueError('Unverified absolute-position projection')
    source = torch.load(args.attention, weights_only=True, map_location='cpu')
    if (source['format'] != 'spora-grouped-absolute-attention-v1' or source['layer'] != LAYER
            or source['source_sha256'] != json.loads(args.checkpoint.with_name('packed.json').read_text(
                encoding='utf-8'))['parent_sha256']):
        raise ValueError('Projection does not share the original model lineage')
    decoder = runtime.decoder
    half = parent.head_dim // 2
    if half < 32:
        raise ValueError('The planned test needs at least 32 rotary coordinate pairs')
    candidates = {'low_8': tuple(range(half - 8, half)), 'low_16': tuple(range(half - 16, half)),
                  'low_32': tuple(range(half - 32, half)), 'high_16': tuple(range(16))}
    prompts = [closed_think_inputs(tokenizer, row['turns'][0], 'cuda') for row in load_scenarios()]
    questions = tuple(ceval_questions(args.ceval, tokenizer))
    prior = reference['result']['ceval']
    threshold = sum(row['choice'] == row['expected'] for row in prior)
    if args.screen is not None:
        existing = json.loads(args.screen.read_text(encoding='utf-8'))
        for path, digest in protected.items():
            if path != args.screen and existing['sources'].get(str(path)) != digest:
                raise ValueError(f'Screen was generated with different inputs: {path}')
        screen = existing['screen']
    else:
        screen = {}
        target = last_token_distributions(decoder, prompts)
    modules = {}
    for name, pairs in candidates.items():
        weight, bias = select_query_key_rows(parent, pairs, source['state'])
        module = PartialQueryKeyPositionAttention(parent, pairs, weight, bias).eval()
        modules[name] = module
        if args.screen is not None:
            if tuple(screen[name]['pairs']) != pairs:
                raise ValueError(f'Screen pair selection does not match: {name}')
        else:
            layer.self_attn = module
            distances = F.kl_div(last_token_distributions(decoder, prompts), target, reduction='none',
                                 log_target=True).sum(dim=-1)
            row = {'pairs': pairs, 'vocabulary_kl': float(distances.mean())}
            if name != 'high_16':
                academic = score_ceval(decoder, questions, MAX_POSITION)
                row['ceval_correct'] = sum(case['choice'] == case['expected'] for case in academic)
                row['ceval_same_choices'] = sum(left == right for left, right in zip(academic, prior, strict=True))
            screen[name] = row
            print('Screen', name, row, flush=True)
    selected = None
    evaluated = {}
    options = [name for name in ('low_32', 'low_16', 'low_8') if screen[name]['ceval_correct'] >= threshold]
    options.sort(key=lambda name: (screen[name]['ceval_correct'], len(candidates[name])), reverse=True)
    for name in options:
        layer.self_attn = modules[name]
        evaluated[name] = full_result(runtime, tokenizer, args, reference, extra_questions, prompts[1])
        print('Full', name, evaluated[name]['checks'], flush=True)
        if evaluated[name]['checks']['passed']:
            selected = name
            break
    return {'screen': screen, 'evaluated': evaluated, 'selected': selected}, modules.get(selected)


def main(args):
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    parent_file = args.checkpoint.with_name('packed.json')
    extra_dir, extra_subjects = next(
        (folder, subjects) for folder, subjects in CEVAL_GROUPS.items() if subjects != CEVAL_SUBJECTS
    )
    extra_path = args.ceval.parent / extra_dir
    paths = [args.checkpoint, parent_file, args.reference, DATA,
             *(args.ceval / f'{subject}.parquet' for subject in CEVAL_SUBJECTS),
             *(extra_path / f'{subject}.parquet' for subject in extra_subjects)]
    if args.attention is not None:
        paths.extend((args.attention, args.attention.with_name('result.json')))
        if args.screen is not None:
            paths.append(args.screen)
    else:
        paths.extend((args.component, args.component.with_suffix('.json')))
    protected = {path: sha(path) for path in paths}
    parent_manifest = json.loads(parent_file.read_text(encoding='utf-8'))
    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    if (parent_manifest['bootstrap_sha256'] != protected[args.checkpoint]
            or parent_manifest['evaluated_component_sha256'] != protected[args.reference]
            or not reference['accepted']):
        raise ValueError('The parent and reference must match the verified native value-position experiment')
    for path in paths:
        if path == DATA or path.suffix == '.parquet':
            if reference['sources'].get(str(path)) != protected[path]:
                raise ValueError(f'Benchmark input does not match the verified parent: {path}')
    torch.set_num_threads(2)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False, local_files_only=True)
    runtime = load_integrated_runtime(args.checkpoint, load_full_streamed(), MODEL)
    runtime.decoder.eval().requires_grad_(False)
    state_digest = frozen_digest(runtime.organism, runtime.state)
    layer = runtime.decoder.base.model.layers[LAYER]
    parent = value_position_modules(runtime.decoder)[LAYER]
    extra_questions = tuple(ceval_questions(extra_path, tokenizer, extra_subjects))
    candidate = None
    try:
        if args.attention is not None:
            report, candidate = scan(args, runtime, layer, parent, tokenizer, reference, extra_questions, protected)
        else:
            selection = json.loads(args.component.with_suffix('.json').read_text(encoding='utf-8'))
            payload = torch.load(args.component, weights_only=True, map_location='cpu')
            if (payload['format'] != 'spora-partial-query-key-v1' or payload['layer'] != LAYER
                    or payload['group'] != parent.group or payload['parent_sha256'] != protected[args.checkpoint]
                    or selection['component_sha256'] != protected[args.component]):
                raise ValueError('Partial query/key component belongs to a different experiment')
            layer.self_attn = PartialQueryKeyPositionAttention(
                parent, payload['pairs'], payload['weight'], payload['bias'],
            ).eval()
            ordinary = closed_think_inputs(tokenizer, load_scenarios()[1]['turns'][0], 'cuda')
            report = full_result(runtime, tokenizer, args, reference, extra_questions, ordinary)
            expected = selection['evaluated'][selection['selected']]
            turns_exact = all(
                [row['tokens'] for row in turns] == [row['tokens'] for row in expected['result']['dialogue'][name]]
                for name, turns in report['result']['dialogue'].items()
            )
            report['selected_reload_exact'] = (turns_exact and report['checks'] == expected['checks']
                                               and report['result']['ceval'] == expected['result']['ceval']
                                               and report['result']['memory'] == expected['result']['memory']
                                               and report['additional_ceval'] == expected['additional_ceval'])
            if not report['checks']['passed'] or not report['selected_reload_exact']:
                raise ValueError('The selected component failed its fresh-process verification')
    finally:
        layer.self_attn = parent
    if frozen_digest(runtime.organism, runtime.state) != state_digest or any(
        sha(path) != digest for path, digest in protected.items()
    ):
        raise ValueError('A protected model or input changed during evaluation')
    report.update({'seconds': time.perf_counter() - started, 'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
                   'sources': {str(path): digest for path, digest in protected.items()}})
    output.parent.mkdir(parents=True, exist_ok=True)
    if candidate is not None:
        artifact = output.with_suffix('.pt')
        if artifact.exists():
            raise FileExistsError(artifact)
        atomic_save({'format': 'spora-partial-query-key-v1', 'layer': LAYER, 'group': parent.group,
                     'parent_sha256': protected[args.checkpoint], 'pairs': candidate.pairs,
                     'weight': candidate.weight.cpu(), 'bias': candidate.bias.cpu()}, artifact)
        report['component_sha256'] = sha(artifact)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('Wrote', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'reference', 'ceval', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--attention', type=Path, help='Screen with the original full absolute-position projection')
    mode.add_argument('--component', type=Path, help='Verify a selected compact component after reload')
    parser.add_argument('--screen', type=Path, help='Reuse the short selection results from a previous run')
    args = parser.parse_args()
    if args.screen is not None and args.attention is None:
        parser.error('--screen requires --attention')
    main(args)
