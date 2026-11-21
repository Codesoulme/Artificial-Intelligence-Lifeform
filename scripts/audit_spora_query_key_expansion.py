"""Compare growing the first absolute query/key group with adding a second group."""

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
from audit_spora_partial_query_key import full_result
from audit_spora_split_position_attention import last_token_distributions
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_checkpoint import load_integrated_runtime, query_key_position_modules
from spora.language_training import frozen_digest
from spora.research_partial_query_key import PartialQueryKeyPositionAttention, select_query_key_rows
from train_spora_grouped_absolute_attention import LAYER, MAX_POSITION


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = args.checkpoint.with_name('before.json')
    attention_report = args.attention.with_name('result.json')
    extra_folder, extra_subjects = next(
        (folder, subjects) for folder, subjects in CEVAL_GROUPS.items() if subjects != CEVAL_SUBJECTS
    )
    extra_path = args.ceval.parent / extra_folder
    benchmarks = (DATA, *(args.ceval / f'{name}.parquet' for name in CEVAL_SUBJECTS),
                  *(extra_path / f'{name}.parquet' for name in extra_subjects))
    sources = {str(path): sha(path) for path in (
        args.checkpoint, manifest, args.reference, args.attention, attention_report, *benchmarks,
    )}
    packed = json.loads(manifest.read_text(encoding='utf-8'))['packed']
    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    attention = json.loads(attention_report.read_text(encoding='utf-8'))
    if (packed['bootstrap_sha256'] != sources[str(args.checkpoint)] or packed['layer'] != LAYER
            or packed['group'] != 0 or not reference['selected_reload_exact'] or not reference['checks']['passed']
            or packed['component_sha256'] not in reference['sources'].values()
            or attention['candidate_sha256'] != sources[str(args.attention)]
            or any(reference['sources'].get(str(path)) != sources[str(path)] for path in benchmarks)):
        raise ValueError('The native parent, projection, and verified benchmark must match')
    projection = torch.load(args.attention, weights_only=True, map_location='cpu')
    if projection['format'] != 'spora-grouped-absolute-attention-v1' or projection['layer'] != LAYER:
        raise ValueError('Expected the trained projection for this attention layer')

    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(args.checkpoint, load_full_streamed(), MODEL)
    runtime.decoder.eval().requires_grad_(False)
    state_digest = frozen_digest(runtime.organism, runtime.state)
    layer = runtime.decoder.base.model.layers[LAYER]
    native = query_key_position_modules(runtime.decoder)[LAYER]
    values = native.parent
    if list(native.pairs) != packed['pairs'] or len(native.pairs) != 16:
        raise ValueError('Expected the first 16-pair group to be present')
    half = native.head_dim // 2
    plans = {
        'extend_first_group': (values, 0, tuple(range(half - 32, half))),
        'add_second_group': (native, 1, native.pairs),
    }
    prompts = [closed_think_inputs(tokenizer, scenario['turns'][0], 'cuda') for scenario in load_scenarios()]
    academic = tuple(ceval_questions(args.ceval, tokenizer))
    extra = tuple(ceval_questions(extra_path, tokenizer, extra_subjects))
    prior = reference['result']['ceval']
    baseline = score_ceval(runtime.decoder, academic, MAX_POSITION)
    if baseline != prior:
        raise ValueError('The native parent did not reproduce its verified academic answers')
    prior_correct = sum(row['choice'] == row['expected'] for row in prior)
    teacher = last_token_distributions(runtime.decoder, prompts)
    results = {}
    try:
        for name, (parent, group, pairs) in plans.items():
            weight, bias = select_query_key_rows(values, pairs, projection['state'], group=group)
            layer.self_attn = PartialQueryKeyPositionAttention(parent, pairs, weight, bias, group=group).eval()
            distributions = last_token_distributions(runtime.decoder, prompts)
            distance = F.kl_div(distributions, teacher, reduction='none', log_target=True).sum(dim=-1)
            choices = score_ceval(runtime.decoder, academic, MAX_POSITION)
            correct = sum(row['choice'] == row['expected'] for row in choices)
            row = {'group': group, 'pairs': pairs, 'mean_vocabulary_kl': float(distance.mean()),
                   'ceval_correct': correct, 'ceval_same_choices': sum(
                       left == right for left, right in zip(choices, prior, strict=True)), 'ceval': choices}
            print('Screen', name, {key: value for key, value in row.items() if key != 'ceval'}, flush=True)
            if correct >= prior_correct:
                row['full'] = full_result(runtime, tokenizer, args, reference, extra, prompts[1])
                print('Full', name, row['full']['checks'], flush=True)
            results[name] = row
    finally:
        layer.self_attn = native
    if state_digest != frozen_digest(runtime.organism, runtime.state) or any(
        sha(Path(path)) != digest for path, digest in sources.items()
    ):
        raise ValueError('The experiment changed an input or the individual memory')
    report = {'sources': sources, 'parent_ceval_correct': prior_correct, 'results': results,
              'seconds': time.perf_counter() - started, 'peak_gpu_bytes': torch.cuda.max_memory_allocated()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('Wrote', args.output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'reference', 'attention', 'ceval', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    main(parser.parse_args())
