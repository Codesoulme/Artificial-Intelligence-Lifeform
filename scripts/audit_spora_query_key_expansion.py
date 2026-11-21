"""Compare growing the first absolute query/key group with adding a second group."""

import argparse
import json
import re
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
from audit_spora_dialogue_migration import CEVAL_GROUPS, dialogue
from audit_spora_partial_query_key import full_result
from audit_spora_split_position_attention import last_token_distributions
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_checkpoint import load_integrated_runtime, query_key_position_modules
from spora.language_training import atomic_save, frozen_digest
from spora.research_partial_query_key import PartialQueryKeyPositionAttention, select_query_key_rows
from train_spora_grouped_absolute_attention import LAYER, MAX_POSITION
from train_spora_query_key_group import TRAIN_FILES, fit_query_key_group


FRESH_DIALOGUES = {
    'revised_pickup': (
        '快递放在北门。只回答取件地点。',
        '更正一下，快递已挪到南门。现在去哪取？只回答地点。',
    ),
    'changed_menu': (
        '我只能用米饭和青菜，推荐一道饭，菜名就行。',
        '现在只剩米饭，青菜也没有了。请换一道，菜名就行。',
    ),
    'inventory': (
        '盒里有九支笔。拿出一支后，盒里剩几支？只回答数字。',
        '又放回三支。现在盒里几支？只回答数字。',
    ),
    'carry_format': (
        '用四个字写一句早安问候。',
        '改成晚安问候，还是四个字。',
    ),
}
EGG_INGREDIENT = re.compile(r'^\s*[-*]\s+.*(?:鸡蛋|蛋黄|蛋清)', re.MULTILINE)


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
    inputs = (args.checkpoint, manifest, args.reference, args.attention, attention_report, *benchmarks)
    if args.calibration_steps:
        inputs += TRAIN_FILES
    sources = {str(path): sha(path) for path in inputs}
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
    scenarios = load_scenarios()
    prompts = [closed_think_inputs(tokenizer, scenario['turns'][0], 'cuda') for scenario in scenarios]
    plans = {'extend_first_group': (values, 0, tuple(range(half - 32, half))),
             'add_second_group': (native, 1, native.pairs)}
    if args.second_group_pairs:
        plans = {f'second_group_{count}': (native, 1, tuple(range(half - count, half)))
                 for count in sorted(args.second_group_pairs, reverse=True)}
    recipe = next(row['turns'] for row in scenarios if row['name'] == 'recipe_constraint')
    parent_recipe = reference['result']['dialogue']['recipe_constraint'][-1]['answer']
    if args.second_group_pairs and EGG_INGREDIENT.search(parent_recipe):
        raise ValueError('The saved parent must pass the egg ingredient control')
    academic = tuple(ceval_questions(args.ceval, tokenizer))
    extra = tuple(ceval_questions(extra_path, tokenizer, extra_subjects))
    prior = reference['result']['ceval']
    baseline = score_ceval(runtime.decoder, academic, MAX_POSITION)
    if baseline != prior:
        raise ValueError('The native parent did not reproduce its verified academic answers')
    prior_correct = sum(row['choice'] == row['expected'] for row in prior)
    teacher = last_token_distributions(runtime.decoder, prompts)
    results = {}
    selected = selected_module = None
    try:
        for name, (parent, group, pairs) in plans.items():
            weight, bias = select_query_key_rows(values, pairs, projection['state'], group=group)
            calibration = None
            if args.calibration_steps:
                layer.self_attn = native
                weight, bias, calibration = fit_query_key_group(runtime, tokenizer, native, group, pairs,
                                                                 weight, bias, args.calibration_steps)
                print('Calibration', name, calibration, flush=True)
            layer.self_attn = PartialQueryKeyPositionAttention(parent, pairs, weight, bias, group=group).eval()
            distributions = last_token_distributions(runtime.decoder, prompts)
            distance = F.kl_div(distributions, teacher, reduction='none', log_target=True).sum(dim=-1)
            choices = score_ceval(runtime.decoder, academic, MAX_POSITION)
            correct = sum(row['choice'] == row['expected'] for row in choices)
            row = {'group': group, 'pairs': pairs, 'mean_vocabulary_kl': float(distance.mean()),
                   'ceval_correct': correct, 'ceval_same_choices': sum(
                       left == right for left, right in zip(choices, prior, strict=True)), 'ceval': choices}
            if calibration is not None:
                row['calibration'] = calibration
            print('Screen', name, {key: value for key, value in row.items() if key != 'ceval'}, flush=True)
            results[name] = row
            if correct < prior_correct:
                continue
            if args.second_group_pairs:
                constraint = dialogue(runtime.decoder, tokenizer, recipe, open_reasoning=False, limit=96, seed=73121)
                row['recipe_regression_probe'] = constraint
                row['recipe_ingredient_passed'] = not bool(EGG_INGREDIENT.search(constraint[-1]['answer']))
                print('Recipe ingredient', name, row['recipe_ingredient_passed'], flush=True)
                if not row['recipe_ingredient_passed']:
                    continue
            if selected is None:
                row['full'] = full_result(runtime, tokenizer, args, reference, extra, prompts[1])
                print('Full', name, row['full']['checks'], flush=True)
                if args.second_group_pairs and row['full']['checks']['passed']:
                    selected, selected_module = name, layer.self_attn
    finally:
        layer.self_attn = native

    fresh = {}
    if selected is not None:
        try:
            for label, module in (('parent', native), ('selected', selected_module)):
                layer.self_attn = module
                fresh[label] = {name: dialogue(runtime.decoder, tokenizer, turns, open_reasoning=False,
                                                limit=64, seed=84521)
                                for name, turns in FRESH_DIALOGUES.items()}
                print('Fresh dialogue', label, flush=True)
        finally:
            layer.self_attn = native
    if state_digest != frozen_digest(runtime.organism, runtime.state) or any(
        sha(Path(path)) != digest for path, digest in sources.items()
    ):
        raise ValueError('The experiment changed an input or the individual memory')
    report = {'sources': sources, 'parent_ceval_correct': prior_correct, 'results': results,
              'selected_for_fresh_dialogue': selected, 'fresh_dialogue': fresh,
              'seconds': time.perf_counter() - started, 'peak_gpu_bytes': torch.cuda.max_memory_allocated()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if selected_module is not None and args.calibration_steps:
        artifact = args.output.with_suffix('.pt')
        if artifact.exists():
            raise FileExistsError(artifact)
        atomic_save({'format': 'spora-second-query-key-group-v1', 'layer': LAYER,
                     'parent_sha256': sources[str(args.checkpoint)], 'group': selected_module.group,
                     'pairs': selected_module.pairs, 'weight': selected_module.weight.detach().cpu(),
                     'bias': selected_module.bias.detach().cpu()}, artifact)
        report['research_component_sha256'] = sha(artifact)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('Wrote', args.output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'reference', 'attention', 'ceval', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--second-group-pairs', nargs='+', type=int, choices=range(1, 16),
                        metavar='PAIRS', help='Screen narrower second groups, widest first')
    parser.add_argument('--calibration-steps', type=int, default=0,
                        help='Fit one specified second-group candidate before screening')
    args = parser.parse_args()
    if args.second_group_pairs and len(args.second_group_pairs) != len(set(args.second_group_pairs)):
        parser.error('Second-group pair counts must be unique')
    if args.calibration_steps and (args.calibration_steps < 8 or len(args.second_group_pairs or ()) != 1):
        parser.error('Calibration requires at least 8 steps and one second-group pair count')
    main(args)
