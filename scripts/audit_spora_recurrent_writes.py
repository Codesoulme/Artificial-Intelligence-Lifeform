"""Compare matched sparse factual and distractor writes to a frozen recurrent summary."""

import argparse
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from scripts.audit_spora_recurrent_history_ablation import summarize
from scripts.audit_spora_recurrent_pathways import BOUNDARY, POSITIONS
from scripts.benchmark_spora_recurrent_dialogue import (
    MODEL, SOURCE, AutoTokenizer, encode, extend_positions, frozen_digest, load_full_streamed,
    load_integrated_runtime, recall_prompts, run_dialogue, sha, torch,
)
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.research_recurrent_decode import load_recurrent_replacements, replace, restrict_full_attention


FIELD_LABELS = {
    'workshop': ('更新后的取展板房间', '仍负责的联络人'),
    'shipment': ('现在的寄出日期', '收货人'),
    'supplies': ('现在要带的物品', '集合地点'),
    'volunteers': ('更新后的组名', '现在负责的岗位'),
    'delivery': ('更新后的数量', '接收人'),
}


def locate(text, offsets, fragment):
    start = text.find(fragment)
    if start < 0 or text.find(fragment, start + len(fragment)) >= 0:
        raise ValueError(f'Expected one occurrence of {fragment!r} in the probe')
    end = start + len(fragment)
    ids = tuple(index for index, (left, right) in enumerate(offsets) if left < right and left < end and right > start)
    if not ids or offsets[ids[0]][0] > start or offsets[ids[-1]][1] < end:
        raise ValueError('The token offsets do not cover the complete value')
    return ids


def write_positions(tokenizer, prompt, *, records=False):
    messages = [*prompt['history'], {'role': 'user', 'content': prompt['question']}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + '</think>\n\n'
    tokens = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if tokens.input_ids != encode(tokenizer, messages) or len(tokens.input_ids) != prompt['input_tokens']:
        raise ValueError('The offset-aware prompt differs from the recorded generation input')
    offsets, values = tokens.offset_mapping, prompt['values']
    targets = [locate(text, offsets, values[field]) for field in ('keep', 'new')]
    distractors = [locate(text, offsets, prompt['history'][i]['content']) for i in (4, 6)]
    controls = [region[:len(target)] for region, target in zip(distractors, targets, strict=True)]
    selected, matched = tuple(sorted((*targets[0], *targets[1]))), tuple(sorted((*controls[0], *controls[1])))
    superseded = locate(text, offsets, values['old'])
    if (any(len(a) != len(b) for a, b in zip(targets, controls, strict=True))
            or len(set((*selected, *matched, *superseded))) != len(selected) + len(matched) + len(superseded)
            or max((*selected, *matched)) >= BOUNDARY - 16
            or max(selected) >= min(matched)):
        raise ValueError('Expected distinct, count-matched fact and nearby distractor writes before the boundary')
    if records:
        statements = tuple(sorted({p for i in (0, 2) for p in locate(text, offsets, prompt['history'][i]['content'])}))
        fillers = tuple(sorted({p for part in distractors for p in part}))
        budget = len(statements)
        if (not set((*selected, *superseded)) <= set(statements) or not max(statements) < min(fillers)
                or len(fillers) < budget or fillers[budget - 1] >= BOUNDARY - 16):
            raise ValueError('Complete records and matched filler must fit before the boundary')
        return {'records': statements, 'values_plus_filler': (*selected, *fillers[:budget - len(selected)]),
                'filler_only': fillers[:budget]}
    return {'facts': selected, 'distractors': matched}


@contextmanager
def selected_summary_writes(core, mask):
    """Apply one absolute token mask to the keys that leave the recent buffer."""
    if core.update_rule != 'sum' or mask.ndim != 1 or mask.dtype != torch.bool:
        raise ValueError('Sparse probe expects cumulative attention and one boolean write mask')
    cursor = {'folded': 0}
    fold, prefixes = core.fold, core.prefixes

    def restrict(keys):
        count = keys.shape[-2]
        start = cursor['folded']
        enabled = mask[start:start + count]
        if enabled.numel() < count:
            enabled = torch.nn.functional.pad(enabled, (0, count - enabled.numel()))
        cursor['folded'] += count
        return keys * enabled[:, None]

    def fold_selected(keys, values, matrix, normalizer):
        return fold(restrict(keys), values, matrix, normalizer)

    def prefixes_selected(keys, values):
        sparse = restrict(keys.flatten(2, 3)).unflatten(-2, keys.shape[-3:-1])
        return prefixes(sparse, values)

    with patch.object(core, 'fold', new=fold_selected), patch.object(core, 'prefixes', new=prefixes_selected):
        yield cursor


def sparse_arm(decoder, tokenizer, candidates, prompts, positions, mode):
    rows, masks = [], []
    with ExitStack() as hooks:
        hooks.enter_context(restrict_full_attention(decoder, BOUNDARY, candidates))
        for block, candidate in candidates.values():
            hooks.enter_context(replace(block, candidate))
        for prompt, indices in zip(prompts, positions, strict=True):
            mask = torch.zeros(BOUNDARY, device=decoder.base.device, dtype=torch.bool)
            mask[list(indices[mode])] = True
            with ExitStack() as writes:
                counters = [writes.enter_context(selected_summary_writes(candidate.core, mask))
                            for _, candidate in candidates.values()]
                output, = run_dialogue(decoder, tokenizer, [prompt], [])['recall']
            folded = [counter['folded'] for counter in counters]
            if len(set(folded)) != 1 or min(folded) <= max(indices[mode]):
                raise ValueError('Both recurrent layers must have received every selected old position')
            rows.append(output)
            masks.append({'positions': indices[mode], 'total_folded': folded[0]})
    return {'rows': rows, 'writes': masks, 'summary': summarize(rows)['extended']}


def exact_arm(decoder, tokenizer, layers, block, prompts, positions, mode, *, all_layers):
    rows = []
    other = set(range(len(decoder.base.model.layers))) - layers
    for prompt, indices in zip(prompts, positions, strict=True):
        with ExitStack() as hooks:
            if all_layers:
                hooks.enter_context(restrict_full_attention(decoder, BOUNDARY, (), extra_keys=indices[mode]))
            else:
                hooks.enter_context(restrict_full_attention(decoder, BOUNDARY, layers))
                hooks.enter_context(restrict_full_attention(decoder, BOUNDARY, other,
                                                            extra_keys=indices[mode], recent_block=block))
            output, = run_dialogue(decoder, tokenizer, [prompt], [])['recall']
        rows.append(output)
    return {'rows': rows, 'read_positions': [indices[mode] for indices in positions],
            'summary': summarize(rows)['extended']}


def with_facts_in_question(tokenizer, prompts, *, labeled=False):
    for prompt in prompts:
        values = prompt['values']
        if labeled:
            first, second = FIELD_LABELS[prompt['case']]
            question = f"{first}是{values['new']}，{second}是{values['keep']}。{prompt['question']}"
        else:
            question = f"核对结果是{values['new']}，仍有效的另一项是{values['keep']}。{prompt['question']}"
        ids = encode(tokenizer, [*prompt['history'], {'role': 'user', 'content': question}])
        yield {**prompt, 'question': question, 'input_tokens': len(ids)}


def near_records(tokenizer, prompts, *, hide_retired=False):
    for prompt in prompts:
        original = prompt['history']
        records = original[:4]
        if hide_retired:
            first, *rest = records
            retired = prompt['values']['old']
            if first['content'].count(retired) != 1:
                raise ValueError('Expected exactly one retired value in the initial record')
            first = {**first, 'content': first['content'].replace(retired, '（旧值略）')}
            records = [first, *rest]
        history = [*original[4:], *records]
        question = {'role': 'user', 'content': prompt['question']}
        ids = encode(tokenizer, [*history, question])
        earlier = encode(tokenizer, history[:-4])
        if len(earlier) <= BOUNDARY or len(ids) > POSITIONS - 48:
            raise ValueError('Near records must fit after the segment boundary')
        if not hide_retired and len(ids) != prompt['input_tokens']:
            raise ValueError('Relocating unchanged turns must preserve the prompt length')
        yield {**prompt, 'history': history, 'input_tokens': len(ids)}


def main(args):
    started = perf_counter()
    paths = (args.checkpoint, SOURCE, *(folder / name for folder in args.candidates
                                      for name in ('attention.pt', 'comparison.json')))
    protected = {path: sha(path) for path in (*paths, args.baseline)}
    baseline = json.loads(args.baseline.read_text(encoding='utf-8'))
    if (baseline['boundary'] != BOUNDARY or baseline['positions'] != POSITIONS
            or any(baseline['sources'].get(str(path)) != protected[path] for path in paths)):
        raise ValueError('The dense-write comparison used different sources or context limits')
    dense = baseline['conditions']['restricted_with_far']
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    prompts = [row for row in recall_prompts(tokenizer, source, extended=True) if row['length'] == 'extended']
    if any(saved[key] != value for saved, prompt in zip(dense['rows'], prompts, strict=True)
           for key, value in prompt.items()):
        raise ValueError('The dense-write baseline used different prompts')
    record_mode = args.record_control or args.record_write or args.near_record_control
    positions = [write_positions(tokenizer, prompt, records=record_mode) for prompt in prompts]
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, POSITIONS)
    torch.cuda.empty_cache()
    base.lm_head.to(base.device)
    individual = frozen_digest(runtime.organism, runtime.state)
    if individual != baseline['individual_sha256']:
        raise ValueError('The dense comparison used a different individual')
    candidates = load_recurrent_replacements(decoder, args.candidates, protected[args.checkpoint])
    if tuple(candidates) != tuple(baseline['layers']):
        raise ValueError('The sparse probe requires the same recurrent layer placement')
    results = {'dense': dense}
    if args.near_record_control:
        results['far_records'] = exact_arm(decoder, tokenizer, set(candidates), None, prompts, positions,
                                           'records', all_layers=True)
        for mode, hide_retired in (('near_records', False), ('near_without_retired', True)):
            with restrict_full_attention(decoder, BOUNDARY, ()):
                rows = run_dialogue(decoder, tokenizer, near_records(tokenizer, prompts,
                                                                    hide_retired=hide_retired), [])['recall']
            results[mode] = {'rows': rows, 'summary': summarize(rows)['extended']}
        for mode in ('far_records', 'near_records', 'near_without_retired'):
            print(mode, results[mode]['summary'], flush=True)
    elif args.record_write:
        block, = {candidate.core.block for _, candidate in candidates.values()}
        for mode in ('records', 'filler_only'):
            results[mode] = sparse_arm(decoder, tokenizer, candidates, prompts, positions, mode)
            exact = f'{mode}_exact_two_layers'
            results[exact] = exact_arm(decoder, tokenizer, set(candidates), block, prompts, positions, mode,
                                       all_layers=False)
            for label in (mode, exact):
                print(label, results[label]['summary'], flush=True)
    elif args.record_control:
        for mode in ('records', 'values_plus_filler', 'filler_only'):
            results[mode] = exact_arm(decoder, tokenizer, set(candidates), None, prompts, positions, mode,
                                      all_layers=True)
            print(mode, results[mode]['summary'], flush=True)
    elif args.exact_control:
        block, = {candidate.core.block for _, candidate in candidates.values()}
        for label, all_layers in (('two_layers', False), ('all_layers', True)):
            for mode in ('facts', 'distractors'):
                arm = f'{mode}_{label}'
                results[arm] = exact_arm(decoder, tokenizer, set(candidates), block, prompts, positions, mode,
                                         all_layers=all_layers)
                print(arm, results[arm]['summary'], flush=True)
    else:
        for mode in ('facts', 'distractors'):
            results[mode] = sparse_arm(decoder, tokenizer, candidates, prompts, positions, mode)
            print(mode, results[mode]['summary'], flush=True)
    if not record_mode:
        with restrict_full_attention(decoder, BOUNDARY, ()):
            rows = run_dialogue(decoder, tokenizer, with_facts_in_question(tokenizer, prompts,
                                                                          labeled=args.exact_control), [])['recall']
        results['facts_in_question'] = {'rows': rows, 'summary': summarize(rows)['extended']}
        print('facts_in_question', results['facts_in_question']['summary'], flush=True)
    if (individual != frozen_digest(runtime.organism, runtime.state)
            or any(sha(path) != expected for path, expected in protected.items())):
        raise ValueError('A protected source or the individual changed')
    store_json(args.output, {'conditions': results, 'boundary': BOUNDARY, 'positions': POSITIONS,
                             'individual_sha256': individual, 'layers': list(candidates),
                             'sources': {str(path): value for path, value in protected.items()},
                             'seconds': perf_counter() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('candidates', nargs=2, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--exact-control', action='store_true', help='Read the same old positions with exact attention')
    mode.add_argument('--record-control', action='store_true', help='Compare whole statements at matched exact budgets')
    mode.add_argument('--record-write', action='store_true', help='Compare recurrence and exact reads of whole records')
    mode.add_argument('--near-record-control', action='store_true',
                      help='Move whole records closer and test omission of the retired value')
    options = parser.parse_args()
    if options.output.exists():
        parser.error('Choose a new output path')
    main(options)
