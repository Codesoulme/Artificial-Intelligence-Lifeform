"""Compare bounded attention and its parent on long Chinese dialogue and decode cost."""

import argparse
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import re
from statistics import median
from time import perf_counter

from scripts.audit_spora_chat_backbones import reply
from scripts.audit_spora_recurrent_attention import (
    MODEL, AutoTokenizer, encode, frozen_digest, load_full_streamed, load_integrated_runtime, sha, torch,
)
from scripts.train_spora_dialogue_composition import settings_for
from spora.pretrained_attention import SPORAPositionTable
from spora.research_recurrent_decode import DynamicCache, RecurrentDecodeLayer, load_recurrent_replacements, replace


SOURCE = Path(__file__).resolve().parents[1] / 'data/recurrent-attention-dialogue-audit-v1.json'


def exchange(history, user, answer):
    history.extend(({'role': 'user', 'content': user}, {'role': 'assistant', 'content': answer}))


def recall_prompts(tokenizer, source, *, extended=False):
    rows = []
    for index, case in enumerate(source['recall']):
        for variant, (old, new, keep) in enumerate(case['values']):
            values = {'old': old, 'new': new, 'keep': keep}
            history = []
            exchange(history, case['first'].format_map(values), '我记下这项安排了。')
            exchange(history, case['update'].format_map(values), '好，我只修改这一项。')
            question = {'role': 'user', 'content': case['question']}
            short = list(history)
            extra = source['extra_filler'] if extended else []
            stages = [('short', short)]
            for label, fillers in (('long', source['filler']), ('extended', extra)):
                if fillers:
                    for i in range(len(fillers)):
                        exchange(history, fillers[(i + index) % len(fillers)], '收到，接着说吧。')
                    stages.append((label, list(history)))
            lengths = [len(encode(tokenizer, [*turns, question])) for _, turns in stages]
            if not (lengths[0] <= 160 and 600 <= lengths[1] <= 720
                    and (not extended or 1280 <= lengths[2] <= 1900)):
                raise ValueError(f'Unexpected prompt lengths for {case["name"]}: {lengths}')
            for (label, turns), length in zip(stages, lengths, strict=True):
                rows.append({'case': case['name'], 'variant': variant, 'length': label, 'values': values,
                             'question': case['question'], 'history': turns, 'input_tokens': length})
    return rows


def grade(answer, required, forbidden):
    return bool(answer['eos'] and all(re.search(pattern, answer['answer']) for pattern in required)
                and all(not re.search(pattern, answer['answer']) for pattern in forbidden))


def run_dialogue(decoder, tokenizer, prompts, open_cases):
    settings = settings_for(decoder, tokenizer, 48)
    position_limit = decoder.position_table_max_position
    recall = []
    for row in prompts:
        if position_limit is not None and row['input_tokens'] + settings['limit'] > position_limit:
            raise ValueError('Recall prompt plus generation exceeds the decoder position table')
        answer = reply(decoder, tokenizer, row['history'], row['question'], **settings)
        answer.pop('seconds')
        values = row['values']
        if answer['prompt_tokens'] != row['input_tokens']:
            raise ValueError('Token counts used for recall prompts differ at generation')
        passed = grade(answer, [re.escape(values['new']), re.escape(values['keep'])], [re.escape(values['old'])])
        recall.append({**row, **answer, 'passed': passed})
    ordinary = []
    for case in open_cases:
        history, turns = [], []
        for question, required, forbidden in case['turns']:
            answer = reply(decoder, tokenizer, history, question, **settings)
            answer.pop('seconds')
            turns.append({'question': question, **answer, 'passed': grade(answer, required, forbidden)})
            exchange(history, question, answer['answer'])
        ordinary.append({'case': case['name'], 'turns': turns, 'passed': all(turn['passed'] for turn in turns)})
    labels = dict.fromkeys(row['length'] for row in recall)
    return {'recall': recall, 'open': ordinary,
            'recall_passed': {label: sum(row['passed'] for row in recall if row['length'] == label)
                              for label in labels},
            'both_variants': {label: sum(all(row['passed'] for row in recall if row['case'] == case and row['length'] == label)
                                        for case in {row['case'] for row in recall}) for label in labels},
            'open_passed': sum(row['passed'] for row in ordinary),
            'open_turns_passed': sum(turn['passed'] for row in ordinary for turn in row['turns'])}


def cached_bytes(cache):
    storage = {}
    for index, layer in enumerate(cache.layers):
        tensors = (layer.keys, layer.values)
        if isinstance(layer, RecurrentDecodeLayer):
            tensors = layer.state.tensors()
        storage[index] = sum(tensor.untyped_storage().nbytes() for tensor in tensors)
    return storage


@torch.inference_mode()
def timed_decode(decoder, input_ids, steps):
    device = input_ids.device
    cache = DynamicCache(config=decoder.base.config)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    initial_bytes = torch.cuda.memory_allocated(device)
    start = perf_counter()
    logits = decoder(input_ids=input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1).logits
    token = logits[:, -1].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize(device)
    ready = perf_counter()
    prefill_bytes = cached_bytes(cache)
    decode_start = perf_counter()
    for _ in range(steps):
        logits = decoder(input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1).logits
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize(device)
    finish = perf_counter()
    return {'first_token_ms': 1000 * (ready - start), 'decode_ms_per_token': 1000 * (finish - decode_start) / steps,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device), 'initial_allocated_bytes': initial_bytes,
            'prefill_cache_bytes': prefill_bytes, 'final_cache_bytes': cached_bytes(cache)}


@contextmanager
def using(replacements, mode):
    with ExitStack() as installed:
        if mode != 'parent':
            for block, candidate in replacements.values():
                installed.callback(setattr, candidate, 'batched_prefill', candidate.batched_prefill)
                candidate.batched_prefill = mode == 'recurrent'
                installed.enter_context(replace(block, candidate))
        yield


def benchmark(decoder, tokenizer, fillers, replacements, repeats, *, extended=False, lengths=None):
    tokens = tokenizer.encode('\n'.join(fillers), add_special_tokens=False)
    if lengths is None:
        lengths = (96, 384, 640, 752, 1024, 1536) if extended else (96, 384, 640, 752)
    if decoder.position_table_max_position is not None and max(lengths) + 16 > decoder.position_table_max_position:
        raise ValueError('The position table is too short for the selected benchmark lengths')
    tokens *= (max(lengths) + len(tokens) - 1) // len(tokens)
    modes = ('parent', 'sequential', 'recurrent')
    timings = {}
    for length in lengths:
        ids = torch.tensor([tokens[:length]], device=decoder.base.device)
        rows = {name: [] for name in modes}
        for name in modes:
            with using(replacements, name):
                timed_decode(decoder, ids, 4)
        for iteration in range(repeats):
            for name in modes if iteration % 2 == 0 else reversed(modes):
                with using(replacements, name):
                    rows[name].append(timed_decode(decoder, ids, 16))
        timings[length] = {}
        for name, samples in rows.items():
            stats = {field: median(row[field] for row in samples)
                     for field in ('first_token_ms', 'decode_ms_per_token', 'peak_allocated_bytes', 'initial_allocated_bytes')}
            stats['prefill_cache_bytes'] = sum(samples[0]['prefill_cache_bytes'].values())
            stats['per_migrated_layer'] = {index: samples[0]['prefill_cache_bytes'][index] for index in replacements}
            timings[length][name] = {'summary': stats, 'samples': samples}
            print(length, name, stats, flush=True)
    return timings


def extend_positions(decoder, native_rotary, limit):
    """Add native coefficients beyond the existing editable table in this run."""
    backbone = decoder.base.model
    old = backbone.rotary_emb
    if not isinstance(old, SPORAPositionTable) or limit <= old.max_position or decoder.anchored_attention_layout:
        raise ValueError('Position extension needs a smaller editable table and full parent attention')
    new = SPORAPositionTable(native_rotary, limit)
    with torch.no_grad():
        new.coefficients.weight[:old.max_position].copy_(old.coefficients.weight)
    new.eval().requires_grad_(False)
    probe = torch.arange(old.max_position, device=new.coefficients.weight.device)[None]
    template = new.coefficients.weight.new_empty((1, 1, 1), dtype=torch.float16)
    if any(not torch.equal(a, b) for a, b in zip(old(template, probe), new(template, probe), strict=True)):
        raise ValueError('Extending the position table changed existing coefficients')
    backbone.rotary_emb = new
    decoder.position_table_max_position = limit


def parent_dialogue(path, checkpoint, protected, individual, positions):
    reference = json.loads(path.read_text(encoding='utf-8'))
    if (reference['individual_sha256'] != individual or reference['position_table'] != positions
            or any(reference['sources'].get(str(source)) != protected[source] for source in (checkpoint, SOURCE))):
        raise ValueError('Parent dialogue uses different weights, prompts, or positions')
    return reference['dialogue']['parent']


def main(args):
    inputs = [args.checkpoint, SOURCE]
    if args.parent_reference:
        inputs.append(args.parent_reference)
    inputs.extend(directory / name for directory in args.candidates for name in ('attention.pt', 'comparison.json'))
    protected = {path: sha(path) for path in inputs}
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    prompts = None if args.timing_only else recall_prompts(tokenizer, source, extended=args.extend)
    base = load_full_streamed()
    native_rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    replacements = load_recurrent_replacements(decoder, args.candidates, protected[args.checkpoint])
    if args.extend:
        extend_positions(decoder, native_rotary, 2048)
    result = {'device': torch.cuda.get_device_name(decoder.base.device), 'layers': list(replacements),
              'position_table': decoder.position_table_max_position,
              'sources': {str(path): value for path, value in protected.items()}, 'individual_sha256': individual}
    if not args.dialogue_only:
        fillers = source['filler'] + (source['extra_filler'] if args.extend else [])
        result['timing'] = benchmark(decoder, tokenizer, fillers, replacements, args.repeats, extended=args.extend,
                                     lengths=args.length)
    if not args.timing_only:
        result['dialogue'] = {}
        modes = ('parent', 'recurrent')
        if args.parent_reference:
            result['dialogue']['parent'] = parent_dialogue(args.parent_reference, args.checkpoint, protected,
                                                           individual, decoder.position_table_max_position)
            modes = ('recurrent',)
        for name in modes:
            with using(replacements, name):
                arm = run_dialogue(decoder, tokenizer, prompts, source['open'])
            result['dialogue'][name] = arm
            print(name, {key: value for key, value in arm.items() if key not in ('recall', 'open')}, flush=True)
    if (frozen_digest(runtime.organism, runtime.state) != individual
            or any(sha(path) != value for path, value in protected.items())):
        raise ValueError('An experiment input or individual state changed')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('candidates', nargs='+', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--length', type=int, action='append', help='Select a timing input length, repeatable')
    parser.add_argument('--parent-reference', type=Path, help='Reuse a verified complete-attention dialogue report')
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument('--timing-only', action='store_true')
    run_mode.add_argument('--dialogue-only', action='store_true')
    parser.add_argument('--extend', action='store_true', help='Extend native position coefficients to 2,048 for this run')
    options = parser.parse_args()
    if options.output.exists() or options.repeats < 1:
        parser.error('Use a new output path and positive repeat count')
    if options.length and (options.dialogue_only or any(length < 1 for length in options.length)
                           or len(set(options.length)) != len(options.length)):
        parser.error('Timing lengths must be distinct positive counts and require a timing run')
    if options.parent_reference and options.timing_only:
        parser.error('Parent references require a dialogue run')
    main(options)
