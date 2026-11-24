"""Audit span continuation and completed-field coverage in the native SPORA decoder."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import re
from time import perf_counter

from scripts.train_spora_memory_conditioned_reply import (
    DATA, HELD, MODEL, SCENARIOS, SOURCE, TRANSFER_FIELDS, AutoTokenizer, evaluate, extend_positions, frozen_digest, heldout,
    install, load_full_streamed, load_integrated_runtime, load_recurrent_replacements, read, recall_prompts,
    records, sha, state_for, store_json, summary, torch,
)
from spora.research_dialogue_working_state import CoveredWorkingTurnReader, MixtureWorkingTurnReader


ADJACENT_REPEAT = re.compile(r'([\u3400-\u9fff]{2,8})\1|([\u3400-\u9fff])\2{2,}')


def repetition(text, fields):
    return bool(ADJACENT_REPEAT.search(text) or any(
        text.count(value) > 1 or value + value[-1] in text or value[0] + value in text
        for value in fields
    ))


def measures(rows):
    result = summary(rows)
    for scope in result:
        clean_pairs = 0
        for row in rows:
            outputs = row['outputs'][scope]
            targets = (row['values']['new'], row['replacement'])
            clean = {}
            for mode, update in zip(('matched', 'swapped'), targets, strict=True):
                answer = outputs[mode]
                repeat = repetition(answer['answer'], (update, row['values']['keep']))
                clean[mode] = answer['passed'] and not repeat
                metrics = result[scope][mode]
                metrics['repeats'] = metrics.get('repeats', 0) + repeat
                metrics['complete_without_detected_repeats'] = metrics.get('complete_without_detected_repeats', 0) + clean[mode]
            clean_pairs += all(clean.values())
        result[scope]['both_without_detected_repeats'] = clean_pairs
    return result


def fresh_cases(path):
    records = read(path)['recall']
    if set(TRANSFER_FIELDS) != {case['name'] for case in records} or any(len(case['values']) != 2 for case in records):
        raise ValueError('Each transfer family needs two new value sets')
    previous = DATA.read_text(encoding='utf-8') + SCENARIOS.read_text(encoding='utf-8')
    rows = []
    for case in records:
        if case['question'] in previous or any(
                value in previous or value in case['question'] for group in case['values'] for value in group):
            raise ValueError('The new question or value leaks into a previous source or the current question')
        for variant, values in enumerate(case['values']):
            rows.append({'case': case['name'], 'variant': variant, 'question': case['question'], 'history': [],
                         'values': dict(zip(('old', 'new', 'keep'), values, strict=True))})
    return rows


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    weight_path = args.training / 'paired.pt'
    archive_path = args.training / 'result.json'
    archive = read(archive_path)
    model_inputs = (args.checkpoint, DATA, SOURCE, SCENARIOS, *(folder / name for folder in args.candidates
                                                               for name in ('attention.pt', 'comparison.json')))
    protected = {path: sha(path) for path in (*model_inputs, weight_path, archive_path)}
    if args.fresh:
        protected[args.fresh] = sha(args.fresh)
    if ({str(path): protected[path] for path in model_inputs} != archive['sources']
            or protected[weight_path] != archive['arms']['paired']['sha256']):
        raise ValueError('The experiment inputs differ from the paired training run')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    if args.fresh:
        held = fresh_cases(args.fresh)
        scopes = ('state_only',)
        modes = ('baseline', 'coverage', 'span_only', 'coverage_and_span')
    else:
        scenarios = {**read(SOURCE), 'recall': read(SCENARIOS)['recall']}
        held = [row for row in recall_prompts(tokenizer, scenarios, extended=True) if row['length'] == 'extended']
        scopes = ('long', 'state_only')
        modes = ('coverage', 'coverage_and_span')
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    torch.cuda.empty_cache()
    base.lm_head.to(base.device)
    individual = frozen_digest(runtime.organism, runtime.state)
    if individual != archive['individual_sha256']:
        raise ValueError('The individual differs from the paired training run')
    candidates = load_recurrent_replacements(decoder, args.candidates, protected[args.checkpoint])
    weights = torch.load(weight_path, map_location='cpu', weights_only=True)
    if weights['mode'] != 'paired':
        raise ValueError('Expected an adapter trained on paired states')
    result = {}
    if not args.fresh:
        baseline = archive['arms']['paired']['rows']
        result['baseline'] = {'summary': measures(baseline), 'rows': baseline}
    state = state_for(tokenizer, {'place': held[0]['values']['keep']})
    memory_cases = records(runtime.organism, HELD)
    with ExitStack() as stack:
        install(stack, candidates)
        for name in modes:
            width = base.config.hidden_size
            reader = (MixtureWorkingTurnReader(width) if name == 'baseline' else
                      CoveredWorkingTurnReader(width, follow_span=name != 'coverage',
                                               suppress_completed=name != 'span_only'))
            reader.load_state_dict(weights['reader'])
            reader.to(base.device).eval().requires_grad_(False)
            print(f'Evaluating {name} on {len(held)} paired prompts', flush=True)
            rows = heldout(decoder, tokenizer, reader, held, scopes=scopes)
            with reader.active(decoder, state):
                memory = evaluate(decoder, tokenizer, runtime.organism, memory_cases)
            if memory != archive['memory']:
                raise ValueError('The coverage reader changed durable memory output')
            metrics = measures(rows)
            result[name] = {'summary': metrics, 'rows': rows}
            print(name, metrics, flush=True)
    if (frozen_digest(runtime.organism, runtime.state) != individual
            or any(sha(path) != digest for path, digest in protected.items())):
        raise ValueError('The experiment changed a frozen input or individual')
    store_json(args.output, {'sources': {str(path): digest for path, digest in protected.items()},
                             'individual_sha256': individual, 'conditions': result,
                             'memory': memory, 'seconds': perf_counter() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('training', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('candidates', type=Path, nargs=2)
    parser.add_argument('--fresh', type=Path, help='Additional question-only paired holdout')
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error('Choose a new output path in an existing directory')
    main(args)
