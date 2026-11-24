"""Compare single-field and joint answers on the same dialogue histories."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import re

from scripts.benchmark_spora_recurrent_dialogue import (
    MODEL, SOURCE, AutoTokenizer, encode, extend_positions, frozen_digest, grade, load_full_streamed,
    load_integrated_runtime, recall_prompts, reply, run_dialogue, settings_for, sha, torch,
)
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.research_recurrent_decode import load_recurrent_replacements, replace


ROOT = SOURCE.parent
SCENARIOS = ROOT / 'recurrent-phase-transfer-v1.json'
QUESTIONS = ROOT / 'recurrent-single-field-questions-v1.json'
FIELDS = ('new', 'keep')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summarize(rows):
    return {'count': len(rows),
            'new': sum(row['single']['new']['passed'] for row in rows),
            'keep': sum(row['single']['keep']['passed'] for row in rows),
            'both_separate': sum(row['both_separate'] for row in rows),
            'joint': sum(row['joint']['passed'] for row in rows),
            'separate_only': sum(row['both_separate'] and not row['joint']['passed'] for row in rows),
            'joint_only': sum(row['joint']['passed'] and not row['both_separate'] for row in rows)}


def compare(decoder, tokenizer, prompts, joint, questions):
    settings = settings_for(decoder, tokenizer, 48)
    rows = []
    for prompt, existing in zip(prompts, joint, strict=True):
        if any(existing[key] != value for key, value in prompt.items()):
            raise ValueError('A joint answer was generated from a different history or question')
        single = {}
        values = prompt['values']
        for field in FIELDS:
            question = questions[prompt['case']][field]
            answer = reply(decoder, tokenizer, prompt['history'], question, **settings)
            answer.pop('seconds')
            single[field] = {'question': question, **answer,
                             'passed': grade(answer, [re.escape(values[field])], [re.escape(values['old'])])}
        rows.append({'case': prompt['case'], 'variant': prompt['variant'], 'values': values,
                     'joint': {key: existing[key] for key in ('question', 'answer', 'eos', 'input_tokens', 'passed')},
                     'single': single, 'both_separate': all(result['passed'] for result in single.values())})
    return {'rows': rows, 'summary': summarize(rows)}


def planned_answers(decoder, tokenizer, prompts, questions):
    for mode in ('two_lines', 'enumerated_fields'):
        requests = []
        for prompt in prompts:
            question = prompt['question'] + '\n请分两行回答，每行只给一个所问的值。'
            if mode == 'enumerated_fields':
                fields = questions[prompt['case']]
                question += f'\n1. {fields["new"]}\n2. {fields["keep"]}'
            length = len(encode(tokenizer, [*prompt['history'], {'role': 'user', 'content': question}]))
            requests.append({**prompt, 'question': question, 'input_tokens': length})
        generated = run_dialogue(decoder, tokenizer, requests, [])
        print(mode, generated['recall_passed'], flush=True)
        yield mode, generated


def main(args):
    model_inputs = (args.checkpoint, SOURCE, SCENARIOS,
                    *(folder / name for folder in args.candidates for name in ('attention.pt', 'comparison.json')))
    protected = {path: sha(path) for path in (*model_inputs, QUESTIONS, args.parent_report, args.recurrent_report)}
    source = read(SOURCE)
    source['recall'] = read(SCENARIOS)['recall']
    questions = read(QUESTIONS)
    if questions.keys() != {case['name'] for case in source['recall']} or any(
        row.keys() != set(FIELDS) for row in questions.values()
    ):
        raise ValueError('Every transfer case needs two matching single-field questions')
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    prompts = recall_prompts(tokenizer, source, extended=True)
    short = [prompt for prompt in prompts if prompt['length'] == 'short']
    extended = [prompt for prompt in prompts if prompt['length'] == 'extended']
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    torch.cuda.empty_cache()
    base.lm_head.to(base.device)
    individual = frozen_digest(runtime.organism, runtime.state)

    joint = {}
    expected = {path.resolve(): protected[path] for path in model_inputs}
    for name, path in (('parent', args.parent_report), ('recurrent', args.recurrent_report)):
        archive = read(path)
        sources = {Path(source).resolve(): digest for source, digest in archive['sources'].items()}
        if sources != expected or archive['individual_sha256'] != individual:
            raise ValueError('The joint archive uses different inputs or individual')
        joint[name] = [row for row in archive['conditions'][name]['recall'] if row['length'] == 'extended']

    results = {}
    if not args.answer_plan:
        print('Full attention, short histories', flush=True)
        results['parent_short'] = compare(decoder, tokenizer, short,
                                          run_dialogue(decoder, tokenizer, short, [])['recall'], questions)
        print('Full attention, long histories', flush=True)
        results['parent_extended'] = compare(decoder, tokenizer, extended, joint['parent'], questions)
    else:
        for prompt, existing in zip(extended, joint['recurrent'], strict=True):
            if any(existing[key] != value for key, value in prompt.items()):
                raise ValueError('The joint baseline used a different history or question')
    print('Recurrent attention, long histories', flush=True)
    candidates = load_recurrent_replacements(decoder, args.candidates, protected[args.checkpoint])
    with ExitStack() as stack:
        for block, candidate in candidates.values():
            stack.enter_context(replace(block, candidate))
        if args.answer_plan:
            results['original'] = {'recall': joint['recurrent'],
                                   'recall_passed': {'extended': sum(row['passed'] for row in joint['recurrent'])}}
            results.update(planned_answers(decoder, tokenizer, extended, questions))
        else:
            results['recurrent_extended'] = compare(decoder, tokenizer, extended, joint['recurrent'], questions)

    if (frozen_digest(runtime.organism, runtime.state) != individual
            or any(sha(path) != digest for path, digest in protected.items())):
        raise ValueError('A diagnostic source or individual changed')
    store_json(args.output, {'sources': {str(path): digest for path, digest in protected.items()},
                             'individual_sha256': individual, 'conditions': results})
    if not args.answer_plan:
        for name, result in results.items():
            print(name, result['summary'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('candidates', type=Path, nargs=2)
    parser.add_argument('--parent-report', type=Path, required=True)
    parser.add_argument('--recurrent-report', type=Path, required=True)
    parser.add_argument('--answer-plan', action='store_true',
                        help='Compare formatting and explicit field guidance with the recurrent joint baseline')
    options = parser.parse_args()
    if options.output.exists() or not options.output.parent.is_dir():
        parser.error('Choose a new result path in an existing directory')
    main(options)
