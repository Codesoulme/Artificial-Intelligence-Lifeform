"""Measure whether changing a past value flips the decoder's token preference."""

import argparse
from contextlib import ExitStack
from functools import partial
import json
from pathlib import Path

from scripts.benchmark_spora_recurrent_dialogue import (
    MODEL, SOURCE, AutoTokenizer, encode, extend_positions, frozen_digest, load_full_streamed,
    load_integrated_runtime, recall_prompts, run_dialogue, sha, torch,
)
from scripts.train_spora_recurrent_counterfactual import evaluation_pairs, store_json
from spora.research_recurrent_decode import load_recurrent_replacements, replace


def contrast(first, second, eos):
    """Compare at the first differing token, including end-of-answer when needed."""
    for index, (left, right) in enumerate(zip([*first, eos], [*second, eos])):
        if left != right:
            return first[:index], left, right
    raise ValueError('The two values have identical token sequences')


def make_examples(tokenizer, pairs):
    examples = []
    for left, right in zip(pairs[::2], pairs[1::2], strict=True):
        values = [row['values']['new'] for row in (left, right)]
        token_ids = [tokenizer.encode(value, add_special_tokens=False) for value in values]
        prefix, first, second = contrast(*token_ids, tokenizer.eos_token_id)
        prompts = []
        for row, target, foil in ((left, first, second), (right, second, first)):
            messages = [*row['history'], {'role': 'user', 'content': row['question']}]
            prompt = encode(tokenizer, messages)
            if len(prompt) != row['input_tokens']:
                raise ValueError('The diagnostic prompt differs from the generation evaluation')
            prompts.append((prompt + prefix, target, foil))
        examples.append((left['case'], left['length'], values, prompts))
    return examples


@torch.inference_mode()
def measure(decoder, examples):
    rows = []
    for case, length, values, prompts in examples:
        scores = []
        for ids, target, foil in prompts:
            if len(ids) > decoder.position_table_max_position:
                raise ValueError('The matching prefix exceeds the position table')
            input_ids = torch.tensor([ids], device=decoder.base.device)
            logits = decoder(input_ids=input_ids, logits_to_keep=1, use_cache=True).logits[0, -1].float()
            correct = logits[target]
            scores.append({'margin': float(correct - logits[foil]),
                           'rank': int((logits > correct).sum()) + 1})
        rows.append({'case': case, 'length': length, 'values': values, 'scores': scores})
    return rows


def summarize(rows):
    results = {}
    for length in dict.fromkeys(row['length'] for row in rows):
        pairs = [row['scores'] for row in rows if row['length'] == length]
        scores = [score for pair in pairs for score in pair]
        margins = [score['margin'] for score in scores]
        results[length] = {'both_preferred': sum(all(score['margin'] > 0 for score in pair) for pair in pairs),
                           'preferred': sum(margin > 0 for margin in margins),
                           'mean_margin': sum(margins) / len(margins),
                           'top_10': sum(score['rank'] <= 10 for score in scores),
                           'count': len(scores)}
    return results


def read_in_phase(phase, module, _args, kwargs):
    first_position = int(kwargs['cache_position'][0])
    module.include_far = first_position == 0 if phase == 'prefill' else first_position > 0


def main(args):
    paths = [args.checkpoint, SOURCE, *(folder / name for folder in args.candidates
                                       for name in ('attention.pt', 'comparison.json'))]
    if args.transfer:
        paths.append(args.transfer)
    protected = {path: sha(path) for path in paths}
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    generative = args.generate_extended or args.phase_ablation or args.transfer is not None
    lengths = ('long', 'extended') if args.transfer else ('extended',)
    if args.transfer:
        source['recall'] = json.loads(args.transfer.read_text(encoding='utf-8'))['recall']
    if generative:
        prompts = [row for row in recall_prompts(tokenizer, source, extended=True) if row['length'] in lengths]
    else:
        _, pairs = evaluation_pairs(tokenizer, source)
        examples = make_examples(tokenizer, pairs)
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    torch.cuda.empty_cache()
    base.lm_head.to(base.device)
    individual = frozen_digest(runtime.organism, runtime.state)
    results = {}
    if args.parent_only:
        arms = [('parent', (), True)]
    else:
        candidates = load_recurrent_replacements(decoder, args.candidates, protected[args.checkpoint]).values()
        arms = [('recurrent', candidates, True), ('without_far_read', candidates, False)]
        if args.transfer:
            arms = [('parent', (), True), ('recurrent', candidates, True), ('decode', candidates, 'decode')]
        elif args.phase_ablation:
            arms = [(phase, candidates, phase) for phase in ('prefill', 'decode')]
        elif not generative:
            arms.insert(0, ('parent', (), True))
    for label, selected, read in arms:
        print(f'Measuring {label}', flush=True)
        with ExitStack() as stack:
            for block, candidate in selected:
                stack.callback(setattr, candidate, 'include_far', candidate.include_far)
                if isinstance(read, str):
                    hook = candidate.register_forward_pre_hook(partial(read_in_phase, read), with_kwargs=True)
                    stack.callback(hook.remove)
                else:
                    candidate.include_far = read
                stack.enter_context(replace(block, candidate))
            if generative:
                dialogue = run_dialogue(decoder, tokenizer, prompts, [])
                results[label] = dialogue
                print(label, dialogue['recall_passed'], flush=True)
            else:
                rows = measure(decoder, examples)
                results[label] = {'rows': rows, 'summary': summarize(rows)}
                print(label, results[label]['summary'], flush=True)
    if (frozen_digest(runtime.organism, runtime.state) != individual
            or any(sha(path) != digest for path, digest in protected.items())):
        raise ValueError('A diagnostic source or individual changed')
    store_json(args.output, {'sources': {str(path): digest for path, digest in protected.items()},
                             'individual_sha256': individual, 'conditions': results})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('candidates', type=Path, nargs=2)
    generation = parser.add_mutually_exclusive_group()
    generation.add_argument('--generate-extended', action='store_true',
                            help='Compare full answers with and without far reads on the longest cases')
    generation.add_argument('--phase-ablation', action='store_true',
                            help='Compare far reads during prefill or cached decoding on the longest cases')
    generation.add_argument('--transfer', type=Path,
                            help='Compare parent, both-phase and decode-only reads on new recall scenarios')
    parser.add_argument('--parent-only', action='store_true',
                        help='Run only the native-attention transfer control')
    arguments = parser.parse_args()
    if arguments.parent_only and arguments.transfer is None:
        parser.error('--parent-only requires --transfer')
    if arguments.output.exists() or not arguments.output.parent.is_dir():
        parser.error('Choose a new result path in an existing directory')
    main(arguments)
