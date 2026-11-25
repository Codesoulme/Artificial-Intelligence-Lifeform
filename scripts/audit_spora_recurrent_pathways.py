"""Test whether far recurrent reads help when full attention cannot see early turns."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_spora_recurrent_history_ablation import summarize
from scripts.benchmark_spora_recurrent_dialogue import (
    MODEL, SOURCE, AutoTokenizer, encode, extend_positions, frozen_digest, load_full_streamed,
    load_integrated_runtime, recall_prompts, run_dialogue, sha, torch,
)
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.research_recurrent_decode import load_recurrent_replacements, replace, restrict_full_attention


BOUNDARY = 512
POSITIONS = 2048


def main(args):
    started = perf_counter()
    common = (args.checkpoint, SOURCE)
    candidate_paths = [folder / name for folder in args.candidates for name in ('attention.pt', 'comparison.json')]
    recovery_paths = tuple(args.recovery / name for name in ('training.json', 'result.json')) if args.recovery else ()
    protected = {path: sha(path) for path in (*common, *candidate_paths, args.reference, *recovery_paths)}
    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    expected_paths = common if args.recovery else (*common, *candidate_paths)
    if (reference['position_table'] != POSITIONS
            or any(reference['sources'].get(str(path)) != protected[path] for path in expected_paths)):
        raise ValueError('The benchmark reference used different inputs or position budget')
    original = reference['dialogue']['recurrent']['recall']
    if args.recovery:
        training, result = [json.loads(path.read_text(encoding='utf-8')) for path in recovery_paths]
        if (any(result.get(key) != value for key, value in training.items())
                or any(training['sources'].get(str(path)) != protected[path] for path in (*common, args.reference))
                or training['individual_sha256'] != reference['individual_sha256']):
            raise ValueError('The recovered results do not match the shared benchmark')
        original = result['original_cases_after']['recall']
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    prompts = [row for row in recall_prompts(tokenizer, source, extended=True) if row['length'] == 'extended']
    original = [row for row in original if row['length'] == 'extended']
    for prompt, saved in zip(prompts, original, strict=True):
        question = {'role': 'user', 'content': prompt['question']}
        evidence_length = len(encode(tokenizer, [*prompt['history'][:4],
                                                  question]))
        tokens = encode(tokenizer, [*prompt['history'], question])
        suffix = tokenizer.decode(tokens[BOUNDARY:], skip_special_tokens=True)
        if (any(saved[key] != value for key, value in prompt.items())
                or not evidence_length + 16 < BOUNDARY < len(tokens)
                or any(value in suffix for value in prompt['values'].values())):
            raise ValueError('The prefix boundary must follow the original events and precede the final question')
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, POSITIONS)
    torch.cuda.empty_cache()
    base.lm_head.to(base.device)
    individual = frozen_digest(runtime.organism, runtime.state)
    if individual != reference['individual_sha256']:
        raise ValueError('The benchmark reference used a different individual')
    candidates = load_recurrent_replacements(decoder, args.candidates, protected[args.checkpoint])
    if tuple(candidates) != tuple(reference['layers']):
        raise ValueError('The benchmark reference used a different layer placement')
    if args.recovery and any(protected[folder / 'attention.pt'] != training['candidate_hashes'][str(layer)]
                             for layer, folder in zip(candidates, args.candidates, strict=True)):
        raise ValueError('The loaded candidates and recovered outputs use different weights')
    results = {'unrestricted': {'rows': original, 'summary': summarize(original)['extended']}}
    for label, selected, far in (('restricted_parent', {}, False),
                                 ('restricted_with_far', candidates, True),
                                 ('restricted_without_far', candidates, False)):
        with ExitStack() as hooks:
            hooks.enter_context(restrict_full_attention(decoder, BOUNDARY, selected))
            for block, candidate in selected.values():
                hooks.callback(setattr, candidate, 'include_far', candidate.include_far)
                candidate.include_far = far
                hooks.enter_context(replace(block, candidate))
            rows = run_dialogue(decoder, tokenizer, prompts, [])['recall']
        results[label] = {'rows': rows, 'summary': summarize(rows)['extended']}
        print(label, results[label]['summary'], flush=True)
    if (individual != frozen_digest(runtime.organism, runtime.state)
            or any(sha(path) != value for path, value in protected.items())):
        raise ValueError('A probe input or individual changed')
    store_json(args.output, {'conditions': results, 'boundary': BOUNDARY, 'positions': POSITIONS,
                             'individual_sha256': individual, 'layers': list(candidates),
                             'sources': {str(path): value for path, value in protected.items()},
                             'seconds': perf_counter() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('candidates', nargs=2, type=Path)
    parser.add_argument('--recovery', type=Path, help='Use verified unrestricted outputs from recovery training')
    options = parser.parse_args()
    if options.output.exists():
        parser.error('Choose a new output path')
    main(options)
