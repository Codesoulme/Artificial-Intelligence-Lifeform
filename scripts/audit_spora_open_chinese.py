"""Compare native and integrated Chinese rollout with neutral and resolved context."""

import argparse
from collections import Counter
from functools import partial
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_named_memory_qwen_structure_control import CHECKPOINT_SHA
from scripts.audit_spora_chat_backbones import FILES, reply
from scripts.audit_spora_dialogue_backbone_pivot import compare_qwen
from scripts.audit_spora_dialogue_evidence import generate, store_json
from scripts.deepseek_local_baseline import AutoTokenizer, MODEL, WEIGHT_SHA, sha, torch
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_open_chinese_cases import CASES
from spora.integrated_checkpoint import load_integrated
from spora.integrated_open_base import IntegratedOpenBase
from spora.language_training import frozen_digest


BUDGET = 72
POSITIONS = 384
EXTENDED_BUDGET = 144
EXTENDED_POSITIONS = 512


def evaluate(answer):
    cases, counts = [], Counter()
    for name, prompts, resolved, required, forbidden in CASES:
        history, rollout = [], []
        for prompt in prompts:
            result = answer(history, prompt)
            history.append((prompt, result['answer']))
            rollout.append(result)
        conditions = {'raw' if resolved else 'ordinary': rollout[-1]}
        if resolved:
            conditions['neutral'] = answer([(prompt, '收到。') for prompt in prompts[:-1]], prompts[-1])
            conditions['resolved'] = answer([], resolved)
        for mode, row in conditions.items():
            text = row['answer']
            missing = [group for group in required if not any(term in text for term in group)]
            abandoned = [term for term in forbidden if term in text]
            row['checks'] = {'missing': missing, 'abandoned': abandoned,
                             'surface_pass': bool(text and not missing and not abandoned)}
            counts.update({f'{mode}_cases': 1, f'{mode}_surface_pass': row['checks']['surface_pass'],
                           f'{mode}_abandoned': bool(abandoned), f'{mode}_eos': row['eos']})
        cases.append({'name': name, 'turns': prompts, 'resolved_prompt': resolved, 'required': required,
                      'forbidden': forbidden, 'rollout': rollout, 'conditions': conditions})
        print(name, {mode: row['checks']['surface_pass'] for mode, row in conditions.items()}, flush=True)
    return {'cases': cases, 'summary': dict(counts)}


def qwen_answer(model, tokenizer, eos, budget, positions=POSITIONS):
    complete = partial(reply, model, tokenizer, device='cuda', deepseek=False, limit=budget, eos=eos)

    def answer(history, prompt):
        messages = [{'role': role, 'content': text} for user, assistant in history
                    for role, text in (('user', user), ('assistant', assistant))]
        result = complete(messages, prompt)
        if result['prompt_tokens'] + budget > positions:
            raise ValueError('Qwen prompt exceeded the shared position budget')
        return result

    return answer


def qwen(model, tokenizer, eos):
    print(type(model).__name__, flush=True)
    return evaluate(qwen_answer(model, tokenizer, eos, BUDGET))


def extend_qwen(previous, model, tokenizer, eos):
    # Only the integrated arm is needed for the final-length intervention.
    if not isinstance(model, IntegratedOpenBase):
        return None
    answer = qwen_answer(model, tokenizer, eos, EXTENDED_BUDGET, EXTENDED_POSITIONS)
    outputs = []
    for case in previous['integrated']['cases']:
        prompts = case['turns']
        for mode in ('raw', 'resolved', 'ordinary'):
            short = case['conditions'].get(mode)
            if short is None or short['eos']:
                continue
            if short['generated_tokens'] != BUDGET:
                raise ValueError('Expected a reply that reached the original token cap')
            history = list(zip(prompts[:-1], (row['answer'] for row in case['rollout'][:-1]), strict=True))
            prompt = prompts[-1]
            if mode == 'resolved':
                history, prompt = [], case['resolved_prompt']
            extended = answer(history, prompt)
            if (extended['prompt_tokens'] != short['prompt_tokens']
                    or not extended['answer'].startswith(short['answer'].rstrip('\ufffd').rstrip())):
                raise ValueError(f'Length intervention changed the prefix for {case["name"]} {mode}')
            outputs.append({'name': case['name'], 'mode': mode, 'short': short, 'extended': extended})
            print(case['name'], mode, extended['eos'], flush=True)
    return outputs


def deepseek(checkpoint):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    paths = (checkpoint, *(MODEL / name for name in FILES))
    sources = {str(path): sha(path) for path in paths}
    if sources[str(MODEL / 'model.safetensors')] != WEIGHT_SHA or sources[str(checkpoint)] != CHECKPOINT_SHA:
        raise ValueError('Expected the pinned DeepSeek backbone and SPORA checkpoint')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed()

    def comparisons(model):
        print(type(model).__name__, flush=True)

        def answer(history, prompt):
            result = generate(model, tokenizer, history, prompt, POSITIONS, budget=BUDGET)
            if result['retained'] != len(history) or result['budget'] != BUDGET:
                raise ValueError('DeepSeek prompt or reply budget was truncated')
            return result

        return evaluate(answer)

    native = comparisons(base)
    decoder, organism, state = load_integrated(checkpoint, base, MODEL, individual_device='cpu')
    decoder.eval().requires_grad_(False)
    before = frozen_digest(organism, state)
    integrated = comparisons(decoder)
    if frozen_digest(organism, state) != before:
        raise ValueError('Dialogue evaluation changed the isolated individual')
    if any(sha(Path(path)) != digest for path, digest in sources.items()):
        raise ValueError('A DeepSeek source changed during evaluation')
    return {'native': native, 'integrated': integrated, 'sources': sources, 'individual_sha256': before,
            'seconds': perf_counter() - started,
            'peak_bytes': torch.cuda.max_memory_allocated()}


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f'{args.model}.json'
    if destination.exists():
        raise FileExistsError(destination)
    manifest = Path(__file__).with_name('spora_open_chinese_cases.py')
    digest = sha(manifest)
    if args.model == 'qwen':
        result = compare_qwen(args.checkpoint, args.donor, args.calibration, qwen)
    elif args.model == 'qwen-extended':
        baseline = args.output / 'qwen.json'
        baseline_sha = sha(baseline)
        previous = json.loads(baseline.read_text(encoding='utf-8'))
        if (previous['cases_sha256'] != digest or previous['budget'] != BUDGET
                or previous['positions'] != POSITIONS):
            raise ValueError('Expected the original 72-token run on this case manifest')
        result = compare_qwen(args.checkpoint, args.donor, args.calibration, partial(extend_qwen, previous))
        binding = ('sources', 'adapter_sha256', 'calibration_sha256', 'individual_sha256')
        if any(result[key] != previous[key] for key in binding) or sha(baseline) != baseline_sha:
            raise ValueError('Length intervention used different sources or individual state')
        result.update(baseline_sha256=baseline_sha, extended_budget=EXTENDED_BUDGET,
                      extended_positions=EXTENDED_POSITIONS)
    else:
        result = deepseek(args.checkpoint)
    if sha(manifest) != digest:
        raise ValueError('Case manifest changed during evaluation')
    store_json(destination, {**result, 'cases_sha256': digest, 'budget': BUDGET, 'positions': POSITIONS})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=('qwen', 'deepseek', 'qwen-extended'))
    parser.add_argument('output', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/named-memory-layer9-packed-reply-v1/packed.pt'))
    parser.add_argument('--donor', type=Path, default=Path('D:/SPORA-runtime/models/Qwen2.5-1.5B-Instruct-989aa798'))
    parser.add_argument('--calibration', type=Path, default=Path('runs/named-memory-qwen-structure-control-v2/result.json'))
    main(parser.parse_args())
