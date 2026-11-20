"""Run a manual-review holdout comparison for the original and residual models."""

import argparse
import json
import time
from pathlib import Path

from audit_awake_language_layers import load_layer_pair
from deepseek_output_aligned_memory import MODEL, ROOT, SOURCE, read, sha, torch, write
from train_awake_original_multilayer_dialogue_residual_20261120 import attach_adapters
from train_awake_replacement_replay import switch
from audit_awake_reasoning_residual_compare_20261120 import load_candidate


DATA = ROOT / 'data/conversation-development-v2.json'


def load_scenarios():
    payload = json.loads(DATA.read_text(encoding='utf-8'))
    scenarios = payload.get('scenarios')
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError('Missing holdout scenarios')
    for scenario in scenarios:
        if not isinstance(scenario.get('name'), str) or not isinstance(scenario.get('turns'), list):
            raise ValueError('Malformed holdout scenario')
        if not scenario['turns']:
            raise ValueError('Empty holdout scenario')
    return scenarios


@torch.inference_mode()
def run_dialogue(decoder, tokenizer, scenario, *, reasoning, limit, seed):
    history = []
    rows = []
    for turn_index, question in enumerate(scenario['turns']):
        history.append({'role': 'user', 'content': question})
        formatted = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        if reasoning:
            prompt = tokenizer.encode(formatted + '\n', add_special_tokens=False)
        else:
            prompt = tokenizer.encode(formatted + '</think>\n\n', add_special_tokens=False)
        if len(prompt) + limit > decoder.base.config.max_position_embeddings:
            raise ValueError('Holdout prompt exceeds model position limit')
        torch.manual_seed(seed + turn_index)
        inputs = torch.tensor([prompt], device='cuda')
        generated = decoder.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            do_sample=reasoning,
            temperature=.6 if reasoning else None,
            top_p=.95 if reasoning else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=limit,
            use_cache=True,
        )
        tokens = generated[0, len(prompt):].tolist()
        raw = tokenizer.decode(tokens, skip_special_tokens=True)
        answer = raw.partition('</think>')[2].strip() if reasoning else raw.strip()
        rows.append({
            'turn': turn_index + 1,
            'question': question,
            'answer': answer,
            'raw': raw,
            'prompt_tokens': len(prompt),
            'generated_tokens': len(tokens),
            'eos': bool(tokens and tokens[-1] == tokenizer.eos_token_id),
        })
        history.append({'role': 'assistant', 'content': answer})
    return rows


def main(candidate_dir, output):
    if output.exists():
        raise FileExistsError(output)
    scenarios = load_scenarios()
    candidate_dir = candidate_dir.resolve()
    candidate_files = [candidate_dir / 'result.json',
                       *(candidate_dir / f'layer-{index}-residuals.pt' for index in (9, 15, 21, 27))]
    protected = {path: sha(path) for path in (SOURCE, MODEL / 'model.safetensors', DATA, *candidate_files)}
    started = time.perf_counter()
    decoder, tokenizer, originals, _, _, _ = load_layer_pair()
    switch(decoder, originals)
    adapters = attach_adapters(decoder)
    settings = (
        {'name': 'closed_64', 'reasoning': False, 'limit': 64},
        {'name': 'open_128', 'reasoning': True, 'limit': 128},
    )
    rows = []
    for variant in ('original', 'residual_candidate'):
        if variant == 'residual_candidate':
            load_candidate(candidate_dir, adapters)
        for setting in settings:
            for scenario_index, scenario in enumerate(scenarios):
                dialogue = run_dialogue(
                    decoder, tokenizer, scenario,
                    reasoning=setting['reasoning'], limit=setting['limit'],
                    seed=91217 + scenario_index * 100 + (1 if setting['reasoning'] else 0),
                )
                rows.append({
                    'variant': variant,
                    'mode': setting['name'],
                    'scenario': scenario['name'],
                    'turns': dialogue,
                })
                print(variant, setting['name'], scenario['name'],
                      [row['answer'] for row in dialogue], flush=True)
    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError('A protected source or holdout changed during the audit')
    write(output, {
        'format': 'spora-original-multilayer-dialogue-residual-holdout-v1',
        'candidate_dir': str(candidate_dir),
        'data': str(DATA),
        'settings': settings,
        'rows': rows,
        'sources': {str(path): digest for path, digest in protected.items()},
        'seconds': time.perf_counter() - started,
        'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
        'limitations': [
            'This is a development holdout for manual review, not final acceptance.',
            'No automatic semantic score is assigned; inspect every answer with the rubric.',
            'The holdout is never added to the training corpus.',
        ],
    })
    print('Holdout report written', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.candidate_dir.is_dir():
        parser.error('Candidate directory must exist')
    main(args.candidate_dir, args.output)
