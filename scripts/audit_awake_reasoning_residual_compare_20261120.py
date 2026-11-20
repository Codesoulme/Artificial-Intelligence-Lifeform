"""Compare the original backbone with a zero initialized residual candidate.

This is a diagnostic only. It keeps the pretrained decoder and tokenizer fixed,
then evaluates the same conversations in the model's natural reasoning format
and in the legacy closed-thinking format.
"""

import argparse
import time
from pathlib import Path

from audit_awake_language_layers import load_layer_pair
from audit_awake_multiturn_layers import CASES, conversation
from deepseek_output_aligned_memory import MODEL, ROOT, SOURCE, read, sha, torch, write
from train_awake_original_multilayer_dialogue_residual_20261120 import LAYERS, attach_adapters
from train_awake_replacement_replay import switch


def load_candidate(candidate_dir, adapters):
    candidate_dir = candidate_dir.resolve()
    report = read(candidate_dir / 'result.json')
    if report.get('format') not in (
            'spora-original-multilayer-dialogue-residual-diagnostic-v1',
            'spora-original-natural-dialogue-residual-diagnostic-v1'):
        raise ValueError('Unexpected candidate format')
    if tuple(report.get('layers', ())) != tuple(LAYERS):
        raise ValueError('Candidate layer list does not match the loader')
    for index, group in adapters.items():
        path = candidate_dir / f'layer-{index}-residuals.pt'
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if (payload.get('format') not in (
                'spora-original-multilayer-dialogue-residual-v1',
                'spora-original-natural-dialogue-residual-v1')
                or payload.get('layer') != index or payload.get('rank') != 8
                or payload.get('source_sha256') != sha(SOURCE)
                or payload.get('base_sha256') != sha(MODEL / 'model.safetensors')):
            raise ValueError(f'Invalid residual provenance for layer {index}')
        for name, module in group.items():
            module.load_state_dict({
                'base.weight': module.base.weight.detach().cpu(),
                'down': payload['state'][name]['down'],
                'up': payload['state'][name]['up'],
            }, strict=False)
            module.requires_grad_(False).eval()


def summarize(rows, mode):
    selected = [row for row in rows if row['mode'] == mode]
    turns = [turn for row in selected for turn in row['turns']]
    return {
        'cases': len(selected),
        'turns': len(turns),
        'correct_turns': sum(turn['correct'] for turn in turns),
        'natural_reasoning_closed': sum(turn.get('reasoning_closed', False) for turn in turns),
        'eos': sum(turn['eos'] for turn in turns),
    }


def evaluate(decoder, tokenizer, position_limit, variant, settings):
    rows = []
    for index, case in enumerate(CASES):
        torch.manual_seed(70393 + index)
        turns = conversation(decoder, tokenizer, case, position_limit, decoding=settings)
        rows.append({'case': index, 'variant': variant, 'mode': settings['name'], 'turns': turns})
        print(variant, settings['name'], index,
              [(turn['answer'], turn['correct'], turn['eos']) for turn in turns], flush=True)
    return rows


def main(candidate_dir, output):
    if output.exists():
        raise FileExistsError(output)
    candidate_dir = candidate_dir.resolve()
    candidate_files = [candidate_dir / 'result.json',
                       *(candidate_dir / f'layer-{index}-residuals.pt' for index in LAYERS)]
    protected = {path: sha(path) for path in (SOURCE, MODEL / 'model.safetensors', *candidate_files)}
    started = time.perf_counter()
    decoder, tokenizer, originals, _, _, _ = load_layer_pair()
    switch(decoder, originals)
    adapters = attach_adapters(decoder)
    position_limit = decoder.base.config.max_position_embeddings
    settings = (
        {'name': 'closed_64', 'limit': 64},
        {'name': 'open_192', 'reasoning': True, 'limit': 192},
    )
    rows = []
    for variant in ('original', 'residual_candidate'):
        if variant == 'residual_candidate':
            load_candidate(candidate_dir, adapters)
        for configuration in settings:
            rows.extend(evaluate(decoder, tokenizer, position_limit, variant, configuration))
    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError('A protected source or candidate changed during the audit')
    summary = {
        f'{variant}_{mode}': summarize([row for row in rows if row['variant'] == variant], mode)
        for variant in ('original', 'residual_candidate')
        for mode in ('closed_64', 'open_192')
    }
    write(output, {
        'format': 'spora-original-multilayer-dialogue-residual-reasoning-compare-v1',
        'candidate_dir': str(candidate_dir),
        'settings': settings,
        'summary': summary,
        'rows': rows,
        'sources': {str(path): digest for path, digest in protected.items()},
        'seconds': time.perf_counter() - started,
        'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
        'limitations': [
            'Eight fixed development conversations, not an independent holdout.',
            'Natural mode uses the model-card sampling settings and a 192-token cap.',
            'Closed mode is retained only to separate language loss from reasoning-format loss.',
            'No memory state is supplied; this compares language preservation only.',
        ],
    })
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.candidate_dir.is_dir():
        parser.error('Candidate directory must exist')
    main(args.candidate_dir, args.output)
