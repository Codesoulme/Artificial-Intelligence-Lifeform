"""Measure the tradeoff between a learned residual and the original backbone."""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit_awake_reasoning_residual_compare_20261120 import load_candidate
from deepseek_output_aligned_memory import MODEL, SOURCE, sha, torch, write
from scripts import train_awake_closed_reviewed_dialogue_residual_20261121 as current
from audit_awake_multiturn_layers import ceval_questions


core = current.parent
LAYERS = core.LAYERS


def main(candidate_dir, ceval, output, scales):
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    candidate_dir = candidate_dir.resolve()
    decoder, tokenizer, originals, _, organism, _ = core.load_layer_pair()
    from train_awake_replacement_replay import switch
    switch(decoder, originals)
    questions = list(ceval_questions(ceval, tokenizer))
    # Importing through the existing audit keeps the memory construction identical.
    from audit_awake_feedforward_memory import history_cases, HISTORY
    _, held_memory = history_cases(organism)
    changed = [(key, name, (earlier + 2) % 8, (current + 2) % 8)
               for key, name, earlier, current in HISTORY]
    _, alternate_memory = history_cases(organism, changed)
    adapters = core.attach_adapters(decoder)
    load_candidate(candidate_dir, adapters)
    rows = []
    for scale in scales:
        for group in adapters.values():
            for module in group.values():
                module.alpha = module.rank * scale
        summary = core.evaluate_with_value_switches(
            decoder, tokenizer, held_memory, alternate_memory, questions,
            decoder.base.config.max_position_embeddings)['summary']
        rows.append({'scale': scale, 'summary': summary})
        print(scale, summary, flush=True)
    write(output, {
        'format': 'spora-closed-residual-scale-diagnostic-v1',
        'candidate_dir': str(candidate_dir), 'scales': scales, 'rows': rows,
        'sources': {
            str(SOURCE): sha(SOURCE), str(MODEL / 'model.safetensors'): sha(MODEL / 'model.safetensors'),
            str(candidate_dir / 'result.json'): sha(candidate_dir / 'result.json'),
            **{str(candidate_dir / f'layer-{index}-residuals.pt'): sha(
                candidate_dir / f'layer-{index}-residuals.pt') for index in LAYERS},
        },
        'seconds': time.perf_counter() - started,
        'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-dir', type=Path, required=True)
    parser.add_argument('--ceval', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scales', type=float, nargs='+', default=[.5, .75, 1.0])
    args = parser.parse_args()
    if not args.candidate_dir.is_dir() or not args.ceval.is_dir():
        parser.error('Expected existing candidate and C-Eval directories')
    main(args.candidate_dir, args.ceval, args.output, args.scales)
