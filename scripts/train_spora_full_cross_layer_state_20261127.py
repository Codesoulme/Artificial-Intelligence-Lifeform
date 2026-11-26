"""Compare an identity-initialized full state map with the no-bridge control.

This is an isolated diagnostic.  It trains only the research bridge and never
writes the integrated model or personal memory.
"""

import argparse
import time
from pathlib import Path

from audit_awake_feedforward_memory import HISTORY, history_cases
from audit_awake_language_layers import load_layer_pair
from audit_awake_multiturn_layers import CEVAL_SUBJECTS, ceval_questions
from audit_awake_original_all_control_20261121 import evaluate
from deepseek_output_aligned_memory import MODEL, ROOT, SOURCE, sha, torch, write
from scripts.deepseek_spora_memory_answer_distill import _memory_args
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.language_training import atomic_save
from spora.research_cross_layer_state import FullCrossLayerStateBridge
from train_awake_cross_layer_state_bridge_20261121 import (
    LAYERS, freeze_for_bridge, train_bridge,
)
from train_awake_fused_state_reader_20261121 import build_alternate_history, training_history_pairs
from train_awake_feedforward_replay import paired_targets
from train_awake_replacement_replay import switch


def configure(decoder, originals):
    switch(decoder, originals)
    decoder.migrate_pretrained_blocks()
    for index in decoder.layer_indices:
        decoder.use_feedforward_memory(index, fraction=0.)
    bridge = FullCrossLayerStateBridge(decoder.base.config.hidden_size).to('cuda')
    decoder.attach_cross_layer_bridge(bridge)
    return bridge


def evaluate_variant(decoder, tokenizer, organism, ceval, bridge):
    decoder.attach_cross_layer_bridge(bridge)
    return evaluate(decoder, tokenizer, organism, ceval,
                    decoder.base.config.max_position_embeddings)


def main(output, ceval, steps, learning_rate, cross_entropy_weight, contrast_weight):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    script = Path(__file__).resolve()
    subjects = tuple(CEVAL_SUBJECTS)
    sources = (SOURCE, MODEL / 'model.safetensors', MODEL / 'tokenizer.json',
               MODEL / 'tokenizer_config.json', ROOT / 'data/chinese-foundations.json',
               script, Path(FullCrossLayerStateBridge.__module__.replace('.', '/') + '.py'),
               *(ceval / f'{subject}.parquet' for subject in subjects))
    sources = tuple(path if path.is_absolute() else ROOT / path for path in sources)
    protected = {path: sha(path) for path in sources}
    started = time.perf_counter()
    decoder, tokenizer, originals, _, organism, _ = load_layer_pair()
    bridge = configure(decoder, originals)
    if tuple(decoder.layer_indices) != LAYERS:
        raise ValueError(f'Unexpected memory layers: {decoder.layer_indices}')
    questions = list(ceval_questions(ceval, tokenizer))
    if not questions:
        raise ValueError('Expected nonempty evaluation questions')
    rows = paired_targets(decoder, tokenizer, organism,
                          history_pairs=training_history_pairs(organism, tokenizer))
    if not rows:
        raise ValueError('Expected paired memory rows')

    no_bridge = evaluate_variant(decoder, tokenizer, organism, ceval, None)
    identity = evaluate_variant(decoder, tokenizer, organism, ceval, bridge)
    losses = train_bridge(decoder, bridge, rows, steps, learning_rate,
                          cross_entropy_weight, contrast_weight)
    trained = evaluate_variant(decoder, tokenizer, organism, ceval, bridge)
    output.mkdir(parents=True, exist_ok=True)
    bridge_path = output / 'bridge.pt'
    atomic_save({'format': FullCrossLayerStateBridge.format, 'config': bridge.config_dict(),
                 'layers': list(decoder.layer_indices), 'source_sha256': protected[SOURCE],
                 'base_sha256': protected[MODEL / 'model.safetensors'],
                 'state': {name: value.detach().cpu() for name, value in bridge.state_dict().items()}},
                bridge_path)
    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError('A protected source changed during training')
    write(output / 'result.json', {
        'format': 'spora-full-cross-layer-state-diagnostic-v1',
        'config': bridge.config_dict(), 'layers': list(decoder.layer_indices),
        'steps': steps, 'learning_rate': learning_rate,
        'cross_entropy_weight': cross_entropy_weight,
        'contrast_weight': contrast_weight, 'memory_sequences': len(rows),
        'no_bridge': no_bridge, 'identity_before_training': identity,
        'trained': trained, 'losses': losses,
        'bridge_sha256': sha(bridge_path),
        'sources': {str(path): digest for path, digest in protected.items()},
        'seconds': time.perf_counter() - started,
        'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
    })
    print({'no_bridge': no_bridge['summary'], 'identity_before_training': identity['summary'],
           'trained': trained['summary']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ceval', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--cross-entropy-weight', type=float, default=.25)
    parser.add_argument('--contrast-weight', type=float, default=.5)
    args = parser.parse_args()
    if (args.output.exists() or args.steps < 1 or args.learning_rate <= 0
            or args.cross_entropy_weight < 0 or args.contrast_weight < 0):
        parser.error('Expected an unused output, positive steps/rate, and nonnegative weights')
    main(args.output, args.ceval, args.steps, args.learning_rate,
         args.cross_entropy_weight, args.contrast_weight)
