"""Measure how much of the native memory read can move into a Qwen FFN."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from audit_awake_feedforward_memory import ORDINARY
from audit_awake_multiturn_layers import encode
from audit_qwen_compatible_raw_control_20261121 import evaluate_memory
from deepseek_output_aligned_memory import MODEL, sha, torch, write
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_checkpoint import load_integrated
from transformers import AutoTokenizer


@torch.inference_mode()
def ordinary_logits(decoder, tokenizer):
    device = next(decoder.parameters()).device
    result = []
    for question in ORDINARY:
        ids = encode(tokenizer, [{'role': 'user', 'content': question}])
        inputs = torch.tensor([ids], device=device)
        result.append(decoder(input_ids=inputs, use_cache=False, logits_to_keep=1).logits.cpu())
    return result


def main(checkpoint, output, layer, *, isolate=False):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    protected = {path: sha(path) for path in (checkpoint, MODEL / 'model.safetensors', MODEL / 'tokenizer.json')}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    decoder, organism, _ = load_integrated(checkpoint, load_full_streamed(), MODEL)
    decoder.eval().requires_grad_(False)
    ordinary = ordinary_logits(decoder, tokenizer)
    results = {}

    def score(label):
        observed = ordinary_logits(decoder, tokenizer)
        identical = all(torch.equal(a, b) for a, b in zip(ordinary, observed, strict=True))
        memory, summary = evaluate_memory(decoder, tokenizer, organism, paired_history=True)
        results[label] = {**memory, 'summary': {**summary, 'ordinary_logits_identical': identical}}
        print(label, results[label]['summary'], flush=True)

    if isolate:
        decoder.use_feedforward_memory(layer, fraction=.5)
        wrapper = decoder.base.model.layers[layer]
        feedforward = wrapper.layer.mlp
        read = feedforward.read_memory
        try:
            feedforward.read_memory = None
            score('output_half_only')

            def read_half_at_input(hidden):
                return hidden + .5 * (wrapper.read_memory(hidden) - hidden)

            decoder.use_feedforward_memory(layer, fraction=0.)
            feedforward.read_memory = read_half_at_input
            score('output_full_plus_input_half')
        finally:
            feedforward.read_memory = read
    else:
        for fraction in (0., .15, .5, 1.):
            decoder.use_feedforward_memory(layer, fraction=fraction)
            score(str(fraction))
    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError('A source file changed during evaluation')
    write(output, {'format': 'spora-qwen-feedforward-memory-fraction-v1', 'layer': layer,
                   'sources': {str(path): digest for path, digest in protected.items()},
                   'results': results, 'peak_gpu_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layer', type=int, default=5)
    parser.add_argument('--isolate', action='store_true')
    args = parser.parse_args()
    main(args.checkpoint, args.output, args.layer, isolate=args.isolate)
