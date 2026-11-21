"""Fit the feed-forward memory residual to the native block output read."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from audit_qwen_compatible_raw_control_20261121 import evaluate_memory
from audit_qwen_feedforward_memory_fraction_20261121 import ordinary_logits
from deepseek_output_aligned_memory import MODEL, sha, torch, write
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_checkpoint import load_integrated, load_integrated_runtime
from train_awake_absolute_attention import replay_requests
from transformers import AutoTokenizer


@torch.inference_mode()
def fit(decoder, tokenizer, organism, wrapper):
    """Fit one bounded gain per channel on disjoint synthetic memory examples."""
    block = wrapper.layer
    input_delta = block_output = None
    width = block.hidden_size
    energy = torch.zeros(width, dtype=torch.float64)
    cross = torch.zeros_like(energy)
    target_energy = torch.zeros_like(energy)
    samples = 0

    def capture_input(_module, args):
        nonlocal input_delta
        read_input = args[0]
        input_delta = (wrapper.read_memory(read_input).float() - read_input.float()).cpu()

    def capture_block(_module, _args, output):
        nonlocal block_output
        block_output = output

    def capture_target(_module, _args, output):
        nonlocal samples
        target = (output.float() - block_output.float()).cpu()
        energy.add_(input_delta.square().sum(dim=(0, 1)))
        cross.add_((input_delta * target).sum(dim=(0, 1)))
        target_energy.add_(target.square().sum(dim=(0, 1)))
        samples += 1

    input_hook = block.mlp.register_forward_pre_hook(capture_input)
    block_hook = block.register_forward_hook(capture_block)
    target_hook = wrapper.register_forward_hook(capture_target)
    try:
        device = next(block.parameters()).device
        for offset in (0, 2):
            for prompt, controls in replay_requests(organism, tokenizer, value_offset=offset):
                ids = torch.tensor([prompt], device=device)
                decoder(input_ids=ids, logits_to_keep=1, **dict(controls, use_cache=False))
    finally:
        input_hook.remove()
        block_hook.remove()
        target_hook.remove()
    if not samples or target_energy.sum() <= 0:
        raise ValueError('No nonzero reference memory residual was measured')
    prior = (.01 * energy.mean()).clamp_min(1e-8)
    gain = ((cross + prior) / (energy + prior)).clamp(.25, 4.)

    def error(multiplier):
        return float(((multiplier ** 2 * energy - 2 * multiplier * cross + target_energy).sum()
                      / target_energy.sum()).clamp_min(0.))

    return gain.float().to(device), {'training_sequences': samples,
                                    'relative_error_unity': error(1.),
                                    'relative_error_calibrated': error(gain)}


def main(checkpoint, output, layer, *, candidate=None, all_memory_sites=False):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    protected = {path: sha(path) for path in (checkpoint, MODEL / 'model.safetensors', MODEL / 'tokenizer.json')}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed()
    if candidate is not None:
        if candidate.exists():
            raise FileExistsError(candidate)
        runtime = load_integrated_runtime(checkpoint, base, MODEL)
        decoder = runtime.decoder.eval().requires_grad_(False)
        layers = decoder.layer_indices if all_memory_sites else (layer,)
        for index in layers:
            decoder.use_feedforward_memory(index, at_output=True)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        runtime.save(candidate)
        if any(sha(path) != digest for path, digest in protected.items()):
            raise ValueError('A source file changed during checkpoint creation')
        write(output, {'format': 'spora-qwen-feedforward-memory-candidate-v1', 'layers': list(layers),
                       'checkpoint': str(candidate), 'checkpoint_sha256': sha(candidate),
                       'sources': {str(path): digest for path, digest in protected.items()}})
        print('Saved', candidate, flush=True)
        return

    decoder, organism, _ = load_integrated(checkpoint, base, MODEL)
    decoder.eval().requires_grad_(False)
    decoder.use_feedforward_memory(layer, fraction=0.)
    wrapper = decoder.base.model.layers[layer]
    mlp = wrapper.layer.mlp
    reference_logits = ordinary_logits(decoder, tokenizer)
    gain, fit_stats = fit(decoder, tokenizer, organism, wrapper)
    print(fit_stats, flush=True)

    results = {}

    def score(label):
        memory, summary = evaluate_memory(decoder, tokenizer, organism, paired_history=True)
        actual = ordinary_logits(decoder, tokenizer)
        summary['ordinary_logits_identical'] = all(torch.equal(a, b) for a, b in zip(reference_logits, actual, strict=True))
        results[label] = {**memory, 'summary': summary}
        print(label, summary, flush=True)

    score('native_read')
    decoder.use_feedforward_memory(layer, fraction=1.)
    saved_read = mlp.read_memory
    mlp.read_memory = None
    try:
        for name, multiplier in (('ffn_residual_unity', 1.), ('ffn_residual_calibrated', gain)):
            def inject(_module, args, output):
                hidden = args[0]
                delta = wrapper.read_memory(hidden).float() - hidden.float()
                return output + (delta * multiplier).to(output.dtype)

            hook = mlp.register_forward_hook(inject)
            try:
                score(name)
            finally:
                hook.remove()
    finally:
        mlp.read_memory = saved_read
        decoder.use_feedforward_memory(layer, fraction=0.)
    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError('A source file changed during evaluation')
    write(output, {'format': 'spora-qwen-feedforward-memory-residual-v1', 'layer': layer,
                   'fit': fit_stats, 'gains': gain.tolist(), 'results': results,
                   'sources': {str(path): digest for path, digest in protected.items()}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layer', type=int, default=5)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--all-memory-sites', action='store_true')
    args = parser.parse_args()
    if args.all_memory_sites and args.candidate is None:
        parser.error('--all-memory-sites requires --candidate')
    main(args.checkpoint, args.output, args.layer,
         candidate=args.candidate, all_memory_sites=args.all_memory_sites)
