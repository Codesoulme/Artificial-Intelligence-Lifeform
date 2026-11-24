"""Probe a function-preserving low-rank replacement of one Qwen FFN projection.

The source projection is initialized by fitting a low-rank map on real hidden
activations, then the factors are fit with frozen full-model teacher logits.
This is an isolated architecture probe: it never writes the source checkpoint
or the durable individual.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "runs" / "open-base-deps")]

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer

from audit_qwen_compatible_raw_control_20261121 import evaluate
from deepseek_output_aligned_memory import MODEL, sha, write
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_checkpoint import load_integrated
from spora.language_training import frozen_digest
from spora.pretrained_attention import SPORAResidualBlock
from train_awake_dialogue_recovery import dialogue_cases, memory_logits
from train_awake_feedforward_replay import paired_targets
from train_awake_native_outputs import cache_language_targets
from train_awake_replacement_replay import answer_logits, divergence


SOURCE = ROOT / "runs" / "qwen-compatible-raw-v1" / "integrated.pt"
SEED = 20261123


@torch.no_grad()
def collect_projection_samples(decoder, tokenizer, layer_index, limit=4096):
    mlp = selected_mlp(decoder, layer_index)
    original = mlp.down
    captured = []

    def before(_module, args):
        if args:
            captured.append(args[0].detach().float().cpu().reshape(-1, args[0].shape[-1]))

    hook = original.register_forward_pre_hook(before)
    try:
        for _, prompt, target in dialogue_cases(tokenizer):
            ids = torch.tensor([prompt + target[:-1]], device="cuda")
            decoder(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                    logits_to_keep=1)
            if sum(row.shape[0] for row in captured) >= limit:
                break
    finally:
        hook.remove()
    if not captured:
        raise ValueError("No projection activations were captured")
    hidden = torch.cat(captured, dim=0)[:limit]
    target = F.linear(hidden, original.weight.detach().float().cpu())
    return hidden, target


def fit_factorization(hidden, target, rank, ridge=1e-3):
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    right = torch.randn(rank, hidden.shape[1], generator=generator) / hidden.shape[1] ** .5
    projected = hidden @ right.t()
    gram = projected.t() @ projected
    gram.diagonal().add_(ridge * gram.diagonal().mean().clamp_min(1e-6))
    left_t = torch.linalg.solve(gram, projected.t() @ target)
    return left_t.t(), right


@torch.no_grad()
def fit_output_subspace(target, weight, rank):
    width = min(rank + 24, *target.shape)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        _, _, directions = torch.pca_lowrank(target, q=width, center=False, niter=4)
    left = directions[:, :rank].contiguous()
    return left, left.t() @ weight


class FactorizedLinear(nn.Module):
    """A dense linear map represented as two trainable low-rank maps."""

    def __init__(self, source, rank, factors):
        super().__init__()
        if source.bias is not None:
            raise ValueError("The probe expects a bias-free projection")
        if type(rank) is not int or not 1 <= rank <= min(source.out_features, source.in_features):
            raise ValueError("Invalid factorization rank")
        left, right = factors
        if left.shape != (source.out_features, rank) or right.shape != (rank, source.in_features):
            raise ValueError("Factorized projection shape does not match the loaded model")
        device = source.weight.device
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.rank = rank
        self.left = nn.Parameter(left.to(device=device, dtype=torch.float32))
        self.right = nn.Parameter(right.to(device=device, dtype=torch.float32))

    def forward(self, hidden):
        return F.linear(F.linear(hidden.float(), self.right), self.left).to(hidden.dtype)

    def relative_weight_error(self, source):
        with torch.no_grad():
            reconstructed = self.left.float() @ self.right.float()
            original = source.weight.detach().float()
            return float((reconstructed - original).norm() / original.norm().clamp_min(1e-8))


def selected_mlp(decoder, index):
    layer = decoder.base.model.layers[index]
    if hasattr(layer, "layer") and layer.__class__.__name__ == "_MemoryDecoderLayer":
        layer = layer.layer
    if isinstance(layer, SPORAResidualBlock):
        layer = layer.base
    mlp = getattr(layer, "mlp", None)
    if mlp is None or not hasattr(mlp, "down"):
        raise ValueError("The selected layer has no compatible FFN output projection")
    return mlp


def set_factor(decoder, index, factor):
    selected_mlp(decoder, index).down = factor


@torch.no_grad()
def teacher_targets(decoder, tokenizer, organism):
    language = cache_language_targets(decoder, dialogue_cases(tokenizer)[::2])
    memory = paired_targets(decoder, tokenizer, organism)
    return language, memory


def train(decoder, tokenizer, organism, language, memory, steps, learning_rate):
    parameters = [p for p in decoder.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("The factorized projection is not trainable")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.)
    rng = random.Random(SEED)
    losses = []
    for step in range(steps):
        if memory and step % 4 == 0:
            prompt, target, controls, reference = memory[(step // 4) % len(memory)]
            logits = memory_logits(decoder, prompt, target, controls)
            source = "memory"
        else:
            prompt, target, reference = language[rng.randrange(len(language))]
            logits = answer_logits(decoder, prompt, target)
            source = "language"
        reference = reference.to(device=logits.device, dtype=logits.dtype)
        kl = divergence(logits, reference)
        nll = F.cross_entropy(logits, torch.tensor(target, device=logits.device))
        loss = kl + 0.10 * nll
        if source == "memory":
            loss = 1.5 * loss
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite factor training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append({"source": source, "kl": float(kl.detach()),
                       "nll": float(nll.detach()), "total": float(loss.detach())})
        if (step + 1) % 32 == 0:
            window = losses[-32:]
            print("Factor step", step + 1,
                  {key: round(sum(row[key] for row in window) / len(window), 6)
                   for key in ("kl", "nll", "total")}, flush=True)
    return losses


def main(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    source_checkpoint = args.source.resolve()
    protected = {source_checkpoint: sha(source_checkpoint), MODEL / "model.safetensors": sha(MODEL / "model.safetensors"),
                 MODEL / "tokenizer.json": sha(MODEL / "tokenizer.json")}
    started = time.perf_counter()
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed()
    decoder, organism, state = load_integrated(source_checkpoint, base, MODEL, individual_device="cpu",
                                             disposable_base=True)
    decoder.eval().requires_grad_(False)
    individual_digest = frozen_digest(organism, state)
    position_limit = min(getattr(layer, "max_length", base.config.max_position_embeddings)
                         for layer in decoder.base.model.layers)
    baseline = evaluate(decoder, tokenizer, args.ceval, position_limit, organism)
    mlp = selected_mlp(decoder, args.layer)
    original_down = mlp.down
    print("Collecting activation-aware projection samples", flush=True)
    hidden, target = collect_projection_samples(decoder, tokenizer, args.layer)
    split = 4 * len(hidden) // 5
    if min(split, target.shape[1]) < args.rank or split == len(hidden):
        raise ValueError("Insufficient calibration samples for the requested rank")
    print("Fitting activation-aware low-rank projection", flush=True)
    if args.initializer == 'output-subspace':
        factors = fit_output_subspace(target[:split], original_down.weight.detach().float().cpu(), args.rank)
    else:
        factors = fit_factorization(hidden[:split], target[:split], args.rank)
    predictions = F.linear(F.linear(hidden[split:], factors[1]), factors[0])
    held_error = float((predictions - target[split:]).norm() / target[split:].norm().clamp_min(1e-8))
    del hidden, target, predictions
    factor = FactorizedLinear(original_down, args.rank, factors=factors).to(original_down.weight.device)
    relative_error = factor.relative_weight_error(original_down)
    set_factor(decoder, args.layer, factor)
    initial = evaluate(decoder, tokenizer, args.ceval, position_limit, organism)
    losses, language, memory = [], (), ()
    trained = initial
    if args.steps:
        set_factor(decoder, args.layer, original_down)
        language, memory = teacher_targets(decoder, tokenizer, organism)
        set_factor(decoder, args.layer, factor)
        decoder.requires_grad_(False)
        factor.requires_grad_(True)
        torch.cuda.empty_cache()
        losses = train(decoder, tokenizer, organism, language, memory, args.steps, args.learning_rate)
        decoder.eval().requires_grad_(False)
        trained = evaluate(decoder, tokenizer, args.ceval, position_limit, organism)
    candidate = output / "factorized-down.pt"
    torch.save({"format": "spora-factorized-qwen-ffn-down-v1", "layer": args.layer,
                "rank": args.rank, "relative_weight_error": relative_error,
                "source_sha256": protected[source_checkpoint],
                "state": {key: value.detach().cpu() for key, value in factor.state_dict().items()}}, candidate)
    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError("A protected source changed during the probe")
    if frozen_digest(organism, state) != individual_digest:
        raise ValueError("The neural individual changed during the probe")
    write(output / "result.json", {
        "format": "spora-factorized-qwen-ffn-down-diagnostic-v1",
        "source_checkpoint": str(source_checkpoint), "layer": args.layer, "rank": args.rank,
        "initializer": args.initializer, "held_activation_relative_error": held_error,
        "position_limit": position_limit, "relative_weight_error": relative_error,
        "baseline": baseline, "initial_factorized": initial, "trained": trained,
        "losses": losses, "language_sequences": len(language), "memory_sequences": len(memory),
        "candidate_sha256": sha(candidate), "sources": {str(path): digest for path, digest in protected.items()},
        "individual_sha256": individual_digest, "seconds": time.perf_counter() - started,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
    })
    print("Baseline", baseline["summary"], flush=True)
    print("Initial", initial["summary"], flush=True)
    print("Trained", trained["summary"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ceval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--initializer", choices=('random-ridge', 'output-subspace'), default='random-ridge')
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    args = parser.parse_args()
    if args.layer < 0 or args.rank < 1 or args.steps < 0 or args.learning_rate <= 0:
        parser.error("Layer and steps must be nonnegative, with positive rank and learning rate")
    if not args.ceval.is_dir():
        parser.error("C-Eval directory does not exist")
    main(args)
