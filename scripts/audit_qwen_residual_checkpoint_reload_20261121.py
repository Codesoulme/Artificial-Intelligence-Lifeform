"""Verify a persisted Qwen-compatible SPORA residual checkpoint."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_qwen_compatible_raw_control_20261121 import evaluate
from deepseek_output_aligned_memory import MODEL, sha, torch, write
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_checkpoint import load_integrated, load_integrated_runtime


def main(checkpoint, ceval, output, *, paired_history=False, resave=None, residual_scales=None,
         candidate=None, candidate_scale=None, memory_only=False):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    protected = {
        checkpoint: sha(checkpoint),
        MODEL / "model.safetensors": sha(MODEL / "model.safetensors"),
        MODEL / "tokenizer.json": sha(MODEL / "tokenizer.json"),
    }
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed()
    resaved_hash = None
    runtime = None
    if resave is None and candidate is None:
        decoder, organism, _ = load_integrated(checkpoint, base, MODEL, individual_device="cpu")
    else:
        if resave is not None and resave.exists():
            raise FileExistsError(resave)
        runtime = load_integrated_runtime(checkpoint, base, MODEL)
        decoder, organism = runtime.decoder, runtime.organism
    if resave is not None:
        resave.parent.mkdir(parents=True, exist_ok=True)
        runtime.save(resave)
        original = torch.load(checkpoint, map_location="cpu", weights_only=True)
        saved = torch.load(resave, map_location="cpu", weights_only=True)
        if original["integrity_sha256"] != saved["integrity_sha256"]:
            raise ValueError("Runtime resave changed the checkpoint metadata or tensors")
        del original, saved
        resaved_hash = sha(resave)
    decoder.eval().requires_grad_(False)
    limit = min(getattr(layer, "max_length", base.config.max_position_embeddings)
                for layer in decoder.base.model.layers)
    if decoder.position_table_max_position is not None:
        limit = min(limit, decoder.position_table_max_position)
    if residual_scales is None:
        result = evaluate(decoder, tokenizer, ceval, limit, organism, paired_history=paired_history)
    else:
        from spora.pretrained_attention import SPORAResidualBlock

        blocks = [module for module in decoder.modules() if isinstance(module, SPORAResidualBlock)]
        if not blocks:
            raise ValueError("A residual scale sweep requires residual weights")
        if memory_only:
            for block in blocks:
                block.memory_context = decoder._memory_context
        original = [block.up.weight.detach().clone() for block in blocks]
        result = {}
        with torch.no_grad():
            for scale in residual_scales:
                for block, weights in zip(blocks, original, strict=True):
                    block.up.weight.copy_(weights * scale)
                result[str(scale)] = evaluate(decoder, tokenizer, ceval, limit, organism, paired_history=paired_history)
                print(scale, result[str(scale)]["summary"], flush=True)
            if candidate is not None:
                if candidate.exists():
                    raise FileExistsError(candidate)
                for block, weights in zip(blocks, original, strict=True):
                    block.up.weight.copy_(weights * candidate_scale)
                candidate.parent.mkdir(parents=True, exist_ok=True)
                runtime.save(candidate)
    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError("Protected source changed during reload audit")
    write(output, {
        "format": "spora-qwen-compatible-residual-reload-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": protected[checkpoint],
        "resaved_sha256": resaved_hash,
        "scaled_candidate_sha256": None if candidate is None else sha(candidate),
        "result": result,
        "sources": {str(path): digest for path, digest in protected.items()},
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
    })
    if residual_scales is None:
        print(result["summary"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ceval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paired-history", action="store_true")
    parser.add_argument("--resave", type=Path)
    parser.add_argument("--residual-scales", type=float, nargs="+")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--candidate-scale", type=float)
    parser.add_argument("--memory-only", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_file() or not args.ceval.is_dir():
        parser.error("Checkpoint or C-Eval directory does not exist")
    if args.residual_scales is not None and (args.resave or any(not 0 <= scale <= 1 for scale in args.residual_scales)):
        parser.error("A residual sweep requires scales in [0, 1] and cannot use --resave")
    if (args.candidate is None) != (args.candidate_scale is None) or (
        args.candidate is not None and args.candidate_scale not in (args.residual_scales or ())
    ):
        parser.error("Specify both candidate options and include its scale in --residual-scales")
    if args.memory_only and args.residual_scales is None:
        parser.error("--memory-only requires --residual-scales")
    main(args.checkpoint, args.ceval, args.output, paired_history=args.paired_history,
         resave=args.resave, residual_scales=args.residual_scales,
         candidate=args.candidate, candidate_scale=args.candidate_scale, memory_only=args.memory_only)
