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
from spora.integrated_checkpoint import load_integrated


def main(checkpoint, ceval, output, *, paired_history=False):
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
    decoder, organism, _ = load_integrated(checkpoint, base, MODEL, individual_device="cpu")
    decoder.eval().requires_grad_(False)
    limit = min(getattr(layer, "max_length", base.config.max_position_embeddings)
                for layer in decoder.base.model.layers)
    result = evaluate(decoder, tokenizer, ceval, limit, organism, paired_history=paired_history)
    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError("Protected source changed during reload audit")
    write(output, {
        "format": "spora-qwen-compatible-residual-reload-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": protected[checkpoint],
        "result": result,
        "sources": {str(path): digest for path, digest in protected.items()},
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
    })
    print(result["summary"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ceval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paired-history", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_file() or not args.ceval.is_dir():
        parser.error("Checkpoint or C-Eval directory does not exist")
    main(args.checkpoint, args.ceval, args.output, paired_history=args.paired_history)
