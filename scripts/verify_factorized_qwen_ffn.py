"""Reload and audit an isolated factorized Qwen FFN projection candidate."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "runs" / "open-base-deps")]

import torch
from transformers import AutoTokenizer

from audit_qwen_compatible_raw_control_20261121 import evaluate
from deepseek_output_aligned_memory import MODEL, sha, write
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_checkpoint import load_integrated
from spora.language_training import frozen_digest
from research_factorized_qwen_ffn import FactorizedLinear, selected_mlp, set_factor


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    source_digest = sha(args.source)
    candidate_digest = sha(args.candidate)
    payload = torch.load(args.candidate, map_location="cpu", weights_only=True)
    if payload.get("format") != "spora-factorized-qwen-ffn-down-v1":
        raise ValueError("Unexpected factorized candidate format")
    if payload.get("layer") != args.layer or payload.get("rank") != args.rank:
        raise ValueError("Candidate layer or rank does not match the request")
    if payload.get("source_sha256", source_digest) != source_digest:
        raise ValueError("Candidate belongs to a different source checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed()
    decoder, organism, state = load_integrated(args.source, base, MODEL, individual_device="cpu")
    decoder.eval().requires_grad_(False)
    individual_digest = frozen_digest(organism, state)
    position_limit = min(getattr(layer, "max_length", base.config.max_position_embeddings)
                         for layer in decoder.base.model.layers)
    baseline = evaluate(decoder, tokenizer, args.ceval, position_limit, organism)
    source = selected_mlp(decoder, args.layer).down
    factor = FactorizedLinear(source, args.rank,
                              factors=(payload["state"]["left"], payload["state"]["right"])).to(
                                  source.weight.device)
    set_factor(decoder, args.layer, factor)
    restored = evaluate(decoder, tokenizer, args.ceval, position_limit, organism)
    if (frozen_digest(organism, state) != individual_digest or sha(args.source) != source_digest
            or sha(args.candidate) != candidate_digest):
        raise ValueError("Reload audit changed a protected input")
    result = {"format": "spora-factorized-qwen-ffn-down-reload-v1", "source_checkpoint": str(args.source),
              "candidate": str(args.candidate), "layer": args.layer, "rank": args.rank,
              "baseline": baseline, "restored": restored, "candidate_sha256": candidate_digest,
              "source_sha256": source_digest, "individual_sha256": individual_digest}
    write(args.output, result)
    print("Baseline", baseline["summary"], flush=True)
    print("Restored", restored["summary"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--ceval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    main(parser.parse_args())
