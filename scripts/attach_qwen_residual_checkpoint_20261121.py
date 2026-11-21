"""Attach a saved zero-init residual to an all-compatible checkpoint."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs/open-base-deps"))

from spora.integrated_checkpoint import _digest
from spora.language_training import atomic_save


def main(source, residual, output):
    if output.exists():
        raise FileExistsError(output)
    source_payload = torch.load(source, map_location="cpu", weights_only=True)
    candidate = torch.load(residual, map_location="cpu", weights_only=True)
    if not isinstance(source_payload, dict) or set(source_payload) != {"metadata", "tensors", "integrity_sha256"}:
        raise ValueError("Invalid integrated source")
    if _digest(source_payload["metadata"], source_payload["tensors"]) != source_payload["integrity_sha256"]:
        raise ValueError("Source integrity mismatch")
    if candidate.get("format") != "spora-qwen-compatible-residual-block-v1":
        raise ValueError("Invalid residual candidate format")
    metadata = dict(source_payload["metadata"])
    tensors = dict(source_payload["tensors"])
    if "residuals" in metadata or any(key.startswith("residual/") for key in tensors):
        raise ValueError("Source already contains residuals")
    index = candidate.get("layer")
    rank = candidate.get("rank")
    if (type(index) is not int or type(rank) is not int or rank < 1
            or index not in metadata.get("compatible_layers", [])):
        raise ValueError("Residual layer is not an all-compatible layer")
    state = candidate.get("state")
    if not isinstance(state, dict) or set(state) != {"down", "up"}:
        raise ValueError("Invalid residual state")
    width = state["down"].shape[1]
    if (state["down"].shape != (rank, width) or state["up"].shape != (width, rank)
            or any(not value.is_floating_point() or not bool(torch.isfinite(value).all())
                   for value in state.values())):
        raise ValueError("Invalid residual shapes")
    spec = {"index": index, "rank": rank}
    spec["sha256"] = _digest(spec, state)
    metadata["residuals"] = [spec]
    tensors.update({f"residual/{index}/{name}": value.detach().cpu().clone()
                    for name, value in state.items()})
    integrity = _digest(metadata, tensors)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save({"metadata": metadata, "tensors": tensors, "integrity_sha256": integrity}, output)
    print({"output": str(output), "layer": index, "rank": rank, "integrity": integrity})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--residual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.residual, args.output)
