"""Create a raw-language, all-compatible SPORA checkpoint for isolation."""

import argparse
import json
from pathlib import Path

import torch

from spora.integrated_checkpoint import _digest
from spora.language_training import atomic_save


def make(source: Path, output: Path, layer_count: int):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"metadata", "tensors", "integrity_sha256"}:
        raise ValueError("Invalid integrated source checkpoint")
    metadata = dict(payload["metadata"])
    tensors = dict(payload["tensors"])
    if payload["integrity_sha256"] != _digest(metadata, tensors):
        raise ValueError("Source checkpoint integrity mismatch")
    replacements = metadata.get("replacements")
    if not isinstance(replacements, list) or {item.get("index") for item in replacements} != {6, 12}:
        raise ValueError("Expected the current two-layer replacement checkpoint")
    if "output_adapter" not in metadata:
        raise ValueError("Expected the memory output adapter to remain bound")
    replacement_keys = [key for key in tensors if key.startswith("replacement/")]
    for key in replacement_keys:
        del tensors[key]
    metadata.pop("replacements", None)
    metadata.pop("feature_variant_sha256", None)
    metadata["compatible_layers"] = list(range(layer_count))
    metadata["position_table_max_position"] = 768
    integrity = _digest(metadata, tensors)
    atomic_save({
        "metadata": metadata,
        "tensors": tensors,
        "integrity_sha256": integrity,
    }, output)
    print(json.dumps({
        "source": str(source),
        "output": str(output),
        "format": metadata["format"],
        "compatible_layers": metadata["compatible_layers"],
        "removed_replacement_tensors": len(replacement_keys),
        "integrity_sha256": integrity,
    }, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer-count", type=int, default=28)
    args = parser.parse_args()
    if args.layer_count < 1:
        parser.error("layer-count must be positive")
    make(args.source, args.output, args.layer_count)
