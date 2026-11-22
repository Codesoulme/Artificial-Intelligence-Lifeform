"""Create an isolated SPORA checkpoint with a larger anchored-attention layout."""

import argparse
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "runs" / "open-base-deps")]

from spora.integrated_checkpoint import _digest, load_integrated_runtime
from scripts.deepseek_local_baseline import MODEL
from scripts.deepseek_spora_full_stream_loader import load_full_streamed


def main(parent: Path, output: Path, layers: tuple[int, ...]) -> None:
    if output.exists():
        raise FileExistsError(output)
    if layers != tuple(sorted(set(layers))) or not layers:
        raise ValueError("Layers must be a nonempty sorted set")

    payload = torch.load(parent, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"metadata", "tensors", "integrity_sha256"}:
        raise ValueError("Invalid integrated checkpoint")
    metadata = dict(payload["metadata"])
    tensors = payload["tensors"]
    previous = metadata.get("anchored_attention")
    if not isinstance(previous, dict):
        raise ValueError("Parent checkpoint has no anchored-attention layout")
    if set(previous) != {"layers", "window", "prefix"}:
        raise ValueError("Unexpected parent layout")
    if any(type(value) is not int for value in (previous["window"], previous["prefix"])):
        raise ValueError("Parent layout has invalid dimensions")
    compatible = set(metadata.get("compatible_layers", []))
    memory_layers = set(metadata["reader"]["layers"])
    if not set(layers) <= compatible or set(layers) & memory_layers:
        raise ValueError("Anchored layers must be compatible and must not be memory sites")
    layout = {"layers": list(layers), "window": previous["window"], "prefix": previous["prefix"]}

    metadata_without_layout = dict(metadata)
    del metadata_without_layout["anchored_attention"]
    temporary = {"metadata": metadata_without_layout, "tensors": tensors}
    temporary["integrity_sha256"] = _digest(metadata_without_layout, tensors)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spora-layout-") as folder:
        source = Path(folder) / "unanchored.pt"
        torch.save(temporary, source)
        runtime = load_integrated_runtime(source, load_full_streamed(), MODEL)
        runtime.decoder.configure_anchored_attention(layout)
        runtime.save(output)

    packed = torch.load(output, map_location="cpu", weights_only=True)
    packed_metadata = dict(packed["metadata"])
    if packed_metadata.pop("anchored_attention", None) != layout:
        raise ValueError("Saved layout does not match the requested layout")
    parent_metadata = dict(metadata)
    parent_metadata.pop("anchored_attention", None)
    if packed_metadata != parent_metadata or packed["tensors"].keys() != tensors.keys():
        raise ValueError("Layout packing changed checkpoint metadata or tensor keys")
    if any(not torch.equal(packed["tensors"][key], tensors[key]) for key in tensors):
        raise ValueError("Layout packing changed a tensor")
    print({"parent": str(parent), "output": str(output), "layers": list(layers)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    args = parser.parse_args()
    main(args.parent, args.output, tuple(args.layers))
