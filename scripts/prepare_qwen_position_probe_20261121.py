"""Write a position-table calibration probe on an isolated checkpoint."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch

from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_checkpoint import load_integrated_runtime


@torch.no_grad()
def prepare(source, output):
    if output.exists():
        raise FileExistsError(output)
    digest = sha(source)
    runtime = load_integrated_runtime(source, load_full_streamed(), MODEL)
    table = runtime.decoder.base.model.rotary_emb
    weight = table.coefficients.weight
    if table.learned_coefficients() is not None:
        raise ValueError('The probe requires an unmodified position table')
    # A small change in one rotary frequency makes position parameters
    # observably different while retaining their paired sine/cosine scale.
    cosine, sine = weight.chunk(2, dim=-1)
    angle = torch.arange(len(weight), device=weight.device, dtype=weight.dtype) * 1e-5
    real, imaginary = cosine[:, 0].clone(), sine[:, 0].clone()
    cosine[:, 0].copy_(real * angle.cos() - imaginary * angle.sin())
    sine[:, 0].copy_(imaginary * angle.cos() + real * angle.sin())
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime.save(output)
    payload = torch.load(output, map_location='cpu', weights_only=True)
    if not torch.equal(payload['tensors']['position/coefficients'], weight.cpu()) or sha(source) != digest:
        raise ValueError('Position weights were lost or the source checkpoint changed')
    print({'source_sha256': digest, 'candidate_sha256': sha(output), 'positions': len(weight)}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
