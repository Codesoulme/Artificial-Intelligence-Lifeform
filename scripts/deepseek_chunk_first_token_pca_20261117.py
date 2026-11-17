"""Isolated PCA-code continuation of the matched first-token experiment.

The PCA basis uses only the exact training token types. The base model and
native individual remain frozen, and model outputs never use a codebook lookup.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch.nn import functional as F
from scripts import deepseek_chunk_first_token_bridge_20261117 as pilot

pilot.OUT = ROOT / 'runs/deepseek-spora-chunk-first-token-pca-4000-20261117'
pilot.STEPS = 4000

_split = pilot.split_token_types
_save = pilot.atomic_save
_codec = pilot.StructuredTextMemoryCodec
_train_ids = None
_basis = None


def split_token_types(*args, **kwargs):
    global _train_ids
    train, held = _split(*args, **kwargs)
    if set(train) & set(held):
        raise ValueError('Training and held types overlap')
    _train_ids = list(train)
    return train, held


class TrainingPcaCodec(_codec):
    def __init__(self, embedding, **kwargs):
        global _basis
        if _train_ids is None:
            raise ValueError('Training split must exist before codec initialization')
        super().__init__(embedding, **kwargs)
        # Use the same unnormalized embeddings as the existing codec, but fit
        # and center using training rows only. Preserve all global RNG state.
        training = embedding.index_select(0, torch.tensor(_train_ids)).float()
        center = training.mean(0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(pilot.SEED + 19)
            _, _, vectors = torch.pca_lowrank(training - center, q=32, center=False, niter=4)
        projection = vectors[:, :self.fact_slot_width].contiguous()
        codes = F.normalize((embedding.float() - center) @ projection, dim=-1)
        if not bool(torch.isfinite(codes).all()):
            raise ValueError('PCA codes contain nonfinite values')
        self.fact_projection.copy_(projection)
        self.fact_codebook.copy_(codes)
        _basis = {'projection': projection.clone(), 'center': center.clone()}


def save_with_basis(checkpoint, path):
    if _basis is None:
        raise ValueError('Missing learned encoder basis')
    checkpoint['research_encoder'] = {'kind': 'train_only_pca_centered_v1', **_basis}
    _save(checkpoint, path)


pilot.split_token_types = split_token_types
pilot.StructuredTextMemoryCodec = TrainingPcaCodec
pilot.atomic_save = save_with_basis

if __name__ == '__main__':
    pilot.main()
