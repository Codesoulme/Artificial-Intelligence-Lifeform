"""Compare equal-width random and train-only PCA projections, offline only."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch.nn import functional as F
from safetensors import safe_open
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha

OUT = ROOT / 'runs/learned-projection-geometry-v2-20261117.json'
SEED = 21741


def evaluate(train, held, projection):
    """Fit a ridge decoder on train only, with an unregularized intercept."""
    center = train.mean(0)
    train_code = F.normalize((train - center) @ projection, dim=-1)
    held_code = F.normalize((held - center) @ projection, dim=-1)
    code_mean = train_code.mean(0)
    a, b = train_code - code_mean, train - center
    width = projection.shape[1]
    mapping = torch.linalg.solve(a.T @ a + 0.01 * torch.eye(width), a.T @ b)
    reconstructed = F.normalize((held_code - code_mean) @ mapping + center, dim=-1)
    sim = reconstructed @ held.T
    target_scores = sim.diagonal().unsqueeze(1)
    rank = (sim > target_scores).sum(1) + 1
    ties = (sim == target_scores).sum(1)
    return {'mean_cosine': float((reconstructed * held).sum(-1).mean()),
            'top1_within_512': int((rank == 1).sum()),
            'strict_top1_within_512': int(((rank == 1) & (ties == 1)).sum()),
            'tied_target_rows': int((ties > 1).sum()),
            'top10_within_512': int((rank <= 10).sum()),
            'mean_rank_within_512': float(rank.float().mean())}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    start = time.perf_counter()
    path = MODEL / 'model.safetensors'
    if sha(path) != WEIGHT_SHA:
        raise ValueError('Base weight hash mismatch')
    torch.set_num_threads(2)
    with safe_open(str(path), framework='pt', device='cpu') as f:
        embedding = f.get_tensor('model.embed_tokens.weight').float()
    rng = torch.Generator().manual_seed(SEED)
    ids = torch.randperm(embedding.shape[0], generator=rng)[:4608]
    x = F.normalize(embedding.index_select(0, ids), dim=-1)
    del embedding
    train, held = x[:4096], x[4096:]
    # PCA fit only to training rows, randomized SVD with independent seed.
    torch.manual_seed(SEED + 19)
    _, _, vectors = torch.pca_lowrank(train - train.mean(0), q=72, center=False, niter=4)
    rows = []
    for width in (9, 18, 64):
        random_projection = torch.randn((x.shape[1], width), generator=rng)
        for name, projection in (('random', random_projection), ('pca', vectors[:, :width])):
            rows.append({'width': width, 'projection': name, **evaluate(train, held, projection)})
    if sha(path) != WEIGHT_SHA:
        raise ValueError('Base weight changed')
    result = {'seed': SEED, 'pca_seed': SEED + 19, 'train_types': 4096, 'held_types': 512,
              'sample_ids_sha256': hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
              'base_sha256': WEIGHT_SHA, 'rows': rows, 'seconds': time.perf_counter() - start,
              'limitations': ['PCA is optimized for embedding reconstruction, not semantic recall.',
                              'Random vocabulary rows; no Chinese dialogue, native memory, or generation.',
                              'Ranking is restricted to 512 known held-out embedding candidates.',
                              'Individual PCA model artifacts and mean are extra parameters.',
                              'One vocabulary split, random seed and approximate PCA.']}
    with OUT.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
