"""Offline geometry probe; no model inference or state writes."""
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

OUT = ROOT / 'runs/random-projection-geometry-v2-20261117.json'
SEED = 21741


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    path = MODEL / 'model.safetensors'
    if sha(path) != WEIGHT_SHA:
        raise ValueError('Base weight hash mismatch')
    torch.set_num_threads(2)
    with safe_open(str(path), framework='pt', device='cpu') as stream:
        embedding = stream.get_tensor('model.embed_tokens.weight').float()
    generator = torch.Generator().manual_seed(SEED)
    ids = torch.randperm(embedding.shape[0], generator=generator)[:4608]
    x = embedding.index_select(0, ids)
    del embedding
    x = F.normalize(x, dim=-1)
    train, held = x[:4096], x[4096:]
    mean_vector = F.normalize(train.mean(dim=0), dim=0)
    baseline_cosine = held @ mean_vector
    rows = []
    for width in (9, 18, 64):
        # Different widths are independently drawn; this compares capacity,
        # not exact nested projections or the structured codec's seeded matrix.
        projection = torch.randn((x.shape[1], width), generator=generator)
        codes = F.normalize(x @ projection, dim=-1)
        train_code, test_code = codes[:4096], codes[4096:]
        # Ridge fit maps normalized projected codes to normalized source vectors.
        ridge = 0.01
        mapping = torch.linalg.solve(train_code.T @ train_code + ridge * torch.eye(width),
                                    train_code.T @ train)
        reconstructed = F.normalize(test_code @ mapping, dim=-1)
        similarities = reconstructed @ held.T
        ranks = (similarities > similarities.diagonal().unsqueeze(1)).sum(dim=1) + 1
        cosine = (reconstructed * held).sum(-1)
        rows.append({'width': width, 'mean_cosine': float(cosine.mean()),
                     'median_cosine': float(cosine.median()),
                     'top1_within_512': int((ranks == 1).sum()),
                     'top10_within_512': int((ranks <= 10).sum()),
                     'mean_rank_within_512': float(ranks.float().mean())})
    result = {'seed': SEED, 'train_types': 4096, 'held_types': 512,
              'sample_ids_sha256': hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
              'base_sha256': WEIGHT_SHA, 'ridge': 0.01, 'rows': rows,
              'train_mean_baseline_cosine': float(baseline_cosine.mean()),
              'limitations': ['Random vocabulary rows; not a Chinese or dialogue benchmark.',
                              '512-way held-only retrieval is not full-vocabulary retrieval.',
                              'Linear reconstruction is not an optimal nonlinear decoder.',
                              'Does not measure native matrix noise or free generation.'],
              'seconds': time.perf_counter() - started}
    if sha(path) != WEIGHT_SHA:
        raise ValueError('Base weight changed')
    with OUT.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
