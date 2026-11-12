"""Audit shared-codebook distinguishability, noise and unknown false matches.

These are direct nearest-neighbor measurements, not decoder generation.
"""
import json
import time

from scripts import deepseek_spora_long_memory_generate_parity as p
from safetensors import safe_open

OUT = p.ROOT / 'runs/continuous-codebook-noise-audit-20261112.json'
PREV = p.ROOT / 'runs/continuous-embedding-bridge-pilot-20261112/result.json'
SEEDS = (761, 762)
NOISES = (0.0, 0.05, 0.1, 0.2, 0.4)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch = p.torch
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Base differs from expected')
    previous = json.loads(PREV.read_text(encoding='utf-8'))
    if previous['base_sha256'] != base_sha:
        raise ValueError('Token split belongs to another base')
    ids = previous['train_ids'] + previous['held_ids']
    if len(ids) != 320 or len(set(ids)) != 320:
        raise ValueError('Unexpected token split')
    with safe_open(p.MODEL / 'model.safetensors', framework='pt', device='cpu') as f:
        embedding = f.get_tensor('model.embed_tokens.weight').float().cuda()
    held = torch.tensor(ids[256:], device='cuda')
    candidate_ids = torch.tensor(ids, device='cuda')
    rows = []
    for seed in SEEDS:
        for width in (22, 90):
            gen = torch.Generator(device='cpu').manual_seed(seed)
            projection = torch.randn((embedding.shape[1], width), generator=gen).cuda()
            with torch.no_grad():
                codebook = torch.nn.functional.normalize(embedding @ projection, dim=-1)
                if not bool(torch.isfinite(codebook).all()):
                    raise ValueError('Nonfinite codebook')
                queries = codebook.index_select(0, held)
                if bool((queries.norm(dim=-1) < .99).any()):
                    raise ValueError('Zero selected code')
                # Draw the same normalized noise per width and seed for every magnitude.
                noise_gen = torch.Generator(device='cpu').manual_seed(seed + 10000)
                noise = torch.randn(queries.shape, generator=noise_gen).cuda() / width**0.5
                random_queries = torch.nn.functional.normalize(
                    torch.randn(queries.shape, generator=noise_gen).cuda(), dim=-1)
                for sigma in NOISES:
                    q = torch.nn.functional.normalize(queries + sigma * noise, dim=-1)
                    for candidates in ('split320', 'full_vocab'):
                        cb = codebook.index_select(0, candidate_ids) if candidates == 'split320' else codebook
                        target = torch.arange(256, 320, device='cuda') if candidates == 'split320' else held
                        scores = q @ cb.T
                        target_scores = scores.gather(1, target[:, None]).squeeze(1)
                        higher = (scores > target_scores[:, None]).sum(-1)
                        ties = (scores == target_scores[:, None]).sum(-1)
                        other = scores.clone()
                        other.scatter_(1, target[:, None], float('-inf'))
                        margins = target_scores - other.max(dim=-1).values
                        row = {'seed': seed, 'width': width, 'noise_rms': sigma, 'candidates': candidates,
                               'correct_unique': int(((higher == 0) & (ties == 1)).sum()), 'count': len(held),
                               'ties': int((ties > 1).sum()), 'min_margin': float(margins.min()),
                               'mean_margin': float(margins.mean()), 'mean_target_score': float(target_scores.mean())}
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                random_scores = random_queries @ codebook.T
                maxima = random_scores.max(dim=-1).values
                rows.append({'seed': seed, 'width': width, 'condition': 'random_unknown_full_vocab',
                             'count': len(held), 'top_score_min': float(maxima.min()),
                             'top_score_mean': float(maxima.mean()), 'top_score_max': float(maxima.max())})
            del projection, codebook, queries, noise, random_queries, random_scores
    if p.sha(p.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Base changed')
    report = {'rows': rows, 'base_sha256': base_sha, 'held_ids': previous['held_ids'],
              'noise_definition': 'isotropic Gaussian, expected L2 norm sigma; same noise scaled across levels',
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Nearest-neighbor access to a shared codebook is not language-model generation.',
                              'Codes and candidates share a projection: exact self-matches are expected.',
                              'Gaussian noise is synthetic and does not represent actual memory interference.',
                              '64 held vocabulary tokens, 2 seeds; unknowns are random unit vectors, not real unknown memories.',
                              'No stored memory, routing, state, or model changed.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'seconds': report['seconds'], 'peak': report['peak_allocated_bytes']}), flush=True)


if __name__ == '__main__':
    main()
