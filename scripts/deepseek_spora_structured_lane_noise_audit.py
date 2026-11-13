"""Measure structured-lane full-vocabulary margins under synthetic perturbation."""
import json
import math
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch.nn import functional as F

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_prefix_features import read_bf16_tensor

PREVIOUS = ROOT / 'runs/continuous-embedding-bridge-pilot-20261112/result.json'
OUTPUT = ROOT / 'runs/deepseek-spora-structured-lane-noise-audit-20261113.json'
SEEDS = (29071, 29072)
WIDTHS = (9, 15, 22)
NOISES = (0.0, 0.01, 0.05, 0.1, 0.2, 0.4)


def projection_for(hidden, width, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    # StructuredTextMemoryCodec draws identity projection before fact projection.
    torch.randn((hidden, 15), generator=generator)
    return torch.randn((hidden, width), generator=generator)


@torch.no_grad()
def score(codebook, ids, noise, sigma, batch_size=32):
    counts = {'correct_unique': 0, 'ties': 0}
    all_margins = []
    for start in range(0, len(ids), batch_size):
        target = ids[start:start + batch_size]
        query = F.normalize(codebook.index_select(0, target) + sigma * noise[start:start + batch_size], dim=-1)
        scores = query @ codebook.T
        target_scores = scores.gather(1, target[:, None]).squeeze(1)
        higher = (scores > target_scores[:, None]).sum(-1)
        ties = (scores == target_scores[:, None]).sum(-1)
        scores.scatter_(1, target[:, None], float('-inf'))
        margins = target_scores - scores.max(-1).values
        counts['correct_unique'] += int(((higher == 0) & (ties == 1)).sum())
        counts['ties'] += int((ties > 1).sum())
        all_margins.append(margins.cpu())
    margins = torch.cat(all_margins)
    counts.update(count=len(ids), margin_min=float(margins.min()), margin_mean=float(margins.mean()),
                  margin_p10=float(torch.quantile(margins, .1)))
    return counts


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    path = MODEL / 'model.safetensors'
    if sha(path) != WEIGHT_SHA:
        raise ValueError('Base model digest mismatch')
    previous = json.loads(PREVIOUS.read_text(encoding='utf-8'))
    if previous['base_sha256'] != WEIGHT_SHA:
        raise ValueError('Token split and base differ')
    ids_list = previous['train_ids'] + previous['held_ids']
    if len(ids_list) != 320 or len(set(ids_list)) != 320:
        raise ValueError('Unexpected token sample')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with path.open('rb') as stream:
        header_size = struct.unpack('<Q', stream.read(8))[0]
        if header_size > 100_000_000 or header_size > path.stat().st_size - 8:
            raise ValueError('Invalid tensor header')
        header = json.loads(stream.read(header_size))
        embedding = read_bf16_tensor(stream, header, 8 + header_size, path.stat().st_size,
                                     'model.embed_tokens.weight').cuda().float()
    ids = torch.tensor(ids_list, dtype=torch.long, device='cuda')
    rows = []
    with torch.no_grad():
        for seed in SEEDS:
            for width in WIDTHS:
                projection = projection_for(embedding.shape[1], width, seed).cuda()
                codebook = F.normalize(embedding @ projection, dim=-1)
                if not bool(torch.isfinite(codebook).all()) or bool((codebook[ids].norm(dim=-1) < .99).any()):
                    raise ValueError('Invalid projected codes')
                generator = torch.Generator(device='cpu').manual_seed(seed + width + 10000)
                noise = torch.randn((len(ids), width), generator=generator).cuda() / math.sqrt(width)
                for sigma in NOISES:
                    row = {'seed': seed, 'width': width, 'noise_expected_norm': sigma,
                           **score(codebook, ids, noise, sigma)}
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                del codebook, projection, noise
    if sha(path) != WEIGHT_SHA:
        raise ValueError('Base model changed')
    report = {'status': 'direct_codebook_perturbation_probe', 'rows': rows, 'token_ids': ids_list,
              'base_sha256': WEIGHT_SHA, 'previous_split_sha256': sha(PREVIOUS),
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Direct projected-code lookup, not memory matrix reads or language generation.',
                              '320 previously selected stable CJK tokens, not an independent language benchmark.',
                              'Noise is synthetic isotropic Gaussian with expected L2 norm, not measured write interference.',
                              '9-dimensional seed 29071 is the current fact projection; other widths/seeds are alternatives.',
                              'Wider lanes would use more capacity and are not equivalent within a fixed 90-dimensional packet.',
                              'Clean self-match is expected because query and full vocabulary use the same codebook.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'seconds': report['seconds'], 'peak': report['peak_allocated_bytes']}), flush=True)


if __name__ == '__main__':
    main()
