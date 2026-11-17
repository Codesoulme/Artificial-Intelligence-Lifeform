"""Bounded synthetic length/numerics check, no language model involved."""
import io
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from spora.research_bound_feature_memory import BoundFeatureMemory

OUT = ROOT / 'runs/bound-feature-long-synthetic-20261117.json'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(44017)
    projection = torch.randn(512, 128, generator=generator)
    keys = torch.randn(16384, 128, generator=generator)
    values = torch.randn(16384, 16, generator=generator)
    queries = torch.randn(8, 128, generator=generator)
    memory = BoundFeatureMemory(projection, base_sha256='a' * 64, layer=5, kv_head=0)
    state = None
    rows = []
    for start in range(0, len(keys), 128):
        chunk = memory.write(keys[start:start+128], values[start:start+128])
        state = chunk if state is None else memory.merge(state, chunk)
        count = start + 128
        if count in (1024, 4096, 16384):
            whole = memory.write(keys[:count], values[:count])
            actual, reference = memory.read(queries, state), memory.read(queries, whole)
            exact = torch.softmax(queries @ keys[:count].T / (128 ** 0.5), -1) @ values[:count]
            baseline = values[:count].mean(0).expand_as(exact)
            buffer = io.BytesIO()
            torch.save(state, buffer)
            buffer.seek(0)
            restored = memory.read(queries, torch.load(buffer, weights_only=True))
            rows.append({'history_tokens': count,
                         'chunk_vs_whole_max_abs': float((actual-reference).abs().max()),
                         'restored_max_abs': float((actual-restored).abs().max()),
                         'approx_attention_mse': float(F.mse_loss(actual, exact)),
                         'mean_baseline_mse': float(F.mse_loss(baseline, exact))})
    wrong = BoundFeatureMemory(projection + 0.01, base_sha256='a' * 64, layer=5, kv_head=0)
    try:
        wrong.read(queries, state)
    except ValueError:
        rejected = True
    else:
        raise AssertionError('Mismatched projection was accepted')
    result = {'seed': 44017, 'chunk_size': 128, 'rows': rows,
              'wrong_projection_rejected': rejected, 'seconds': time.perf_counter()-started,
              'limitations': ['Random Gaussian tensors, not DeepSeek activations or factual recall.',
                              'One seed and projection; no crash recovery or long-duration drift.',
                              'In-memory serialization only; no filesystem transactions.',
                              'No concurrent writers, corrections, or memory identity authentication.']}
    with OUT.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
