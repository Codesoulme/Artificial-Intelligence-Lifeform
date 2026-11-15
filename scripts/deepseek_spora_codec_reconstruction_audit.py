"""Compare embedding reconstruction on a fixed vocabulary probe sample."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from safetensors import safe_open
from torch.nn import functional as F
from scripts.deepseek_local_baseline import MODEL, sha

SOURCES = ('deepseek-spora-joint-codec-bridge-continuation-20261115',
           'deepseek-spora-joint-codec-gradient-repair-20261115',
           'deepseek-spora-codec-reconstruction-20261115')
OUT = ROOT / 'runs/deepseek-spora-codec-reconstruction-audit-20261116.json'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.set_num_threads(2)
    with safe_open(str(MODEL / 'model.safetensors'), framework='pt', device='cpu') as f:
        source = f.get_slice('model.embed_tokens.weight')
        generator = torch.Generator().manual_seed(16071)
        ids = torch.randperm(source.get_shape()[0], generator=generator)[:1024].sort().values
        embedding = torch.stack([source[int(i)] for i in ids]).float()
    expected = F.normalize(embedding, dim=-1)
    rows = []
    for name in SOURCES:
        path = ROOT / 'runs' / name / 'candidate.pt'
        payload = torch.load(path, map_location='cpu', weights_only=True)
        codes = F.normalize(embedding @ payload['codec_state']['projection'].float(), dim=-1)
        pred = F.normalize(F.linear(codes, payload['bridge']['linear.weight'].float()), dim=-1)
        similarity = (pred * expected).sum(-1)
        rows.append({'candidate': name, 'sha256': sha(path), 'mean_cosine': float(similarity.mean()),
                     'median_cosine': float(similarity.median()), 'tokens': len(ids)})
    result = {'rows': rows, 'sample_seed': 16071,
              'note': 'Random probe is not proven disjoint from either pretraining or auxiliary samples.'}
    OUT.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
