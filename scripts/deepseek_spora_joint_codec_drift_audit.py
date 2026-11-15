"""Measure divergence between saved lookup codes and the learned projection."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from torch.nn import functional as F

from scripts.deepseek_local_baseline import MODEL, sha

SOURCES = (
    'deepseek-spora-joint-codec-bridge-continuation-20261115',
    'deepseek-spora-joint-relation-lane-reset-20261115',
    'deepseek-spora-joint-relation-lane-full-coverage-20261115',
)
OUT = ROOT / 'runs/deepseek-spora-joint-codec-drift-audit-20261115.json'
WORDS = ('苹果', '香蕉', '西瓜', '葡萄', '茶', '咖啡', '牛奶', '豆浆',
         '果汁', '可乐', '红色', '蓝色', '绿色', '黄色', '白色', '黑色',
         '读书', '跑步', '画画', '唱歌', '写字', '跳舞')


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    ids = sorted({x for word in WORDS for x in tok.encode(word, add_special_tokens=False)}
                 | {tok.eos_token_id})
    with safe_open(str(MODEL / 'model.safetensors'), framework='pt', device='cpu') as f:
        embedding = f.get_slice('model.embed_tokens.weight')
        selected = torch.stack([embedding[i] for i in ids]).float()
    results = []
    for source in SOURCES:
        path = ROOT / 'runs' / source / 'candidate.pt'
        payload = torch.load(path, map_location='cpu', weights_only=True)
        state = payload['codec_state']
        old = state['codebook'].index_select(0, torch.tensor(ids)).float()
        new = F.normalize(selected @ state['projection'].float(), dim=-1)
        cosine = F.cosine_similarity(old, new, dim=-1)
        scores = new @ state['codebook'].float().T
        predicted = scores.argmax(-1)
        expected = torch.tensor(ids)
        results.append({'candidate': source, 'sha256': sha(path), 'tokens': len(ids),
                        'min_cosine': float(cosine.min()), 'mean_cosine': float(cosine.mean()),
                        'max_abs_difference': float((old-new).abs().max()),
                        'old_codebook_top1_for_current_codes': int((predicted == expected).sum())})
    OUT.write_text(json.dumps({'results': results}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
