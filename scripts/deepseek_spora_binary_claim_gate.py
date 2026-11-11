"""Develop a binary factual-claim confidence gate beside the multiclass gate."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_value_span_probe import feature_batch
from scripts.deepseek_spora_observation_retrain import SPLITS
from scripts.deepseek_spora_observation_transfer_audit import CHECKPOINT
from scripts.deepseek_spora_polarity_gate_probe import NEGATIVE
from spora.integrated_checkpoint import load_integrated, replacement_feature_variant_sha256
from spora.structure_transfer import SPORALayerReplacement

OUTPUT = ROOT / 'runs/deepseek-spora-binary-claim-gate-20261111'
UNKNOWN = {'train': ('茉莉花茶', '苏打水', '绿豆汤'),
           'calibration': ('热巧克力', '冰红茶'),
           'held': ('柠檬水', '苹果汁', '白开水', '汽水', '橙汁')}


def build(split):
    names, positives, _ = SPLITS[split]
    words = ('茶', '咖啡', '牛奶', '果汁', '豆浆', '可乐', '水', '酸奶') + UNKNOWN[split]
    rows = []
    for name in names:
        for word in words:
            for template in positives:
                rows.append({'text': template.format(n=name, w=word), 'label': 1})
            for templates in NEGATIVE[split].values():
                for template in templates:
                    rows.append({'text': template.format(n=name, w=word), 'label': 0})
    return rows


def evaluate(model, vectors, rows, threshold=.9):
    with torch.inference_mode():
        probability = model(vectors).sigmoid().squeeze(-1)
    result = [{**row, 'probability': float(probability[i]),
               'accepted': bool(probability[i] >= threshold)} for i, row in enumerate(rows)]
    return {'positive_total': sum(row['label'] == 1 for row in result),
            'positive_accepted': sum(row['label'] == 1 and row['accepted'] for row in result),
            'negative_total': sum(row['label'] == 0 for row in result),
            'negative_accepted': sum(row['label'] == 0 and row['accepted'] for row in result),
            'rows': result}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    datasets = {split: build(split) for split in ('train', 'calibration', 'held')}
    paths = (MODEL / 'model.safetensors', CHECKPOINT)
    source = {str(path): sha(path) for path in paths}
    if source[str(paths[0])] != WEIGHT_SHA:
        raise ValueError('Base provenance mismatch')
    torch.manual_seed(90116)
    torch.set_num_threads(2)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    decoder, organism, state = load_integrated(CHECKPOINT, base, MODEL, individual_device='cuda')
    del decoder, organism, state
    blocks = {i: layer for i, layer in enumerate(base.model.layers) if isinstance(layer, SPORALayerReplacement)}
    variant = replacement_feature_variant_sha256(MODEL, blocks)
    vectors = {}
    for split, rows in datasets.items():
        features, masks, _, _ = feature_batch(base, tokenizer, rows)
        weights = masks[:, None, :, None].float()
        means = (features.float() * weights).sum(2) / weights.sum(2).clamp_min(1)
        vectors[split] = torch.cat([F.layer_norm(means[:, i], (means.shape[-1],)) for i in range(2)], dim=-1)
        print(split, len(rows), flush=True)
    model = nn.Linear(vectors['train'].shape[-1], 1)
    labels = torch.tensor([row['label'] for row in datasets['train']], dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.01)
    for _ in range(600):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(vectors['train']).squeeze(-1), labels)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite loss')
        loss.backward()
        optimizer.step()
    model.eval()
    results = {split: evaluate(model, vectors[split], rows) for split, rows in datasets.items()}
    if any(sha(Path(path)) != digest for path, digest in source.items()):
        raise ValueError('Protected input changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT / 'gate.pt'
    torch.save({'format': 'spora-binary-claim-gate-development-v1', 'state': model.state_dict(),
                'write_floor': .9, 'base_sha256': WEIGHT_SHA, 'source_checkpoint_sha256': source[str(CHECKPOINT)],
                'feature_variant_sha256': variant}, checkpoint)
    report = {'status': 'research_only_no_writes', 'write_floor': .9, 'checkpoint_sha256': sha(checkpoint),
              'source_hashes': source, 'feature_variant_sha256': variant, 'replacement_layers': sorted(blocks),
              'results': results, 'seconds': time.perf_counter() - started,
              'limitations': ['Synthetic templates and finite values/names.',
                              'Binary factuality does not replace polarity, source, time or subject gates.',
                              'The 0.9 floor was not tuned and no persistence occurred.',
                              'No independent dialogue benchmark.']}
    (OUTPUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for split, result in results.items():
        print(split, {key: result[key] for key in ('positive_accepted', 'positive_total', 'negative_accepted', 'negative_total')}, flush=True)


if __name__ == '__main__':
    main()
