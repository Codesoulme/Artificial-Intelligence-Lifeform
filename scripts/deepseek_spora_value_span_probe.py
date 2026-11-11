"""Prototype open-vocabulary value-span extraction on replaced features.

This is a research artifact. It never writes to a SPORA individual.
"""
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
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_observation_retrain import SPLITS
from scripts.deepseek_spora_observation_transfer_audit import CHECKPOINT
from spora.integrated_checkpoint import load_integrated, replacement_feature_variant_sha256
from spora.structure_transfer import SPORALayerReplacement

OUTPUT = ROOT / 'runs/deepseek-spora-value-span-probe-v3-20261111'
NEGATIVE = {
    'train': ('{n}不喜欢喝{w}。', '{n}给朋友买了{w}。', '{n}说朋友喜欢{w}。',
              '{n}觉得{w}难喝。', '我猜{n}可能喜欢{w}。',
              '{n}从来不喝{w}。', '假如{n}喜欢{w}就好了。',
              '{n}以后不再喜欢{w}。', '{n}为客人准备了{w}。'),
    'calibration': ('{n}是否爱喝{w}？', '{n}替同事带了{w}。', '{n}以前喝{w}，现在不清楚。'),
    'held': ('{n}从来不喝{w}。', '{n}给客人准备了{w}。', '{n}说阿明喜欢{w}。',
             '如果{n}喜欢{w}就好了。', '{n}不再喜欢{w}了。'),
}
UNKNOWN = {'train': ('茉莉花茶', '苏打水', '绿豆汤'),
           'calibration': ('热巧克力', '冰红茶'),
           'held': ('柠檬水', '苹果汁', '白开水', '汽水', '橙汁')}


class SpanHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.accept = nn.Linear(width, 1)
        self.boundary = nn.Linear(width, 2)

    def forward(self, vectors):
        pooled = vectors.mean((1, 2))
        boundary = self.boundary(vectors).mean(1)
        return self.accept(pooled).squeeze(-1), boundary


def build(split):
    names = SPLITS[split][0]
    positive = SPLITS[split][1]
    rows = []
    words = ('茶', '咖啡', '牛奶', '果汁', '豆浆', '可乐', '水', '酸奶') + UNKNOWN[split]
    for name in names:
        for word in words:
            for template in positive:
                text = template.format(n=name, w=word)
                start = text.index(word)
                rows.append({'text': text, 'start': start, 'end': start + len(word), 'positive': True})
            for template in NEGATIVE[split]:
                rows.append({'text': template.format(n=name, w=word), 'start': None, 'end': None, 'positive': False})
    return rows


def feature_batch(model, tokenizer, rows):
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': r['text']}],
                                            tokenize=False, add_generation_prompt=True) for r in rows]
    encoded = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False,
                        return_offsets_mapping=True)
    offsets = encoded.pop('offset_mapping')
    mask = encoded['attention_mask'].bool()
    with torch.no_grad():
        output = model(**{k: v.to('cuda') for k, v in encoded.items()},
                       use_cache=False, output_hidden_states=True)
        vectors = torch.stack([output.hidden_states[i] for i in (6, 18)], dim=1).float().cpu()
    starts = torch.tensor([text.find(row['text']) for row, text in zip(rows, texts)], dtype=torch.long)
    if bool((starts < 0).any()):
        raise ValueError('Question text missing from chat template')
    return vectors, mask, offsets, starts


def labels(rows, offsets, starts, mask):
    batch, sites, tokens = mask.shape[0], 2, mask.shape[1]
    accepted = torch.tensor([r['positive'] for r in rows], dtype=torch.float32)
    boundaries = torch.zeros(batch, tokens, 2)
    for row_index, row in enumerate(rows):
        if not row['positive']:
            continue
        start = int(starts[row_index]) + row['start']
        end = int(starts[row_index]) + row['end']
        selected = ((offsets[row_index, :, 0] < end) &
                    (offsets[row_index, :, 1] > start) & mask[row_index])
        if not bool(selected.any()):
            raise ValueError('Value has no token overlap')
        indices = selected.nonzero(as_tuple=True)[0]
        boundaries[row_index, indices[0], 0] = 1
        boundaries[row_index, indices[-1], 1] = 1
    return accepted, boundaries


def decode(rows, model, vectors, offsets, starts, mask, threshold):
    with torch.no_grad():
        accept, boundary = model(vectors)
        accepted = accept.sigmoid() >= threshold
    result = []
    for i, row in enumerate(rows):
        available = mask[i]
        start_scores = boundary[i, :, 0].masked_fill(~available, -float('inf'))
        end_scores = boundary[i, :, 1].masked_fill(~available, -float('inf'))
        start = int(start_scores.argmax())
        end_candidates = torch.arange(start, len(end_scores))[available[start:]]
        end = int(end_candidates[end_scores[end_candidates].argmax()])
        start_char = int(offsets[i, start, 0]) - int(starts[i])
        end_char = int(offsets[i, end, 1]) - int(starts[i])
        value = row['text'][start_char:end_char] if bool(accepted[i]) else None
        result.append({**row, 'accepted': bool(accepted[i]), 'value': value,
                       'accept_logit': float(accept[i]), 'start_token': start, 'end_token': end})
    return result


def summary(rows):
    positive = [r for r in rows if r['positive']]
    negative = [r for r in rows if not r['positive']]
    positive_correct = sum(r['accepted'] and r['value'] == r['text'][r['start']:r['end']] for r in positive)
    negative_accept = sum(r['accepted'] for r in negative)
    return {'positive_correct': positive_correct, 'positive_total': len(positive),
            'negative_accept': negative_accept, 'negative_total': len(negative), 'rows': rows}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    datasets = {split: build(split) for split in ('train', 'calibration', 'held')}
    paths = (MODEL / 'model.safetensors', CHECKPOINT)
    source_hashes = {str(p): sha(p) for p in paths}
    if source_hashes[str(paths[0])] != WEIGHT_SHA:
        raise ValueError('Base weight mismatch')
    torch.manual_seed(90112)
    torch.set_num_threads(2)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    decoder, organism, state = load_integrated(CHECKPOINT, base, MODEL, individual_device='cuda')
    blocks = {i: layer for i, layer in enumerate(base.model.layers) if isinstance(layer, SPORALayerReplacement)}
    variant = replacement_feature_variant_sha256(MODEL, blocks)
    vectors, masks, offsets, starts = {}, {}, {}, {}
    for split, rows in datasets.items():
        vectors[split], masks[split], offsets[split], starts[split] = feature_batch(base, tokenizer, rows)
        print(split, len(rows), flush=True)
    model = SpanHead(vectors['train'].shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    accepted, boundary = labels(datasets['train'], offsets['train'], starts['train'], masks['train'])
    for _ in range(500):
        optimizer.zero_grad(set_to_none=True)
        accept_logits, boundary_logits = model(vectors['train'])
        active = masks['train']
        loss = F.binary_cross_entropy_with_logits(accept_logits, accepted)
        loss += F.binary_cross_entropy_with_logits(boundary_logits[active], boundary[active])
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite loss')
        loss.backward()
        optimizer.step()
    model.eval()
    calibration_rows = []
    with torch.no_grad():
        calibration_accept = model(vectors['calibration'])[0].sigmoid()
    thresholds = sorted(set(float(v) for v in calibration_accept))
    thresholds += [thresholds[-1] + 0.01]
    choices = [(summary(decode(datasets['calibration'], model, vectors['calibration'], offsets['calibration'], starts['calibration'], masks['calibration'], t)), t)
               for t in thresholds]
    feasible = [(result, t) for result, t in choices if result['negative_accept'] == 0]
    chosen, threshold = max(feasible, key=lambda pair: (pair[0]['positive_correct'], -pair[1]))
    results = {split: summary(decode(rows, model, vectors[split], offsets[split], starts[split], masks[split], threshold))
               for split, rows in datasets.items()}
    if any(sha(p) != source_hashes[str(p)] for p in paths):
        raise ValueError('Protected input changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT / 'span-head.pt'
    torch.save({'format': 'spora-open-value-span-development-v1', 'state': model.state_dict(),
                'threshold': threshold, 'base_sha256': WEIGHT_SHA,
                'source_checkpoint_sha256': source_hashes[str(CHECKPOINT)],
                'feature_variant_sha256': variant}, checkpoint)
    report = {'status': 'research_only_no_writes', 'threshold': threshold,
              'checkpoint_sha256': sha(checkpoint), 'source_hashes': source_hashes,
              'feature_variant_sha256': variant, 'replacement_layers': sorted(blocks),
              'steps': 500, 'seconds': time.perf_counter() - started, 'results': results,
              'limitations': ['Synthetic templates and finite names; not real dialogue.',
                              'Subject, source, temporal meaning and authorization are not modeled.',
                              'Token boundary decoding is experimental and no individual was changed.',
                              'A zero calibration false-accept count is not a general guarantee.']}
    (OUTPUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for split, result in results.items():
        print(split, {key: result[key] for key in ('positive_correct', 'positive_total', 'negative_accept', 'negative_total')}, flush=True)


if __name__ == '__main__':
    main()
