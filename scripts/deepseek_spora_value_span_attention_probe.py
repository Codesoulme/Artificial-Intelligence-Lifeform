"""Train an attentive open-vocabulary observation span probe on SPORA features."""
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
from scripts.deepseek_spora_observation_retrain import SPLITS
from scripts.deepseek_spora_value_span_probe import UNKNOWN
from scripts.deepseek_spora_observation_transfer_audit import CHECKPOINT
from spora.integrated_checkpoint import load_integrated, replacement_feature_variant_sha256
from spora.structure_transfer import SPORALayerReplacement

OUTPUT = ROOT / 'runs/deepseek-spora-value-span-hybrid-v4-20261111'
NEGATIVE = {
    'train': ('{n}不喜欢喝{w}。', '{n}给朋友买了{w}。', '{n}说朋友喜欢{w}。',
              '{n}觉得{w}难喝。', '我猜{n}可能喜欢{w}。', '{n}从来不喝{w}。',
              '假如{n}喜欢{w}就好了。', '{n}以后不再喜欢{w}。', '{n}为客人准备了{w}。'),
    'calibration': ('{n}是否爱喝{w}？', '{n}替同事带了{w}。', '{n}以前喝{w}，现在不清楚。'),
    'held': ('{n}从来不喝{w}。', '{n}给客人准备了{w}。', '{n}说阿明喜欢{w}。',
             '如果{n}喜欢{w}就好了。', '{n}不再喜欢{w}了。'),
}


class AttentiveSpanHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.accept = nn.Linear(width * 2, 1)
        self.boundary = nn.Linear(width, 2)

    def forward(self, vectors, mask):
        vectors = vectors.mean(1)
        weights = mask.unsqueeze(-1).float()
        mean = (vectors * weights).sum(1) / weights.sum(1).clamp_min(1)
        maximum = vectors.masked_fill(~mask.unsqueeze(-1), -float('inf')).max(1).values
        pooled = torch.cat([mean, maximum], dim=-1)
        return self.accept(pooled).squeeze(-1), self.boundary(vectors)


def build(split):
    names, positive, _ = SPLITS[split]
    words = ('茶', '咖啡', '牛奶', '果汁', '豆浆', '可乐', '水', '酸奶') + UNKNOWN[split]
    rows = []
    for name in names:
        for word in words:
            for template in positive:
                text = template.format(n=name, w=word)
                start = text.index(word)
                rows.append({'text': text, 'start': start, 'end': start + len(word), 'positive': True})
            for template in NEGATIVE[split]:
                rows.append({'text': template.format(n=name, w=word), 'start': None, 'end': None, 'positive': False})
    return rows


def features(model, tokenizer, rows):
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': row['text']}], tokenize=False,
                                            add_generation_prompt=True) for row in rows]
    encoded = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False,
                        return_offsets_mapping=True)
    offsets = encoded.pop('offset_mapping')
    mask = encoded['attention_mask'].bool()
    starts = torch.tensor([text.find(row['text']) for row, text in zip(rows, texts)], dtype=torch.long)
    with torch.no_grad():
        output = model(**{k: v.to('cuda') for k, v in encoded.items()}, use_cache=False, output_hidden_states=True)
        vectors = torch.stack([output.hidden_states[i] for i in (6, 18)], dim=1).float().cpu()
    return vectors, mask, offsets, starts


def targets(rows, offsets, starts, mask):
    accepted = torch.tensor([row['positive'] for row in rows], dtype=torch.float32)
    boundary = torch.zeros(len(rows), mask.shape[1], 2)
    for i, row in enumerate(rows):
        if not row['positive']:
            continue
        begin, end = int(starts[i]) + row['start'], int(starts[i]) + row['end']
        selected = (offsets[i, :, 0] < end) & (offsets[i, :, 1] > begin) & mask[i]
        if not bool(selected.any()):
            raise ValueError('Missing value token')
        ids = selected.nonzero(as_tuple=True)[0]
        boundary[i, ids[0], 0] = 1
        boundary[i, ids[-1], 1] = 1
    return accepted, boundary


def decode(rows, model, vectors, mask, offsets, starts, threshold):
    with torch.no_grad():
        accept, boundary = model(vectors, mask)
        accepted = accept.sigmoid() >= threshold
    result = []
    for i, row in enumerate(rows):
        available = mask[i]
        start_scores = boundary[i, :, 0].masked_fill(~available, -float('inf'))
        start = int(start_scores.argmax())
        candidates = torch.arange(start, len(available))[available[start:]]
        end = int(candidates[boundary[i, candidates, 1].argmax()])
        begin_char = int(offsets[i, start, 0]) - int(starts[i])
        end_char = int(offsets[i, end, 1]) - int(starts[i])
        value = row['text'][begin_char:end_char] if bool(accepted[i]) else None
        result.append({**row, 'accepted': bool(accepted[i]), 'value': value,
                       'accept_logit': float(accept[i])})
    return result


def summarize(rows):
    positive = [row for row in rows if row['positive']]
    negative = [row for row in rows if not row['positive']]
    return {'positive_correct': sum(row['accepted'] and row['value'] == row['text'][row['start']:row['end']] for row in positive),
            'positive_total': len(positive), 'negative_accept': sum(row['accepted'] for row in negative),
            'negative_total': len(negative), 'rows': rows}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    datasets = {split: build(split) for split in ('train', 'calibration', 'held')}
    paths = (MODEL / 'model.safetensors', CHECKPOINT)
    source = {str(path): sha(path) for path in paths}
    if source[str(paths[0])] != WEIGHT_SHA:
        raise ValueError('Base hash mismatch')
    torch.manual_seed(90113)
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
        vectors[split], masks[split], offsets[split], starts[split] = features(base, tokenizer, rows)
        print(split, len(rows), flush=True)
    model = AttentiveSpanHead(vectors['train'].shape[-1])
    accepted, boundary = targets(datasets['train'], offsets['train'], starts['train'], masks['train'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.01)
    for _ in range(700):
        optimizer.zero_grad(set_to_none=True)
        accept_logits, boundary_logits = model(vectors['train'], masks['train'])
        loss = F.binary_cross_entropy_with_logits(accept_logits, accepted)
        loss += F.binary_cross_entropy_with_logits(boundary_logits[masks['train']], boundary[masks['train']])
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite loss')
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        scores = model(vectors['calibration'], masks['calibration'])[0].sigmoid()
    thresholds = sorted(set(float(score) for score in scores))
    thresholds.append(thresholds[-1] + 0.01)
    choices = [(summarize(decode(datasets['calibration'], model, vectors['calibration'], masks['calibration'], offsets['calibration'], starts['calibration'], threshold)), threshold)
               for threshold in thresholds]
    feasible = [(result, threshold) for result, threshold in choices if result['negative_accept'] == 0]
    selected, threshold = max(feasible, key=lambda item: (item[0]['positive_correct'], -item[1]))
    results = {split: summarize(decode(rows, model, vectors[split], masks[split], offsets[split], starts[split], threshold))
               for split, rows in datasets.items()}
    if any(sha(path) != source[str(path)] for path in paths):
        raise ValueError('Protected input changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT / 'span-head.pt'
    torch.save({'format': 'spora-attentive-open-value-span-development-v1', 'state': model.state_dict(),
                'threshold': threshold, 'base_sha256': WEIGHT_SHA, 'source_checkpoint_sha256': source[str(CHECKPOINT)],
                'feature_variant_sha256': variant}, checkpoint)
    report = {'status': 'research_only_no_writes', 'threshold': threshold,
              'checkpoint_sha256': sha(checkpoint), 'source_hashes': source,
              'feature_variant_sha256': variant, 'replacement_layers': sorted(blocks),
              'steps': 700, 'seconds': time.perf_counter() - started, 'results': results,
              'limitations': ['Synthetic templates and finite names; no independent dialogue.',
                              'No subject, source, temporal, authorization or correction modeling.',
                              'Threshold selected on finite calibration data; no general guarantee.',
                              'No persistent memory writes.']}
    (OUTPUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for split, result in results.items():
        print(split, {key: result[key] for key in ('positive_correct', 'positive_total', 'negative_accept', 'negative_total')}, flush=True)


if __name__ == '__main__':
    main()
