"""Measure text-writer generalization on a broad stable Chinese token split."""
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch.nn import functional as F

from scripts import deepseek_local_baseline as p
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.language_training import atomic_save
from spora.semantic_text_writer import TextSemanticWriter, value_positions


OUT = ROOT / 'runs/deepseek-spora-text-writer-broad-generalization-20261116'
TRAIN_NAMES = ('小林', '阿宁')
HELD_NAMES = ('小赵', '小郑')
TRAIN_COUNT = 256
HELD_COUNT = 64
STEPS = 512
LR = 3e-4
SEED = 93471


def stable_cjk_pool(tokenizer):
    pool = []
    for token_id in range(tokenizer.vocab_size):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if (len(text) == 1 and '\u4e00' <= text <= '\u9fff'
                and tokenizer.encode(text, add_special_tokens=False) == [token_id]):
            pool.append(text)
    return pool


def collect(model, tokenizer, embedding, names, values):
    features, targets = [], []
    for name in names:
        for value in values:
            fact = f'{name}喜欢喝{value}。'
            encoded, positions = value_positions(tokenizer, fact, value)
            device = next(model.parameters()).device
            inputs = {key: tensor.to(device) for key, tensor in encoded.items()}
            with torch.inference_mode():
                hidden = model.model(**inputs, use_cache=False, return_dict=True).last_hidden_state[0].float()
            token_id = tokenizer.encode(value, add_special_tokens=False)[0]
            features.append(hidden[positions[0]])
            targets.append(embedding[token_id])
    return torch.stack(features), torch.stack(targets)


def metrics(features, targets, writer):
    with torch.inference_mode():
        direct = F.normalize(features, dim=-1)
        predicted = F.normalize(writer(features), dim=-1)
        expected = F.normalize(targets, dim=-1)
        direct_cosine = (direct * expected).sum(-1)
        writer_cosine = (predicted * expected).sum(-1)
    return {'cases': int(writer_cosine.numel()),
            'direct_cosine_mean': float(direct_cosine.mean()),
            'writer_cosine_mean': float(writer_cosine.mean()),
            'writer_cosine_min': float(writer_cosine.min()),
            'writer_top90': int((writer_cosine >= .9).sum())}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Unexpected base weight')
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True,
                                                trust_remote_code=False)
    pool = stable_cjk_pool(tokenizer)
    if len(pool) < TRAIN_COUNT + HELD_COUNT:
        raise ValueError('Stable Chinese token pool is too small')
    random.Random(SEED).shuffle(pool)
    train_values, held_values = pool[:TRAIN_COUNT], pool[TRAIN_COUNT:TRAIN_COUNT + HELD_COUNT]
    model = load_full_streamed().eval()
    model.requires_grad_(False)
    embedding = model.model.embed_tokens.weight.detach().float()
    train_features, train_targets = collect(model, tokenizer, embedding, TRAIN_NAMES, train_values)
    held_features, held_targets = collect(model, tokenizer, embedding, HELD_NAMES, held_values)
    print(json.dumps({'prepared_train': len(train_features), 'prepared_held': len(held_features)}, ensure_ascii=False), flush=True)
    writer = TextSemanticWriter(embedding.shape[-1]).to('cuda')
    before = {'train': metrics(train_features, train_targets, writer),
              'held': metrics(held_features, held_targets, writer)}
    optimizer = torch.optim.AdamW(writer.parameters(), lr=LR, weight_decay=1e-5, foreach=False)
    trace = []
    for step in range(STEPS):
        prediction = writer(train_features)
        cosine_loss = 1.0 - F.cosine_similarity(prediction, train_targets, dim=-1).mean()
        scale = train_targets.detach().norm(dim=-1).mean().clamp_min(1e-6)
        mse_loss = ((prediction - train_targets.detach()) / scale).square().mean()
        loss = cosine_loss + .25 * mse_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(writer.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 64 == 0:
            trace.append({'step': step + 1, 'loss': float(loss.detach()),
                          'cosine_loss': float(cosine_loss.detach()),
                          'gradient_norm': float(gradient)})
            print(json.dumps(trace[-1]), flush=True)
    after = {'train': metrics(train_features, train_targets, writer),
             'held': metrics(held_features, held_targets, writer)}
    OUT.mkdir()
    writer_path = OUT / 'writer.pt'
    atomic_save({'format': 'spora-text-semantic-writer-broad-v1', 'base_sha256': base_sha,
                 'width': embedding.shape[-1], 'seed': SEED, 'steps': STEPS,
                 'train_names': TRAIN_NAMES, 'held_names': HELD_NAMES,
                 'train_values': train_values, 'held_values': held_values,
                 'state': {key: value.detach().cpu() for key, value in writer.state_dict().items()}}, writer_path)
    result = {'status': 'text_writer_broad_generalization', 'before': before, 'after': after,
              'trace': trace, 'train_count': TRAIN_COUNT, 'held_count': HELD_COUNT,
              'base_sha256': base_sha, 'writer_sha256': p.sha(writer_path),
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['This is hidden-state-to-embedding reconstruction, not generation.',
                              'Stable CJK token split is synthetic and does not test natural fact parsing.',
                              'Held values are token-held, not a complete open-vocabulary guarantee.']}
    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'before': before, 'after': after}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
