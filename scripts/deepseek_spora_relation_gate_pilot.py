"""Train an isolated finite-beverage-relation classifier on frozen features."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch import nn
from torch.nn import functional as F
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_named_binding_probe import NAMES
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_pooled_address import extract

OUT = ROOT / 'runs/deepseek-spora-relation-gate-pilot-20261114'
SEED = 70183
TEMPLATES = {
    'train': {
        'read': ('请告诉我{name}喜欢喝什么饮品。', '{name}最爱喝哪一种饮料？',
                 '查找{name}现在的饮品偏好。', '记录里{name}喜欢的饮料是哪种？'),
        'attribute': ('{name}喜欢的饮料是什么颜色？', '{name}最爱的饮品价格是多少？',
                      '谁买了{name}喜欢的饮料？', '{name}的饮品是热的还是冷的？'),
        'negative': ('不要说出{name}喜欢喝的饮料。', '别查询{name}的饮品偏好。'),
        'write': ('把{name}的饮料偏好改成茶。', '请记录{name}喜欢喝水。'),
        'hypothesis': ('假如{name}喜欢喝牛奶，请写个故事。', '猜一猜{name}爱喝什么。'),
        'recommend': ('请给{name}推荐一种饮料。', '给{name}介绍一种饮品。'),
    },
    'calibration': {
        'read': ('想知道{name}的饮料喜好是什么。', '回忆{name}目前喜欢喝哪种东西。'),
        'attribute': ('{name}爱喝的饮品有什么颜色？', '那瓶{name}爱喝的饮料多少钱？'),
        'negative': ('我不想知道{name}的饮品喜好。',),
        'write': ('删掉{name}喜欢的饮料记录。',),
        'hypothesis': ('假设{name}爱喝咖啡，讲个故事。',),
        'recommend': ('为{name}挑选一种饮料吧。',),
    },
    'test': {
        'read': ('依照记录，{name}平时偏爱哪种饮品？', '请说出{name}保存的饮料名称。',
                 '能从记忆里找出{name}的喝饮料喜好吗？'),
        'attribute': ('{name}喜欢喝的饮料味道怎么样？', '{name}钟爱的饮品有多少毫升？',
                      '请告诉我{name}爱喝的饮料是什么颜色。'),
        'negative': ('不用告诉我{name}的饮品偏好。', '请不要查{name}爱喝什么。'),
        'write': ('将{name}爱喝的东西更新为果汁。', '忘掉{name}的饮料喜好。'),
        'hypothesis': ('假设{name}偏爱饮料，帮我续写故事。', '推测一下{name}可能爱喝什么。'),
        'recommend': ('替{name}选一款没试过的饮品。', '推荐给{name}一种解渴的饮料。'),
    },
}
NAMES_BY_SPLIT = {'train': NAMES[:4], 'calibration': NAMES[4:6], 'test': NAMES[6:8]}


def make_rows():
    rows = []
    for split, groups in TEMPLATES.items():
        for group, templates in groups.items():
            for name in NAMES_BY_SPLIT[split]:
                for template in templates:
                    rows.append({'split': split, 'group': group, 'question': template.format(name=name),
                                 'label': int(group == 'read')})
    questions = [row['question'] for row in rows]
    if len(questions) != len(set(questions)):
        raise ValueError('Repeated examples across splits')
    return rows


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Base checksum mismatch')
    rows = make_rows()
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='spora-relation-gate-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        original_digest = p.frozen_digest(chat.organism, chat.state)
        features, masks = extract(chat.base, chat.tokenizer, [r['question'] for r in rows], batch_size=1)
        vectors = pool(features, masks).float()
        labels = torch.tensor([r['label'] for r in rows], dtype=torch.float32)
        train = torch.tensor([r['split'] == 'train' for r in rows])
        calibration = torch.tensor([r['split'] == 'calibration' for r in rows])
        model = nn.Linear(vectors.shape[-1], 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.02)
        pos_weight = (int(train.sum()) - labels[train].sum()) / labels[train].sum()
        for _ in range(300):
            optimizer.zero_grad(set_to_none=True)
            logits = model(vectors[train]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels[train], pos_weight=pos_weight)
            loss = loss + 0.01 * model.weight.square().sum()
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite relation loss')
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            scores = model(vectors).squeeze(-1)
        calibration_negative = scores[calibration & (labels == 0)]
        threshold = float(torch.nextafter(calibration_negative.max(), torch.tensor(float('inf'))))
        for i, row in enumerate(rows):
            row['score'] = float(scores[i])
            row['accepted'] = row['score'] >= threshold
        if p.frozen_digest(chat.organism, chat.state) != original_digest:
            raise ValueError('Original state changed')
        gate_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        feature_variant = chat.router['feature_variant_sha256']
    if p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(p.runtime.CHECKPOINT) != checkpoint_sha:
        raise ValueError('Protected files changed')
    groups = {}
    for split in ('train', 'calibration', 'test'):
        groups[split] = {}
        for group in TEMPLATES[split]:
            subset = [r for r in rows if r['split'] == split and r['group'] == group]
            groups[split][group] = {'cases': len(subset), 'accepted': sum(r['accepted'] for r in subset)}
    OUT.mkdir()
    gate_path = OUT / 'gate.pt'
    torch.save({'format': 'spora-beverage-relation-gate-pilot-v1', 'state': gate_state,
                'threshold': threshold, 'base_sha256': base_sha, 'checkpoint_sha256': checkpoint_sha,
                'feature_variant_sha256': feature_variant}, gate_path)
    report = {'groups': groups, 'rows': rows, 'training_loss': float(loss.detach()),
              'threshold': threshold, 'seed': SEED, 'base_sha256': base_sha,
              'checkpoint_sha256': checkpoint_sha, 'feature_variant_sha256': feature_variant,
              'gate_sha256': p.sha(gate_path), 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Synthetic templates and names; possible overlap with earlier development.',
                              'Held names and templates are disjoint only within this experiment.',
                              'Threshold selected to reject this small calibration-negative set, not a statistical guarantee.',
                              'Offline intent classification only; no identity, temporal routing, memory read or answer tested.',
                              'Finite beverage relation only, not a general semantic parser or access control.']}
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(groups), flush=True)


if __name__ == '__main__':
    main()
