"""Isolated multiclass probe for question-conditioned relation selection.

No memory values, automatic writes, or production routing are used here.
"""
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
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_pooled_address import extract

OUT = ROOT / 'runs/deepseek-spora-relation-selection-probe-20261116'
NAMES = {'train': ('小林', '小周', '小王', '小李'),
         'dev': ('小赵', '小吴'), 'test': ('小郑', '小钱')}
TEMPLATES = {
    'train': {
        'drink': ('{n}喜欢喝什么饮料？', '查一下{n}偏爱的饮品名称。', '{n}最爱喝哪种东西？', '说出{n}的饮料偏好。'),
        'color': ('{n}喜欢什么颜色？', '查一下{n}偏爱的颜色名称。', '{n}最爱哪种色彩？', '说出{n}的颜色偏好。'),
        'activity': ('{n}喜欢什么活动？', '查一下{n}偏爱的活动名称。', '{n}最爱做什么事？', '说出{n}的活动偏好。'),
        'reject': ('不要查{n}喜欢的饮料。', '把{n}的偏好改成茶。', '猜猜{n}喜欢哪种颜色。', '给{n}推荐一种活动。', '{n}的饮料是什么颜色？', '删除{n}的颜色记录。')},
    'dev': {
        'drink': ('从记录看{n}爱喝哪种饮品？', '请回忆{n}的饮料喜好。'),
        'color': ('从记录看{n}偏好哪种颜色？', '请回忆{n}的色彩喜好。'),
        'activity': ('从记录看{n}偏好哪种活动？', '请回忆{n}平时爱做的事。'),
        'reject': ('别告诉我{n}爱喝什么。', '假如{n}喜欢蓝色，写个故事。', '请推荐{n}一个爱好。', '{n}爱喝的饮品多少钱？')},
    'test': {
        'drink': ('记忆里{n}最中意的饮料是？', '说说{n}平时更爱喝哪一款。'),
        'color': ('记忆里{n}最中意的色彩是？', '说说{n}平时更喜欢哪种颜色。'),
        'activity': ('记忆里{n}最中意的消遣是？', '说说{n}平时更喜欢做什么。'),
        'reject': ('请不要回忆{n}的饮品。', '假设{n}爱画画，请讲故事。', '替{n}挑一种颜色吧。', '{n}喜欢的饮料味道如何？')},
}
LABELS = ('drink', 'color', 'activity', 'reject')


def rows_for_probe():
    rows = [{'split': split, 'label': label, 'question': template.format(n=name)}
            for split, groups in TEMPLATES.items() for label, templates in groups.items()
            for name in NAMES[split] for template in templates]
    if len({row['question'] for row in rows}) != len(rows):
        raise ValueError('Duplicate probe questions')
    return rows


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.manual_seed(91631)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    source_sha = p.sha(p.runtime.CHECKPOINT)
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Base checksum mismatch')
    rows = rows_for_probe()
    with tempfile.TemporaryDirectory(prefix='spora-relation-selection-') as temp:
        p.runtime.WORKING_CHECKPOINT = Path(temp) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        protected = p.frozen_digest(chat.organism, chat.state)
        features, masks = extract(chat.base, chat.tokenizer, [r['question'] for r in rows], batch_size=1)
        x = pool(features, masks).float()
        y = torch.tensor([LABELS.index(r['label']) for r in rows])
        train = torch.tensor([r['split'] == 'train' for r in rows])
        head = nn.Linear(x.shape[-1], len(LABELS))
        optim = torch.optim.AdamW(head.parameters(), lr=0.0003, weight_decay=0.02)
        counts = torch.bincount(y[train], minlength=len(LABELS)).float()
        weights = counts.sum() / (len(LABELS) * counts)
        for _ in range(300):
            optim.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(x[train]), y[train], weight=weights)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite probe loss')
            loss.backward()
            optim.step()
        with torch.no_grad():
            scores = head(x)
            predictions = scores.argmax(-1)
            for i, row in enumerate(rows):
                row['predicted'] = LABELS[int(predictions[i])]
                row['correct'] = row['predicted'] == row['label']
        if p.frozen_digest(chat.organism, chat.state) != protected:
            raise ValueError('Original individual changed')
        state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    if p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(p.runtime.CHECKPOINT) != source_sha:
        raise ValueError('Protected source changed')
    summaries = {}
    for split in NAMES:
        summaries[split] = {}
        for label in LABELS:
            group = [r for r in rows if r['split'] == split and r['label'] == label]
            summaries[split][label] = {'correct': sum(r['correct'] for r in group), 'total': len(group),
                                      'predicted': {kind: sum(r['predicted'] == kind for r in group) for kind in LABELS}}
    OUT.mkdir()
    torch.save({'state': state, 'labels': LABELS, 'base_sha256': base_sha,
                'checkpoint_sha256': source_sha}, OUT / 'head.pt')
    result = {'summary': summaries, 'rows': rows, 'seed': 91631, 'loss': float(loss.detach()),
              'base_sha256': base_sha, 'checkpoint_sha256': source_sha,
              'head_sha256': p.sha(OUT / 'head.pt'), 'seconds': time.perf_counter() - start,
              'limitations': ['Synthetic examples; test wording previously unseen only within this probe.',
                              'Only pooled question features and explicit multiclass labels, no actual memory read.',
                              'No unknown-entity, correction, held relation, or natural dialogue evaluation.',
                              'No calibrated abstention or authorization guarantee.']}
    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
