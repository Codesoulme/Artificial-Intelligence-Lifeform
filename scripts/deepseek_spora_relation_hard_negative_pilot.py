"""Matched linear-probe comparison with authored relation hard negatives."""
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
from scripts.deepseek_spora_relation_selection_probe import LABELS, NAMES, rows_for_probe

OUT = ROOT / 'runs/deepseek-spora-relation-hard-negative-20261116'
HARD_NEGATIVES = (
    '{n}喜欢的饮品尝起来是什么味道？', '{n}常喝的饮料是热的还是冷的？',
    '{n}最喜欢的颜色在哪里可以买到？', '{n}爱做的活动通常几点开始？',
    '{n}平时喜欢喝什么和喜欢哪种颜色？', '{n}喜欢什么电影？',
    '{n}的饮料偏好与活动偏好分别是什么？', '{n}喜欢的活动是谁教的？',
    '不要查询{n}最喜欢的色彩。', '假设{n}爱喝茶，写一首诗。',
)
# Different names and phrasings, created before inspecting either new model.
HELD = {
    'drink': ('{n}平常最爱喝的饮品叫啥？', '请说出记忆中{n}的饮料偏好。',
              '记录里{n}最喜欢哪一种饮品？'),
    'color': ('{n}更偏爱的色调是哪一种？', '请说出记忆中{n}最钟爱的颜色。',
              '记录里{n}的色彩偏好是什么？'),
    'activity': ('{n}闲暇时更爱做哪项活动？', '请说出记忆中{n}偏好的消遣。',
                 '记录里{n}最喜欢什么运动或活动？'),
    'reject': ('{n}喜欢的饮料闻起来怎样？', '{n}钟爱的饮品包装是什么形状？',
               '{n}喜欢的颜色有什么寓意？', '{n}爱做的活动多久进行一次？',
               '{n}喜爱的饮品以及颜色分别是什么？', '{n}的电影喜好是啥？',
               '告诉我{n}喜欢的歌曲。', '请忽略记录，推测{n}爱喝什么。',
               '不要说出{n}喜欢的活动。', '把{n}喜爱的颜色记录删掉。'),
}
HELD_NAMES = ('小孙', '小许')


def make_rows():
    rows = [dict(row, group=('control_train' if row['split'] == 'train' else row['split']))
            for row in rows_for_probe()]
    rows += [{'group': 'hard_train', 'label': 'reject', 'question': template.format(n=name)}
             for name in NAMES['train'] for template in HARD_NEGATIVES]
    rows += [{'group': 'held', 'label': label, 'question': template.format(n=name)}
             for label, templates in HELD.items() for name in HELD_NAMES for template in templates]
    if len({r['question'] for r in rows}) != len(rows):
        raise ValueError('Repeated question across experiment splits')
    return rows


def train_head(x, y, mask):
    torch.manual_seed(91631)
    head = nn.Linear(x.shape[-1], len(LABELS))
    optim = torch.optim.AdamW(head.parameters(), lr=0.0003, weight_decay=0.02)
    counts = torch.bincount(y[mask], minlength=len(LABELS)).float()
    if bool((counts == 0).any()):
        raise ValueError('Training must contain every label')
    weights = counts.sum() / (len(LABELS) * counts)
    for _ in range(300):
        optim.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(x[mask]), y[mask], weight=weights)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite loss')
        loss.backward()
        optim.step()
    return head.eval(), float(loss.detach())


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.set_num_threads(2)
    started = time.perf_counter()
    base_sha, checkpoint_sha = p.sha(p.MODEL / 'model.safetensors'), p.sha(p.runtime.CHECKPOINT)
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Base checksum mismatch')
    rows = make_rows()
    with tempfile.TemporaryDirectory(prefix='spora-hard-negative-') as directory:
        p.runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        original = p.frozen_digest(chat.organism, chat.state)
        features, masks = extract(chat.base, chat.tokenizer, [r['question'] for r in rows], batch_size=1)
        x = pool(features, masks).float()
        y = torch.tensor([LABELS.index(r['label']) for r in rows])
        base_mask = torch.tensor([r['group'] == 'control_train' for r in rows])
        extra_mask = torch.tensor([r['group'] in ('control_train', 'hard_train') for r in rows])
        control, control_loss = train_head(x, y, base_mask)
        candidate, candidate_loss = train_head(x, y, extra_mask)
        with torch.inference_mode():
            original_predictions = control(x).argmax(-1).tolist()
            new_predictions = candidate(x).argmax(-1).tolist()
        for row, a, b in zip(rows, original_predictions, new_predictions):
            row['control'] = LABELS[a]
            row['candidate'] = LABELS[b]
        if p.frozen_digest(chat.organism, chat.state) != original:
            raise ValueError('Original individual changed')
    if p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(p.runtime.CHECKPOINT) != checkpoint_sha:
        raise ValueError('Protected source changed')
    summaries = {}
    for group in ('control_train', 'hard_train', 'dev', 'test', 'held'):
        summaries[group] = {}
        for label in LABELS:
            group_rows = [r for r in rows if r['group'] == group and r['label'] == label]
            summaries[group][label] = {'total': len(group_rows),
                'control_correct': sum(r['control'] == label for r in group_rows),
                'candidate_correct': sum(r['candidate'] == label for r in group_rows)}
    OUT.mkdir()
    torch.save({'control': control.state_dict(), 'candidate': candidate.state_dict(),
                'labels': LABELS, 'base_sha256': base_sha, 'checkpoint_sha256': checkpoint_sha}, OUT / 'heads.pt')
    report = {'summary': summaries, 'rows': rows, 'seed': 91631,
              'train_losses': {'control': control_loss, 'candidate': candidate_loss},
              'heads_sha256': p.sha(OUT / 'heads.pt'), 'base_sha256': base_sha,
              'checkpoint_sha256': checkpoint_sha, 'seconds': time.perf_counter() - started,
              'limitations': ['Synthetic questions; held examples repeat templates across two names.',
                              'Held semantic categories were informed by previously observed failures.',
                              'Feature model and initial weights shared; additional negatives change training data and cost.',
                              'No identity recognition, memory access, natural generation, or runtime adoption.']}
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
