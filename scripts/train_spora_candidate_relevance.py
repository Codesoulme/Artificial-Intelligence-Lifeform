"""Fit an absolute field match score before reading any version tensor."""

import argparse
from collections import Counter
from itertools import pairwise
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from scripts.audit_spora_grouped_revision_routes import REQUESTS, SCENES
from scripts.spora_version_chat import FrozenTurnFeatures, base, local_model
from scripts.train_spora_recurrent_counterfactual import sha, store_json
from spora.language_training import atomic_save


# Each field has its catalog name and two spoken descriptions.
TRAIN = (
    ('校车', ('出发时间', '什么时候发车', '从学校几点开走'), ('乘车地点', '在哪上车', '等班车的位置')),
    ('家具维修', ('取件柜台', '在哪儿领回来', '领修好东西到哪个台'), ('代领人员', '谁来替我取', '托付给哪位去拿')),
    ('讲座', ('主持人员', '谁负责主持', '请哪位来串场'), ('报告厅', '在哪间厅举行', '报告安排在哪儿')),
    ('药品寄送', ('包裹数量', '总共寄几件', '一共有多少包'), ('签收人员', '由谁签收', '包裹交到哪位手上')),
    ('苗圃参访', ('开放日期', '哪天开放', '定在什么日子参观'), ('入口位置', '从哪个口进', '参访要走哪道门')),
    ('档案整理', ('卷宗标题', '卷宗叫什么', '封面要写哪个名字'), ('保存位置', '放在什么柜里', '卷宗收在哪儿')),
)
CALIBRATION = (
    ('跑团活动', ('起跑时刻', '几点开跑', '从什么时候开始跑'), ('号码布领取点', '去哪里拿号码布', '比赛号码在哪儿领')),
    ('声乐考核', ('评委姓名', '谁负责打分', '哪位老师评分'), ('考核教室', '在哪一个房间考', '考核在哪儿举行')),
    ('报刊配送', ('发行数量', '送多少本来', '这回配送的册数'), ('交接人员', '交给哪一位', '哪位来接这批报刊')),
)
HELD = (
    ('陶艺巡展', ('现场引导员', '谁来带路', '负责指路的是哪位'), ('布展时段', '什么时候能进来布置', '几点开始摆展')),
    ('读书沙龙', ('活动会场', '大家到什么地方见', '到哪儿集合座谈'), ('选读篇目', '这次大家读哪一篇', '该挑哪篇来读')),
    ('文创领用', ('领取件数', '到底要领多少件', '实际拿走多少个'), ('代办人员', '哪一位帮忙取走', '请谁来代为领取')),
    ('微电影放映', ('散场时刻', '几点放完', '大概何时结束'), ('出口方位', '结束后该从哪儿走出去', '往哪边出门')),
)
FRAMES = (
    ('麻烦核对{first}，顺便告诉我{second}。', '帮我列出{first}，还要注明{second}。'),
    ('交接单需要填{first}和{second}。', '现在请查清{first}，以及{second}。'),
    ('我得把{first}和{second}发给同事，帮我查下。', '我要确认两件事，{first}，另外{second}。'),
)
UNAVAILABLE = (
    ('紧急联络电话', '备用电源型号'),
    ('延期处理办法', '登记表颜色'),
    ('现场应急出口', '预计报销金额'),
)
SEED = 171127

HARD_HELD = (
    ('峡谷赛事几点开始检录，领号码牌的窗口在哪里？',
     ('峡谷赛事的检录时间', '峡谷赛事的完赛时间', '峡谷赛事的领牌窗口', '山地赛事的领牌窗口'), (0, 2)),
    ('古乐会在哪儿彩排，花灯会的成品送到哪里？',
     ('古乐会的演出场地', '花灯会的制作场地', '花灯会的成品交接点', '古乐会的彩排场地'), (2, 3)),
    ('谁负责收瑜伽课的垫子，茶艺课几点开始？',
     ('瑜伽课的材料归还时间', '瑜伽课的器材回收人员', '瑜伽课的开课时间', '茶艺课的结束时间'), (1,)),
    ('科普短片的导演是哪位，江边摄影活动什么时候开拍？',
     ('江边摄影的带队人员', '江边摄影的开拍时段', '科普短片的剪辑人员', '科普短片的上映时段'), (1,)),
    ('小剧场的入场门在哪，散场从哪儿出去？',
     ('小剧场的停车入口', '露台影厅的观众入口', '露台影厅的观众出口', '小剧场的检票时间'), ()),
    ('铜器展的撤展负责人是谁，石刻展哪天开始布展？',
     ('铜器展的布展负责人', '石刻展的撤展负责人', '石刻展的开放日期', '铜器展的布展日期'), ()),
    ('志愿者培训在什么地方签到，剧团演练的签到由谁负责？',
     ('志愿者培训的签到地点', '志愿者培训的签到人员', '剧团演练的签到人员', '剧团演练的签到地点'), (0, 2)),
    ('陶板画的接收人数是多少，展览册谁来签收？',
     ('陶板画的接收人员', '陶板画的作品份数', '展览册的交付人员', '展览册的签收人'), (3,)),
)


def suite(scenes, frames, unavailable):
    rows = []
    for index, (owner, first, second) in enumerate(scenes):
        other, third, fourth = scenes[(index + 1) % len(scenes)]
        fields = (first, second, third, fourth)
        owners = (owner, owner, other, other)
        candidates = [f'{name}的{field[0]}' for name, field in zip(owners, fields, strict=True)]
        for phrasing, frame in enumerate(frames, start=1):
            phrases = [f'关于{name}，{field[phrasing]}' for name, field in zip(owners, fields, strict=True)]
            for target, requested in (
                ((0, 1), (phrases[0], phrases[1])),
                ((1, 2), (phrases[1], phrases[2])),
                ((3,), (phrases[3], f'{owner}的{unavailable[phrasing - 1]}')),
                ((), (f'{owner}的{unavailable[0]}', f'{other}的{unavailable[1]}')),
            ):
                rows.append(dict(question=frame.format(first=requested[0], second=requested[1]),
                                 candidates=candidates, target=target))
    return rows


def regression_cases():
    return [dict(question=question, candidates=[f'{scene}的{field[0]}' for scene in (first, second)
                for field in SCENES[scene]], target=target, group=group)
            for group, first, second, question, target in REQUESTS]


def features(extract, rows):
    texts = [f'请求：{row["question"]}\n档案字段：{candidate}\n能否回答请求中的任一项？'
             for row in rows for candidate in row['candidates']]
    unique = sorted(set(texts))
    vectors = {}
    for start in range(0, len(unique), 16):
        batch = unique[start:start + 16]
        hidden = extract(batch)
        vectors.update((text, hidden[i, len(text) - 1].float()) for i, text in enumerate(batch))
    return torch.stack([vectors[text] for text in texts]).reshape(len(rows), 4, -1)


def target_tensor(rows):
    return torch.tensor([[i in row['target'] for i in range(len(row['candidates']))] for row in rows])


def lexical_scores(rows):
    def bigrams(text):
        return set(pairwise(text))

    return torch.tensor([[len(bigrams(row['question']) & bigrams(candidate)) / max(1, len(bigrams(candidate)))
                          for candidate in row['candidates']] for row in rows])


def fit_layer(inputs, targets, *, rank_weight=0.):
    torch.manual_seed(SEED)
    head = nn.Sequential(nn.LayerNorm(inputs.shape[-1]), nn.Linear(inputs.shape[-1], 24), nn.GELU(), nn.Linear(24, 1))
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.03)
    ranked_pairs = targets[:, :, None] & ~targets[:, None, :]
    for _ in range(160):
        logits = head(inputs).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=torch.tensor(2.2))
        if rank_weight:
            margins = (logits[:, :, None] - logits[:, None, :])[ranked_pairs]
            loss = loss + rank_weight * F.softplus(1. - margins).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return head.eval().requires_grad_(False)


def cutoff(scores, truth, *, wrong_weight=1.):
    def utility(threshold):
        chosen = scores.ge(threshold)
        correct = chosen.eq(truth).all(dim=-1).sum().item()
        erroneous = (chosen & ~truth).sum().item()
        return correct - wrong_weight * erroneous, -erroneous, threshold

    candidates = sorted(set(scores.flatten().tolist()))
    return max([*candidates, 1.01], key=utility)


def evaluate(rows, scores, threshold):
    results, totals = [], {}
    for row, row_scores in zip(rows, scores.tolist(), strict=True):
        expected = set(row['target'])
        selected = {i for i, value in enumerate(row_scores) if value >= threshold}
        separated = bool(expected and min(row_scores[i] for i in expected) > max(
            value for i, value in enumerate(row_scores) if i not in expected))
        totals.setdefault(str(len(expected)), Counter()).update(
            count=1, exact=expected == selected, wanted=len(expected), accepted=len(selected & expected),
            erroneous=len(selected - expected), contaminated=bool(selected - expected), separated=separated)
        results.append({**row, 'scores': row_scores, 'selected': sorted(selected)})
    return dict(totals=totals, rows=results)


def main(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {path: sha(path) for path in (args.checkpoint, *(folder / name for folder in args.layers
                                                            for name in ('attention.pt', 'comparison.json')))}
    torch.set_num_threads(2)
    rows = dict(train=suite(TRAIN, FRAMES[0], UNAVAILABLE[0]),
                calibration=suite(CALIBRATION, FRAMES[1], UNAVAILABLE[1]),
                held=suite(HELD, FRAMES[2], UNAVAILABLE[2]), regression=regression_cases())
    if args.ranking:
        rows['hard'] = [dict(question=q, candidates=candidates, target=target) for q, candidates, target in HARD_HELD]
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    with local_model([args.checkpoint, *args.layers]) as model:
        extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=23, single_sequence=True)
        hidden = {name: features(extract, group) for name, group in rows.items()}
    truth = {name: target_tensor(group) for name, group in rows.items()}
    channels, checkpoints = {}, {}
    arms = (('neural', 0.), ('pairwise', 1.)) if args.ranking else (('neural', 0.),)
    for name, weight in arms:
        head = fit_layer(hidden['train'], truth['train'], rank_weight=weight)
        artifact = args.output / ('relevance.pt' if name == 'neural' else f'{name}.pt')
        atomic_save(head.state_dict(), artifact)
        head.load_state_dict(torch.load(artifact, weights_only=True, map_location='cpu'), strict=True)
        checkpoints[name] = sha(artifact)
        with torch.no_grad():
            channels[name] = {split: head(values).squeeze(-1).sigmoid() for split, values in hidden.items()}
    channels['lexical'] = {name: lexical_scores(group) for name, group in rows.items()}
    results = {}
    for name, scores in channels.items():
        threshold = cutoff(scores['calibration'], truth['calibration'])
        metrics = {split: evaluate(group, scores[split], threshold) for split, group in rows.items() if split != 'train'}
        results[name] = dict(threshold=threshold, splits=metrics)
        print(name, threshold, {split: data['totals'] for split, data in metrics.items()}, flush=True)
        if args.ranking:
            relaxed = cutoff(scores['calibration'], truth['calibration'], wrong_weight=.25)
            metrics = {split: evaluate(group, scores[split], relaxed)['totals']
                       for split, group in rows.items() if split != 'train'}
            results[name]['relaxed'] = dict(threshold=relaxed, splits=metrics)
            print(name, 'relaxed', relaxed, metrics, flush=True)
    if any(sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('Source checkpoints changed during the relevance comparison')
    store_json(args.output / 'result.json', dict(results=results, seconds=perf_counter() - started,
               training_requests=len(rows['train']), checkpoint_sha256=checkpoints['neural'], checkpoints=checkpoints,
               sources={str(path): digest for path, digest in sources.items()}))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('layers', type=Path, nargs=2)
    cli.add_argument('--ranking', action='store_true', help='Compare an added within-request ranking objective')
    main(cli.parse_args())
