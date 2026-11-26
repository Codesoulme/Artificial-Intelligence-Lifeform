"""Learn which subset of available addresses answers a request."""

import argparse
from itertools import combinations, product
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from scripts.spora_version_chat import FrozenTurnFeatures, base, local_model
from scripts.train_spora_candidate_relevance import (
    CALIBRATION, FRAMES, HARD_HELD, HELD, SEED, TRAIN, UNAVAILABLE, cutoff, evaluate, features,
    fit_layer, regression_cases, suite, target_tensor,
)
from scripts.train_spora_recurrent_counterfactual import sha, store_json
from spora.language_training import atomic_save


MATCHED = (
    (
        ('木工课', '编织课', '几点开课', '开课时刻', '下课时刻', '在哪报名', '报名地点', '上课地点'),
        ('展架组', '舞台组', '谁来收工具', '收工具的人', '发工具的人', '一共放多少把椅子', '椅子数量', '桌子数量'),
        ('煤油灯展', '彩纸展', '作品怎么称呼', '作品名', '展厅名', '到哪还展品', '展品归还处', '展品领取处'),
    ),
    (
        ('雕版展', '篆刻展', '几号撤展', '撤展日期', '开幕日期', '哪里存放空箱', '空箱存放处', '作品存放处'),
        ('纸袋订单', '麻布订单', '谁核收货物', '收货人', '打包人', '总共要送几箱', '发货箱数', '退货箱数'),
    ),
    (
        ('壁画修补', '砖雕修补', '几点动工', '动工时间', '验收时间', '谁检查成果', '验收人', '领料人'),
        ('庭院展演', '露台展演', '演员到哪处候场', '演员候场地点', '观众等候地点', '哪日预演', '预演日期', '正式演出日期'),
        ('青年画刊', '乡土画刊', '领回来几册', '实领册数', '印制册数', '谁替我领', '代领人', '发放人'),
    ),
)
REQUESTS = (
    '核对一下{first}，也查一下{second}。',
    '交接单请填这两处：{first}。{second}。',
    '替我确认两件事。{second}？另外，{first}？',
)
SUBSETS = ((), *combinations(range(4), 1), *combinations(range(4), 2))

FRESH_MATCHED = (
    ('温室修整', '露圃养护', '哪天换玻璃', '换玻璃日期', '清扫日期', '谁检查新苗', '新苗检查员', '旧苗修剪员'),
    ('街巷画展', '河畔影展', '哪里挂导览图', '导览图悬挂处', '检票处', '收回几台放映机', '放映机归还数量', '音箱归还数量'),
    ('面包试做', '饼干试做', '由谁负责醒面', '醒面负责人', '揉面负责人', '样品送到哪间房', '样品送达处', '原料存放处'),
)


def matched_rows(split, *, fresh=False, annotate=False):
    rows = []
    families = FRESH_MATCHED if fresh else MATCHED[split]
    for index, (owner, other, first, field, near, second, other_field, other_near) in enumerate(families):
        frame = '我需要核实{first}，以及{second}。' if fresh else REQUESTS[split]
        parts = f'{owner}{first}', f'{other}{second}'
        question = frame.format(first=parts[0], second=parts[1])
        for has_first, has_second in product((False, True), repeat=2):
            # Mirrored substitutions preserve both scene and field vocabulary across each family.
            candidates = ((owner, field if has_first else near), (other, near if has_first else field),
                          (other, other_field if has_second else other_near),
                          (owner, other_near if has_second else other_field))
            order = [(index + offset) % len(candidates) for offset in range(len(candidates))]
            target = tuple(i for i, position in enumerate(order)
                           if (position == 0 and has_first) or (position == 2 and has_second))
            row = dict(question=question, candidates=[f'{candidates[i][0]}的{candidates[i][1]}' for i in order],
                       target=target)
            if annotate:
                part_targets = (order.index(0) if has_first else None, order.index(2) if has_second else None)
                row['parts'] = sorted(zip(parts, part_targets, strict=True), key=lambda pair: question.index(pair[0]))
            rows.append(row)
    return rows


def deletion_pairs(rows):
    by_question = {}
    for index, row in enumerate(rows):
        by_question.setdefault(row['question'], []).append(index)
    edges = []
    for family in by_question.values():
        for first, second in combinations(family, 2):
            small, large = sorted((first, second), key=lambda i: len(rows[i]['target']))
            before, after = rows[small], rows[large]
            kept, present = set(before['target']), set(after['target'])
            if kept < present and len(present) == len(kept) + 1:
                if any(before['candidates'][i] != after['candidates'][i] for i in kept):
                    raise ValueError('A retained target was replaced in a deletion pair')
                edges.append((small, large, next(iter(present - kept)), tuple(sorted(kept))))
    return edges


class AddressSets(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.encode = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 24), nn.GELU())
        self.field = nn.Linear(24, 1, bias=False)
        self.count = nn.Linear(48, 3)
        self.register_buffer('choices', target_tensor([dict(candidates=range(4), target=subset) for subset in SUBSETS]))

    def forward(self, inputs):
        candidates = self.encode(inputs)
        totals = self.count(torch.cat((candidates.mean(dim=-2), candidates.amax(dim=-2)), dim=-1))
        fields = self.field(candidates).squeeze(-1)
        logits = fields @ self.choices.to(fields.dtype).T + totals[:, self.choices.sum(-1)]
        return logits, fields


def fit_sets(inputs, targets, *, auxiliary, pairs=()):
    torch.manual_seed(SEED)
    head = AddressSets(inputs.shape[-1])
    matching = targets[:, None, :].eq(head.choices).all(dim=-1)
    if not matching.sum(dim=-1).eq(1).all():
        raise ValueError('Training target is not one supported address set')
    labels = matching.long().argmax(dim=-1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.03)
    if pairs:
        smaller, larger, changed = (torch.tensor(values) for values in zip(*(pair[:3] for pair in pairs), strict=True))
        stable = targets[smaller]
        choices = head.choices.float()
    for _ in range(160):
        logits, fields = head(inputs)
        loss = F.cross_entropy(logits, labels)
        if auxiliary:
            loss = loss + .5 * F.binary_cross_entropy_with_logits(fields, targets.float(), pos_weight=torch.tensor(2.2))
        if pairs:
            inclusion = logits.softmax(dim=-1) @ choices
            change = inclusion[larger] - inclusion[smaller]
            count_gap = change.sum(dim=-1)
            field_gap = change[torch.arange(len(pairs)), changed]
            stability = change[stable].square().mean()
            loss = loss + .5 * ((1. - count_gap).square().mean()
                                + (1. - field_gap).square().mean() + stability)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return head.eval().requires_grad_(False)


def summarize(rows, fields, selected):
    report = evaluate(rows, fields, 2.)
    counts = {n: dict(count_correct=0, exact=0, accepted=0, erroneous=0, contaminated=0)
              for n in report['totals']}
    for row, chosen in zip(report['rows'], selected.tolist(), strict=True):
        actual = {i for i, flag in enumerate(chosen) if flag}
        target = set(row['target'])
        result = counts[str(len(target))]
        result['count_correct'] += len(actual) == len(target)
        result['exact'] += actual == target
        result['accepted'] += len(actual & target)
        result['erroneous'] += len(actual - target)
        result['contaminated'] += bool(actual - target)
        row['selected'] = sorted(actual)
    for n, count in counts.items():
        report['totals'][n] = dict(report['totals'][n]) | count
    return report


def deletion_summary(rows):
    pairs = deletion_pairs(rows)
    tally = dict(total=len(pairs), count_delta=0, removed=0, retained=0, exact_transition=0)
    for smaller, larger, changed, kept in pairs:
        before, after = rows[smaller], rows[larger]
        reduced, full = set(before['selected']), set(after['selected'])
        tally['count_delta'] += len(full) - len(reduced) == 1
        tally['removed'] += changed in full and changed not in reduced
        tally['retained'] += set(kept) <= reduced & full
        tally['exact_transition'] += reduced == set(before['target']) and full == set(after['target'])
    return tally


def main(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {path: sha(path) for path in (args.checkpoint, *(layer / name for layer in args.layers
                                                            for name in ('attention.pt', 'comparison.json')))}
    torch.set_num_threads(2)
    rows = dict(train=suite(TRAIN, FRAMES[0], UNAVAILABLE[0]) + matched_rows(0),
                calibration=suite(CALIBRATION, FRAMES[1], UNAVAILABLE[1]) + matched_rows(1),
                held=suite(HELD, FRAMES[2], UNAVAILABLE[2]), regression=regression_cases(), matched=matched_rows(2),
                hard=[dict(question=q, candidates=candidates, target=target) for q, candidates, target in HARD_HELD])
    if args.paired:
        rows['fresh_matched'] = matched_rows(2, fresh=True)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    with local_model([args.checkpoint, *args.layers]) as model:
        extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=23, single_sequence=True)
        hidden = {name: features(extract, items) for name, items in rows.items()}
    training_rows = rows.pop('train')
    training, truth = hidden.pop('train'), target_tensor(training_rows)
    for parameters in hidden.values():
        if not parameters.isfinite().all():
            raise ValueError('A held feature vector is not finite')
    heads = {'binary': fit_layer(training, truth, rank_weight=1.),
             'sets': fit_sets(training, truth, auxiliary=False),
             'sets_aux': fit_sets(training, truth, auxiliary=True)}
    if args.paired:
        pairs = deletion_pairs(training_rows)
        if len(pairs) != 4 * len(MATCHED[0]):
            raise ValueError('Unexpected number of matched training interventions')
        heads['sets_paired'] = fit_sets(training, truth, auxiliary=True, pairs=pairs)
    results, checkpoints = {}, {}
    for name, head in heads.items():
        artifact = args.output / f'{name}.pt'
        atomic_save(head.state_dict(), artifact)
        head.load_state_dict(torch.load(artifact, weights_only=True, map_location='cpu'), strict=True)
        checkpoints[name] = sha(artifact)
        if name == 'binary':
            scores = {split: head(data).squeeze(-1).sigmoid() for split, data in hidden.items()}
            calibration = target_tensor(rows['calibration'])
            threshold = cutoff(scores['calibration'], calibration)
            relaxed = cutoff(scores['calibration'], calibration, wrong_weight=.25)
            results[name] = dict(threshold=threshold, relaxed=relaxed,
                splits={split: summarize(rows[split], score, score.ge(threshold)) for split, score in scores.items()},
                relaxed_splits={split: summarize(rows[split], score, score.ge(relaxed)) for split, score in scores.items()})
        else:
            splits = {}
            for split, data in hidden.items():
                logits, fields = head(data)
                selected = head.choices[logits.argmax(dim=-1)]
                reversed_logits, _ = head(data.flip(dims=(1,)))
                reversed_selected = head.choices[reversed_logits.argmax(dim=-1)].flip(dims=(1,))
                if not torch.equal(selected, reversed_selected):
                    raise RuntimeError('Reversing the candidates changed the predicted addresses')
                splits[split] = summarize(rows[split], fields, selected)
                for row, scores in zip(splits[split]['rows'], logits.tolist(), strict=True):
                    row['subset_logits'] = scores
            results[name] = dict(splits=splits)
        if args.paired:
            results[name]['deletions'] = {split: deletion_summary(results[name]['splits'][split]['rows'])
                                          for split in ('matched', 'fresh_matched')}
            print(name, 'deletions', results[name]['deletions'], flush=True)
        print(name, {split: data['totals'] for split, data in results[name]['splits'].items()}, flush=True)
    if any(sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('Source checkpoint changed')
    store_json(args.output / 'result.json', dict(results=results, checkpoints=checkpoints, subsets=SUBSETS,
               training_requests=len(training), sources={str(path): digest for path, digest in sources.items()},
               seconds=perf_counter() - started))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('layers', type=Path, nargs=2)
    cli.add_argument('--paired', action='store_true', help='Evaluate same-request deletion supervision')
    main(cli.parse_args())
