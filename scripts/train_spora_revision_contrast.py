"""Train a correction-version residual while retaining SPORA's read-intent gate."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer

from scripts.audit_qwen_compatible_natural_memory_20261121 import (
    NAMES, controlled, fresh_version_cases, load_head, score,
)
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_temporal_contrast_cases import ABSTAIN, temporal_rows
from scripts.train_qwen_temporal_token_gate_20261122 import evaluate, suites, threshold_for
from spora.integrated_checkpoint import load_integrated_runtime
from spora.language_training import atomic_save, frozen_digest
from spora.research_temporal_intent import question_mean, temporal_character_features, time_selection


TRAIN_TRIPLES = (
    ('更正不是问的重点，眼下给{name}记着什么{field}？',
     '更正还没有发生时，给{name}记着什么{field}？', '更正发生之后，给{name}新记了什么{field}？'),
    ('不要之前那个，以仍在用的{name}的{field}为准。是哪一项？',
     '不要更新出的那个，以先前{name}的{field}为准。是哪一项？',
     '不要先前那个，以刚更新出的{name}的{field}为准。是哪一项？'),
    ('旧条目已经换掉了，{name}的{field}现在填的啥？',
     '旧条目还没换掉的那会儿，{name}的{field}填的啥？',
     '刚刚换掉旧条目那会儿，{name}的{field}改成的啥？'),
    ('我用得着{name}正在采用的{field}，能查一下吗？',
     '我用得着{name}最初采用的{field}，能查一下吗？',
     '我用得着{name}一改完才采用的{field}，能查一下吗？'),
    ('{name}这会儿用什么{field}？我知道之前换过。',
     '{name}起先用什么{field}？我知道之后换过。',
     '{name}那次换完就用什么{field}？我只问那一版。'),
    ('笔记里的{name}现在用哪个{field}？改前的不算。',
     '笔记里的{name}没更改前用哪个{field}？改后的不算。',
     '笔记里的{name}更改完用上哪个{field}？改前的不算。'),
    ('现行清单里{name}的{field}是哪一个？请查给我。',
     '第一次留底的清单里{name}的{field}是哪一个？请查给我。',
     '换成另一版的那张清单里{name}的{field}是哪一个？请查给我。'),
    ('就说{name}目前登记的{field}，不追究那回更正。',
     '回到{name}还未更正的时候，就说那时登记的{field}。',
     '回到{name}刚更正完的时候，就说那时登记的{field}。'),
)
CAL_TRIPLES = (
    ('查{name}执行中的{field}。', '查{name}尚未替换那页的{field}。', '查{name}替换成的新页里的{field}。'),
    ('旧单收起来，只念{name}手头正在用的{field}。',
     '新单收起来，只念{name}改之前还在用的{field}。',
     '旧单收起来，只念{name}刚改过来的{field}。'),
)
CAL_NEGATIVES = (
    '只把{name}这条{field}记录的标题改写一下。',
    '我想给{name}填一个没用过的{field}，请推荐。',
    '查一下{name}哪一天动过{field}这一栏，只给日期。',
    '不要念出{name}的任何{field}记录，帮我写个通知。',
)
HELD_TRIPLES = (
    ('我今天替{name}办事，现行记录的{field}写的什么？',
     '我在给{name}核最老的一份档案，第一回记的{field}写的什么？',
     '我在给{name}核那次更新，当时换进去的{field}写的什么？'),
    ('给{name}办的时候照仍在执行的清单，{field}一项是什么？',
     '退回到修改还没动笔那一页，{name}的{field}是什么？',
     '翻到落笔换上去的那一页，{name}的{field}是什么？'),
    ('{name}前面改过，我要今天有效的{field}是哪样。',
     '{name}还没来得及改的那份，我要当初的{field}是哪样。',
     '{name}把那份改好了以后，我要那次新增的{field}是哪样。'),
    ('{name}的草稿可以跳过，按手里在执行的版本说{field}。',
     '{name}的定稿可以跳过，按动笔重写以前的版本说{field}。',
     '{name}的草稿可以跳过，按刚重写成的版本说{field}。'),
)
HELD_NEGATIVES = (
    '先封存{name}的旧记录，只介绍一下{field}是什么。',
    '把「{name}最初选了什么{field}」写成英文，原值不用告诉我。',
    '我在修订{name}的{field}，请给我一个全新的建议。',
    '只说明{name}那次更改是哪天，别列出改前改后的{field}。',
)
SEED = 221829


def held_questions():
    questions = []
    for index, name in enumerate(NAMES):
        for text, mode in temporal_rows((name,), ('饮品',), HELD_TRIPLES, HELD_NEGATIVES):
            questions.append((text, index, mode) if mode != ABSTAIN else (text, None, None))
    return questions


def build_data():
    prior = suites()
    new_train = temporal_rows(
        ('曹霏', '温霁', '于舟', '阮棠', '夏平', '姚杉', '程晋', '宁远'),
        ('发货地址', '筛选条件', '背景颜色', '发票抬头', '饮料', '预约地点', '印章样式', '点心'),
        TRAIN_TRIPLES, (),
    )
    new_calibration = temporal_rows(('姜遇', '齐砚', '严琛', '何问'), ('取餐地址', '联系人'),
                                    CAL_TRIPLES, CAL_NEGATIVES)
    distinct = [(text, ABSTAIN if mode is None else mode) for text, _, mode in held_questions()]
    data = {'train': [row for row in prior['train'] if row[1] != ABSTAIN] + new_train,
            'calibration': prior['calibration'] + new_calibration,
            'fresh_v1': prior['fresh_v1'], 'fresh_v2': prior['fresh_v2'], 'new_holdout': distinct}
    texts = [text for rows in data.values() for text, _ in rows]
    if len(set(texts)) != len(texts):
        raise ValueError('Temporal data splits contain repeated sentences')
    return data


def fit(inputs, old_version, labels):
    model = nn.Linear(inputs.shape[-1], 3, bias=False)
    nn.init.zeros_(model.weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0006, weight_decay=.06)
    generator = torch.Generator().manual_seed(SEED)
    for _ in range(12):
        for batch in torch.randperm(len(labels), generator=generator).split(24):
            delta = model(inputs[batch])
            loss = F.cross_entropy(old_version[batch] + delta, labels[batch]) + .004 * delta.square().mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite temporal residual loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.weight.detach()


def choose_settings(old_version, read_mass, delta, labels, baseline_threshold):
    options = []
    for strength in (.25, .5, .75, 1., 1.25):
        versions = (old_version + strength * delta).softmax(-1)
        probabilities = torch.cat((versions * read_mass, 1. - read_mass), dim=-1)
        threshold = max(baseline_threshold, threshold_for(probabilities, labels, factorized=True))
        predicted = time_selection(probabilities, threshold, factorized=True)
        positive = labels != ABSTAIN
        past = labels == 1
        negative = ~positive
        rank = (int((predicted[negative] == ABSTAIN).sum()), int((predicted[positive] == labels[positive]).sum()),
                int((predicted[past] == 1).sum()), -strength)
        options.append((rank, strength, threshold))
    return max(options)


def fit_candidates(decoder, tokenizer, old):
    data = build_data()
    boxes = {}
    questions = []
    for name, rows in data.items():
        boxes[name] = slice(len(questions), len(questions) + len(rows))
        questions.extend(text for text, _ in rows)
    vectors = []
    for offset in range(0, len(questions), 80):
        features, masks = extract(decoder.base, tokenizer, questions[offset:offset + 80], batch_size=2,
                                  hidden_indices=tuple(old['hidden_indices']), last_logit_only=True)
        vectors.append(question_mean(features, masks))
    vectors = torch.cat(vectors)
    labels = torch.tensor([mode for rows in data.values() for _, mode in rows])
    old_version = F.linear(vectors, old['state']['version.weight'], old['state']['version.bias'])
    read_mass = F.linear(vectors, old['state']['read.weight'], old['state']['read.bias']).sigmoid()
    baseline_probs = torch.cat((old_version.softmax(-1) * read_mass, 1. - read_mass), dim=-1)
    models = [('original', None, old['threshold'], baseline_probs, None)]
    for name, char_width in (('semantic', 0), ('semantic_character', 1024)):
        inputs = vectors
        if char_width:
            inputs = torch.cat((vectors, temporal_character_features(questions, char_width)), dim=-1)
        training, cal = boxes['train'], boxes['calibration']
        weight = fit(inputs[training], old_version[training], labels[training])
        delta = inputs @ weight.T
        rank, strength, threshold = choose_settings(old_version[cal], read_mass[cal], delta[cal], labels[cal], old['threshold'])
        probs = torch.cat(((old_version + strength * delta).softmax(-1) * read_mass, 1. - read_mass), dim=-1)
        models.append((name, {'char_width': char_width, 'weight': weight, 'strength': strength}, threshold, probs, rank))
    results = {}
    for name, _, threshold, probs, rank in models:
        evaluated = {split: evaluate(rows, probs[boxes[split]], threshold, factorized=True)
                     for split, rows in data.items() if split != 'train'}
        results[name] = {'threshold': threshold, 'calibration_rank': rank, 'results': evaluated}
        print(name, {key: trial['by_label'] for key, trial in evaluated.items()}, flush=True)
    winner = max(models[1:], key=lambda row: row[-1])
    return dict(old, version_residual=winner[1], threshold=winner[2]), {
        'selected': winner[0], 'counts': {name: len(rows) for name, rows in data.items()},
        'classification': results,
    }


def natural(runtime, tokenizer, router, model, questions):
    states = [controlled(runtime.organism, offset) for offset in (0, 2)]
    digests = [frozen_digest(runtime.organism, state) for state in states]
    rows, _ = score(runtime.decoder, tokenizer, runtime.organism, router, model, states,
                    evaluation_questions=questions)
    if digests != [frozen_digest(runtime.organism, state) for state in states]:
        raise ValueError('Temporal evaluation changed a controlled memory state')
    known = [row for row in rows if row['expected_key'] is not None]
    unsupported = [row for row in rows if row['expected_key'] is None]
    return {'summary': {'known': len(known), 'time_correct': sum(r['mode'] == r['expected_mode'] for r in known),
                        'correct_pairs': sum(r['correct_pair'] for r in known),
                        'past_pairs': sum(r['expected_mode'] == 1 and r['correct_pair'] for r in known),
                        'unsupported': len(unsupported), 'unsupported_routed': sum(r['key'] is not None for r in unsupported)},
            'rows': rows}


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    original_sha = sha(args.checkpoint)
    route_source_sha = sha(args.route_source)
    old, old_sha = load_head(args.temporal, 'temporal.pt', 'temporal_sha256', route_source_sha)
    router, router_sha = load_head(args.router, 'router.pt', 'router_sha256', route_source_sha)
    if old.get('architecture') != 'factorized' or old['hidden_indices'] != router['hidden_indices']:
        raise ValueError('Use paired factorized temporal and identity heads')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(args.checkpoint, load_full_streamed(), MODEL)
    runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    candidate, report = fit_candidates(runtime.decoder, tokenizer, old)
    candidate['checkpoint_sha256'] = original_sha
    if any(not torch.equal(value, candidate['state'][key]) for key, value in old['state'].items()):
        raise ValueError('The original temporal weights changed')
    initial = args.output / 'classification.json'
    args.output.mkdir(parents=True, exist_ok=False)
    initial.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    natural_results = {}
    for name, rows in (('fresh_v2', fresh_version_cases()), ('new_holdout', held_questions())):
        natural_results[name] = {}
        for label, model in (('original', old), ('candidate', candidate)):
            trial = natural(runtime, tokenizer, router, model, rows)
            natural_results[name][label] = trial
            print('End to end', name, label, trial['summary'], flush=True)
    if frozen_digest(runtime.organism, runtime.state) != individual or sha(args.checkpoint) != original_sha:
        raise ValueError('An original model or individual changed during evaluation')
    weights = args.output / 'temporal.pt'
    atomic_save(candidate, weights)
    report.update({'checkpoint_sha256': original_sha, 'baseline_temporal_sha256': old_sha, 'router_sha256': router_sha,
                   'temporal_sha256': sha(weights), 'natural': natural_results,
                   'seconds': time.perf_counter() - started})
    (args.output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'route-source', 'router', 'temporal', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
