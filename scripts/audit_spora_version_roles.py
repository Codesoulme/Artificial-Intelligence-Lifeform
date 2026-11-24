"""Compare clause roles and whole-question labels for SPORA version selection."""

import argparse
from collections import Counter
from datetime import date
import gc
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_fast_weight_lexical_write import base
from scripts.audit_spora_version_scope import verify_answers
from scripts.audit_spora_version_timeline import learned_routes, SUBJECTS, TODAY
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.identity_fingerprints import IdentityFingerprints
from spora.language_training import atomic_save
from spora.research_version_roles import (ABSTAIN, CURRENT, DENY, EXCLUDE, EXPIRED, FIRST, NEUTRAL, PREVIOUS, SEEK,
                                          UNSPECIFIED, VersionRoleHead, apply_version_plan, clause_spans, version_plan)
from spora.research_version_scope import addressed_subject
from spora.research_version_timeline import effective_order
from spora.research_versioned_memory_pointer import VersionReplyActions


SEED = 26344
TRAIN_NAMES = (('唐心雅', '饮品'), ('董纪文', '见面地点'), ('姚恩平', '物品'), ('钟希扬', '书目'))
CAL_NAMES = (('吕白星', '书目'), ('苏芷悦', '饮品'))
TRAIN = (
    ('拿{v}来答', '把{v}划掉', ('现在生效的版本', '首次登记的原稿', '紧挨现行前的一版')),
    ('我想知道{v}', '{v}不在本次范围', ('当下那一版', '最初有效的那份', '上次仍在用的那份')),
    ('请采用{v}', '省略{v}', ('依然有效的记录', '第一次填报的项目', '前一份记录')),
    ('按{v}回忆', '{v}先搁在一边', ('目前在用的项目', '开头写过的那份', '被最后一次更新替下的那份')),
)
CAL = (
    ('请核实{v}', '这次排除{v}', ('眼下有效的这版', '第一回抄下来的那版', '现行的前一份')),
    ('以{v}为准', '跳过{v}', ('手头执行的内容', '最前面的稿子', '最新改动之前的稿子')),
)
HELD = (
    ('{v}是我想对照的', '至于{v}先放下', ('眼下还管用的那一页', '起笔时候那张底稿', '紧挨在用那页之前的那张')),
    ('我这一问针对{v}', '{v}从回答里划去', ('此时执行的那份', '刚开档案时写下的内容', '换成现用前还作数的那张')),
)


def make_question(clauses, target, pair=None):
    text = '。'.join(part for part, _, _ in clauses) + '？'
    return {'question': text, 'roles': [role for _, role, _ in clauses],
            'versions': [version for _, _, version in clauses], 'target': target, 'pair': pair}


def contrast_examples(names, families):
    rows = []
    for name, field in names:
        prompt = (f'{name}的{field}登记的是哪一项', SEEK, UNSPECIFIED)
        for family, (seek, exclude, cues) in enumerate(families):
            for left, right in ((CURRENT, FIRST), (CURRENT, PREVIOUS), (FIRST, PREVIOUS)):
                key = f'{name}-{family}-{left}-{right}'
                for requested, excluded in ((left, right), (right, left)):
                    parts = [(seek.format(v=cues[requested]), SEEK, requested),
                             (exclude.format(v=cues[excluded]), EXCLUDE, excluded)]
                    if family % 2:
                        parts.reverse()
                    rows.append(make_question([*parts, prompt], requested, key))
            for version, cue in enumerate(cues):
                rows.append(make_question([(seek.format(v=cue), SEEK, version), prompt], version))
            rows.append(make_question([(exclude.format(v=cues[CURRENT]), EXCLUDE, CURRENT), prompt], ABSTAIN))
            rows.append(make_question([(seek.format(v=cues[CURRENT]), SEEK, CURRENT),
                                       (exclude.format(v=cues[CURRENT]), EXCLUDE, CURRENT), prompt], ABSTAIN))
            rows.append(make_question([(exclude.format(v='已经作废的全部版本'), EXCLUDE, EXPIRED), prompt], CURRENT))
            rows.append(make_question([('请勿开启事实读取', DENY, UNSPECIFIED), prompt], ABSTAIN))
            rows.append(make_question([(f'请为{name}拟一张{field}空表', NEUTRAL, UNSPECIFIED)], ABSTAIN))
    return list({row['question']: row for row in rows}.values())


def held_examples():
    rows = contrast_examples([(name, field) for _, name, field, _, _ in SUBJECTS], HELD)
    # Separate challenge families for polarity with implicit current and earliest.
    for _, name, field, _, _ in SUBJECTS:
        request = (f'{name}的{field}写了啥', SEEK, UNSPECIFIED)
        rows.append(make_question([('那些作废的先收出去', EXCLUDE, EXPIRED), request], CURRENT))
        rows.append(make_question([('问的是开头落笔时的那份', SEEK, FIRST), request], FIRST))
        rows.append(make_question([('现下这页先收出去', EXCLUDE, CURRENT), request], ABSTAIN))
    return rows


@torch.no_grad()
def clause_features(model, tokenizer, questions, *, tail_chars=None):
    extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=12)
    all_spans = [clause_spans(question) for question in questions]
    width = max(map(len, all_spans))
    vectors = torch.zeros(len(questions), width, model.config.hidden_size)
    mask = torch.zeros(len(questions), width, dtype=torch.bool)
    tail = torch.zeros_like(vectors) if tail_chars is not None else None
    for offset in range(0, len(questions), 20):
        encoded = extract(questions[offset:offset + 20]).float()
        for index, spans in enumerate(all_spans[offset:offset + len(encoded)]):
            for pos, (start, end) in enumerate(spans):
                vectors[offset + index, pos] = encoded[index, start:end].mean(0)
                mask[offset + index, pos] = True
                if tail is not None:
                    tail[offset + index, pos] = encoded[index, max(start, end - tail_chars):end].mean(0)
    return (vectors, mask) if tail is None else (vectors, mask, tail)


def flat_features(vectors, mask):
    pooled = (vectors * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
    return F.layer_norm(pooled, (pooled.shape[-1],))


def make_heads(width):
    return {'flat': nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 4)),
            'roles': VersionRoleHead(width)}


def fit(heads, features, mask, rows, steps, *, role_features=None):
    shape = features.shape[:2]
    role_labels = torch.full(shape, -100, dtype=torch.long)
    version_labels = role_labels.clone()
    flat_labels = torch.tensor([row['target'] for row in rows])
    for index, row in enumerate(rows):
        count = int(mask[index].sum())
        if count != len(row['roles']):
            raise ValueError('Clause boundaries disagree with role supervision')
        role_labels[index, :count] = torch.tensor(row['roles'])
        version_labels[index, :count] = torch.tensor(row['versions'])
    flat = flat_features(features, mask)
    for mode, head in heads.items():
        optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
        random = torch.Generator().manual_seed(SEED)
        for step in range(steps):
            batch = torch.randperm(len(rows), generator=random)[:32]
            if mode == 'flat':
                loss = F.cross_entropy(head(flat[batch]), flat_labels[batch])
            else:
                evidence = None if role_features is None else role_features[batch]
                roles, versions = head(features[batch], evidence)
                loss = F.cross_entropy(roles.flatten(0, 1), role_labels[batch].flatten())
                loss += F.cross_entropy(versions.flatten(0, 1), version_labels[batch].flatten())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step in (0, steps - 1):
                print('trained', mode, step + 1, float(loss.detach()), flush=True)


@torch.inference_mode()
def role_predictions(head, vectors, mask, role_features=None):
    role_logits, version_logits = head(vectors, role_features)
    role_scores, role_ids = role_logits.softmax(-1).max(-1)
    version_scores, version_ids = version_logits.softmax(-1).max(-1)
    proposed = []
    for index in range(len(vectors)):
        live = mask[index]
        roles, versions = role_ids[index, live].tolist(), version_ids[index, live].tolist()
        scores = [float(score) for role, score in zip(roles, role_scores[index, live], strict=True) if role != NEUTRAL]
        scores.extend(float(score) for role, v, score in zip(roles, versions, version_scores[index, live], strict=True)
                      if role in (SEEK, EXCLUDE) and v != UNSPECIFIED)
        proposed.append({'plan': version_plan(roles, versions), 'confidence': min(scores, default=0.),
                         'roles': roles, 'versions': versions})
    return proposed


@torch.inference_mode()
def predictions(heads, vectors, mask):
    flat_scores, flat_ids = heads['flat'](flat_features(vectors, mask)).softmax(-1).max(-1)
    return {'flat': [{'plan': int(value), 'confidence': float(score)} for value, score in zip(flat_ids, flat_scores)],
            'roles': role_predictions(heads['roles'], vectors, mask)}


def compatible_plan(prediction, threshold):
    return prediction['plan'] if prediction['confidence'] >= threshold else None


def calibrate(proposals, rows):
    def score(mode, threshold):
        pair = [(row['target'], compatible_plan(prediction, threshold))
                for row, prediction in zip(rows, proposals[mode], strict=True)]
        wrong_reads = sum(gold == ABSTAIN and predicted not in (ABSTAIN, None) for gold, predicted in pair)
        correct_reads = sum(gold != ABSTAIN and predicted == gold for gold, predicted in pair)
        return -wrong_reads, correct_reads, threshold
    return {mode: max((.40, .50, .60, .70, .80, .90), key=lambda t: score(mode, t)) for mode in proposals}


def scoped_row(row, prediction, thresholds, fingerprints, fields, schedules):
    question, label = row['question'], row['legacy_label']
    subject = addressed_subject(question, fingerprints, fields)
    modes, reasons = {}, {}
    for mode in ('baseline', 'flat', 'roles'):
        if subject is None:
            modes[mode], reasons[mode] = None, 'unresolved_address'
            continue
        plan = compatible_plan(prediction[mode], thresholds[mode]) if mode != 'baseline' else None
        picked, reason = apply_version_plan(question, schedules[subject], TODAY, label, plan)
        modes[mode], reasons[mode] = (None if picked is None else [subject, picked]), reason
    return {**row, 'predictions': prediction, 'routes': modes, 'reasons': reasons}


def summarize(rows):
    result = {}
    for group in ('fresh', 'regression'):
        selected = [row for row in rows if row['split'] == group]
        pairs = {row['pair'] for row in selected if row.get('pair') is not None}
        result[group] = {}
        for mode in ('baseline', 'flat', 'roles'):
            counts = Counter(total=len(selected), pairs=len(pairs))
            for row in selected:
                gold, choice = row['gold'], row['routes'][mode]
                counts.update({'correct': choice == gold, 'positive': gold is not None,
                               'negative': gold is None, 'wrong_read': choice is not None and choice != gold,
                               'negative_read': gold is None and choice is not None,
                               'missed_read': gold is not None and choice is None})
            counts['paired_correct'] = sum(all(row['routes'][mode] == row['gold'] for row in selected if row.get('pair') == pair)
                                             for pair in pairs)
            result[group][mode] = counts
    return result


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    weights_path, result_path = args.output / 'heads.pt', args.output / 'result.json'
    prior = json.loads(result_path.read_text(encoding='utf-8')) if args.verify else None
    scope = json.loads(args.scope.read_text(encoding='utf-8'))
    bank_path, reader_path = args.timeline / 'versions.pt', args.reader / 'heads.pt'
    protected = {path: base.sha(Path(path)) for path in scope['sources']}
    if any(path not in protected for path in (str(bank_path), str(reader_path))) or protected != scope['sources']:
        raise ValueError('Scope artifacts differ from their recorded inputs')
    protected[str(args.scope)] = base.sha(args.scope)
    if prior is not None and (prior['sources'] != protected or prior['heads_sha256'] != base.sha(weights_path)):
        raise ValueError('Replay sources changed')
    before = {row['question']: row for row in scope['rows']}
    fixtures = held_examples()
    train = list({row['question']: row for row in contrast_examples(TRAIN_NAMES, TRAIN)}.values())
    cal = list({row['question']: row for row in contrast_examples(CAL_NAMES, CAL)}.values())
    corpora = [set(map(lambda row: row['question'], collection)) for collection in (train, cal, fixtures)]
    if any(left & right for index, left in enumerate(corpora) for right in corpora[index + 1:]) or corpora[-1] & set(before):
        raise ValueError('Held-out question overlaps the training or regression data')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    former = learned_routes(list(dict.fromkeys(row['question'] for row in fixtures)), tokenizer, protected)
    gc.collect()
    torch.cuda.empty_cache()
    print('Finished old temporal classifier', flush=True)
    model = load_full_streamed()
    heads = make_heads(model.config.hidden_size)
    if args.verify:
        saved = torch.load(weights_path, map_location='cpu', weights_only=True)
        for name, head in heads.items():
            head.load_state_dict(saved['heads'][name])
        thresholds = saved['thresholds']
    else:
        fit(heads, *clause_features(model, tokenizer, [row['question'] for row in train]), train, args.steps)
        proposals = predictions(heads, *clause_features(model, tokenizer, [row['question'] for row in cal]))
        thresholds = calibrate(proposals, cal)
        atomic_save({'heads': {name: head.state_dict() for name, head in heads.items()}, 'thresholds': thresholds}, weights_path)
    print('Thresholds', thresholds, flush=True)
    bank = torch.load(bank_path, map_location='cpu', weights_only=True)
    fingerprints = IdentityFingerprints(len(SUBJECTS))
    fields, schedules, found = {}, {}, {}
    for index, (_, name, field, _, _) in enumerate(SUBJECTS):
        fingerprints.bind(index, name)
        fields[index] = (field,)
        schedules[index] = tuple((date.fromisoformat(e), date.fromisoformat(o)) for e, o in bank['revisions'][name])
        found[name] = index
    cases = []
    for row in fixtures:
        # Recover the gold subject from the authored terminal request, independently of the address rule.
        subjects = [index for name, index in found.items() if name in row['question']]
        if len(subjects) != 1:
            raise ValueError('A contrast must contain one unambiguous fixture subject')
        subject = subjects[0]
        order = effective_order(schedules[subject], as_of=TODAY, known_by=TODAY)
        slots = (order[-1], order[0], order[-2], None)
        slot = slots[row['target']]
        cases.append({**row, 'split': 'fresh', 'legacy_label': former[row['question']],
                      'gold': None if slot is None else [subject, slot]})
    cases.extend({'question': row['question'], 'split': 'regression', 'pair': None,
                  'legacy_label': row['legacy_label'], 'gold': row['gold']} for row in before.values())
    proposals = predictions(heads, *clause_features(model, tokenizer, [row['question'] for row in cases]))
    rows = [scoped_row(row, {mode: items[index] for mode, items in proposals.items()}, thresholds,
                       fingerprints, fields, schedules) for index, row in enumerate(cases)]
    generate_rows = []
    for row in rows:
        old = before.get(row['question']) if row['split'] == 'regression' else None
        if old is not None and row['routes']['baseline'] != old['routes']['scoped_auto']:
            raise ValueError('The unchanged baseline did not reproduce a regression case')
        if old is not None and row['routes']['roles'] == old['routes']['scoped_auto']:
            row['automatic_output'] = old['automatic_output']
        else:
            generate_rows.append(row)
    summary = summarize(rows)
    print('Version routing', summary, flush=True)
    state = torch.load(reader_path, map_location='cpu', weights_only=True)['atomic']
    reader = VersionReplyActions(model.config.hidden_size, state['scaffold_tokens'])
    reader.load_state_dict(state, strict=True)
    reader.eval().to('cuda')
    verify_answers(model, tokenizer, bank, reader, generate_rows, route_mode='roles')
    if any(base.sha(Path(path)) != digest for path, digest in protected.items()):
        raise ValueError('Version evaluation changed a protected source')
    generation = {group: {'correct_values': sum(row['automatic_output']['correct_value'] for row in rows if row['split'] == group),
                          'negative_mentions': sum(row['gold'] is None and bool(row['automatic_output']['mentioned_slots'])
                                                   for row in rows if row['split'] == group)}
                  for group in ('fresh', 'regression')}
    report = {'sources': protected, 'heads_sha256': base.sha(weights_path), 'thresholds': thresholds,
              'training': len(train), 'calibration': len(cal), 'steps': args.steps if prior is None else prior['steps'],
              'summary': summary, 'generation': generation, 'rows': rows}
    if prior is not None and any(prior[key] != value for key, value in report.items()):
        raise ValueError('Version roles or generated responses changed after reload')
    print(generation, flush=True)
    store_json(args.output / 'reload.json' if args.verify else result_path,
               {**report, 'seconds': perf_counter() - started, 'peak_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scope', type=Path)
    parser.add_argument('timeline', type=Path)
    parser.add_argument('reader', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--steps', type=int, default=180)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
