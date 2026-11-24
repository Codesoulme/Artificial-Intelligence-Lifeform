"""Train a native role and name boundary head, then replay memory corrections."""

import argparse
import json
from pathlib import Path

from scripts.audit_spora_entity_field_address import (
    F, FIELDS, HELD_NAMES, HELD_TEXT, HELD_VALUES, MODEL, SEED, TRAIN_NAMES, AutoTokenizer,
    FrozenTurnFeatures, IdentityFingerprints, JointAddress, TokenValue, examples, frozen_digest,
    load_full_streamed, load_integrated_runtime, representations, sha, store_json, torch,
)
from scripts.audit_spora_natural_field_writes import measure, observations, summarize
from spora.language_training import atomic_save
from spora.research_name_role import NameRoleHead, mention_features, select_name


TRAIN = (
    '把{subject}的登记地点换到西门售票处。', '轮到{subject}时需要带多少盒？',
    '关于{subject}，到场时间仍记为早上九点。', '请为{subject}保留前天定的三捆缎带。',
    '查一下单子上{subject}负责的是哪个时段。', '明天先在木门旁等{subject}到场。',
    '{subject}也要改成周二下午来签字。', '请改为{value}通知{subject}领取材料。',
    '{subject}的联络事宜之后交给{value}跟进。', '这周{value}接手{subject}登记的活动联络。',
    '给{subject}换个联系人，新对接的是{value}。', '新对接人是{value}，请写进{subject}那一栏。',
)
CALIBRATION = (
    '给{subject}留的收货地址请重新填一下。', '{subject}先改到午后过来。',
    '以后由{value}负责给{subject}送回执。', '从{subject}那一栏查现在的联络人。',
)
FRESH = (
    '下午才能签收，把这个新时间记在{subject}名下。', '最晚几点能碰到{subject}？',
    '{subject}主要在五楼等候，登记地点请跟着变。', '给我确认预留给{subject}的件数。',
    '另外由{value}给{subject}发进场通知。', '请让{subject}以后只和{value}确认排期。',
    '我想改{subject}这单，新联系人填{value}即可。', '这次以{value}为准，让他对接{subject}的申请。',
)
CALIBRATION_NAMES = ('曾沫青', '柴昭旭', '蒲若锦', '焦允冬', '殷照禾', '莫云帆', '瞿初瑾', '汤云嘉')
FRESH_NAMES = ('夏礼文', '尹嘉言', '景怀宁', '康溪月', '易临风', '姚知序', '白可心', '陶念青')


def identity_group(names):
    registry = IdentityFingerprints(len(names))
    for key, name in enumerate(names):
        registry.bind(key, name)
    return registry


def role_examples(names, templates, affixes, *, separate_boundary=False, annotate_roles=False):
    rows = []
    for index in range(0, len(names), 4):
        group = names[index:index + 4]
        registry = identity_group(group)
        for i, subject in enumerate(group):
            foreign = names[(index + 4 + i) % len(names)] if separate_boundary else '袁继北'
            value = group[(i + 1) % 4]
            alternative = group[(i + 2) % 4]
            for j, pattern in enumerate(templates):
                affix = affixes[(index + i + j) % len(affixes)]
                variants = (
                    ('valid', subject, i), ('suffix', subject + affix, None),
                    ('prefix', affix + subject, None), ('two_names', subject + '和' + alternative, None),
                    ('unknown', foreign, None),
                )
                for kind, mention, target in variants:
                    text = pattern.format(subject=mention, value=value)
                    if kind == 'valid' and '{value}' in pattern:
                        kind = 'registered_contact'
                    candidates = registry.matching_spans(text, max_name_chars=8)
                    row = {'text': text, 'kind': kind, 'target': target, 'candidates': candidates}
                    if separate_boundary:
                        row['complete'] = [kind not in ('prefix', 'suffix') or key != i for _, _, key in candidates]
                    if annotate_roles:
                        mentions = [subject, alternative] if kind == 'two_names' else [mention]
                        if '{value}' in pattern:
                            mentions.append(value)
                        bounds = sorted((text.index(person), text.index(person) + len(person)) for person in mentions)
                        start = text.index(mention)
                        row['mentions'] = bounds
                        row['semantic_target'] = None if kind == 'two_names' else (start, start + len(mention))
                    rows.append(row)
    return rows


def feature_batch(featurize, rows):
    text = [row['text'] for row in rows]
    hidden = featurize(text)
    out, targets, counts = [], [], []
    for state, row in zip(hidden, rows, strict=True):
        counts.append(len(row['candidates']))
        for start, stop, key in row['candidates']:
            out.append(mention_features(state[:len(row['text'])], start, stop))
            targets.append(key == row['target'])
    return torch.stack(out), torch.tensor(targets, dtype=torch.float32), counts


def fit(head, inputs, targets, *, seed):
    rng = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.004, weight_decay=.03)
    positive_weight = (targets == 0).sum() / targets.sum()
    for _ in range(14):
        for selected in torch.randperm(len(targets), generator=rng).split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(head(inputs[selected]), targets[selected], pos_weight=positive_weight)
            loss.backward()
            optimizer.step()
    return head.eval().requires_grad_(False)


def results(rows, scores, threshold):
    details = []
    for row, score in zip(rows, scores, strict=True):
        picked = select_name(row['candidates'], score, threshold=threshold)
        winner = picked[2] if picked is not None else None
        details.append({**row, 'scores': score.tolist(), 'selected': winner, 'correct': winner == row['target']})
    return details


def summary(rows):
    return {key: {'correct': sum(row['correct'] for row in rows if row['kind'] == key),
                  'count': sum(row['kind'] == key for row in rows)}
            for key in dict.fromkeys(row['kind'] for row in rows)}


def split_scores(head, inputs, counts):
    return head(inputs).split(counts)


def calibration_floor(rows, scores):
    # The fixed floor is a baseline. The calibrated floor rejects every individual
    # false candidate that scores above it, even in examples with multiple names.
    bad = [float(score) for row, seq in zip(rows, scores, strict=True)
           for candidate, score in zip(row['candidates'], seq, strict=True) if candidate[2] != row['target']]
    return max(0., *bad)


def replay(source, featurize, args_by_name, joint, tokenizer, *, resolve=None):
    size = 4 * len(FIELDS)
    groups = [source[start:start + size] for start in range(0, len(source), size)]
    results = {name: [] for name in args_by_name}
    codec = TokenValue(len(tokenizer))
    for i, group in enumerate(groups):
        registry = identity_group([row['name'] for row in group[::len(FIELDS)]])
        foreign = [row['name'] for row in groups[(i + 1) % len(groups)][::len(FIELDS)]]
        events = list(observations(group, foreign, i))
        hidden = featurize([text for _, _, text in events]).float()
        rows = [{'text': text, 'candidates': registry.matching_spans(text, max_name_chars=8)} for _, _, text in events]
        all_features = [mention_features(state[:len(row['text'])], start, stop)
                        for state, row in zip(hidden, rows, strict=True) for start, stop, _ in row['candidates']]
        counts = [len(row['candidates']) for row in rows]
        inputs = torch.stack(all_features)
        initial = joint(*representations(featurize, group))[:, 0]
        for name, (head, floor) in args_by_name.items():
            if resolve is None:
                scores = split_scores(head, inputs, counts)
                selected = [select_name(row['candidates'], values, threshold=floor)
                            for row, values in zip(rows, scores, strict=True)]
            else:
                selected = resolve(head, floor, rows, inputs, counts)
            details = measure(group, events, selected, hidden, initial, codec, tokenizer, joint, name,
                              selector=lambda _mode, _registry, _text, chosen, _gold: chosen)
            results[name].extend(detail | {'predicted_spans': None} for detail in details)
    return {name: {'summary': summarize(rows), 'rows': rows} for name, rows in results.items()}


def main(checkpoint, addresses, output, verify):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    files = {path: sha(path) for path in (checkpoint, addresses)}
    source = torch.load(addresses, map_location='cpu', weights_only=True)
    if source['format'] != 'spora-entity-field-address-probe-v1' or source['checkpoint_sha256'] != files[checkpoint]:
        raise ValueError('Incompatible checkpoint and field address weights')
    metadata = {'checkpoint': files[checkpoint], 'addresses': files[addresses], 'individual': source['individual_sha256']}
    weights = None if verify is None else torch.load(verify / 'roles.pt', map_location='cpu', weights_only=True)
    if weights is not None and weights['sources'] != metadata:
        raise ValueError('Incompatible name role weights')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    runtime.decoder.eval().requires_grad_(False)
    if frozen_digest(runtime.organism, runtime.state) != metadata['individual']:
        raise ValueError('Field addresses belong to a different individual')
    featurize = FrozenTurnFeatures(runtime.decoder, tokenizer)
    if weights is None:
        training = role_examples(TRAIN_NAMES, TRAIN, '荣安云庆岚薇青岳东梦')
        t_input, t_target, _ = feature_batch(featurize, training)
        print('Training features', len(training), len(t_target), flush=True)
    calibration = role_examples(CALIBRATION_NAMES, CALIBRATION, '嘉河亭琛柠添')
    fresh = role_examples(FRESH_NAMES, FRESH, '飞川宁华星明')
    c_input, _, c_counts = feature_batch(featurize, calibration)
    f_input, _, f_counts = feature_batch(featurize, fresh)
    arms, configs, states = {}, {}, {}
    for contextual in (False, True):
        name = 'contextual' if contextual else 'boundary'
        torch.manual_seed(SEED)
        head = NameRoleHead(source['hidden'], contextual=contextual)
        if weights is None:
            head = fit(head, t_input, t_target, seed=SEED)
        else:
            head.load_state_dict(weights['heads'][name], strict=True)
            head.eval().requires_grad_(False)
        states[name] = head.state_dict()
        calibration_scores = split_scores(head, c_input, c_counts)
        fresh_scores = split_scores(head, f_input, f_counts)
        floors = {'fixed': 0., 'calibrated': calibration_floor(calibration, calibration_scores)}
        for policy, floor in floors.items():
            label = name + '_' + policy
            configs[label] = (head, floor)
            details = results(fresh, fresh_scores, floor)
            arms[label] = {'threshold': floor, 'parameters': sum(p.numel() for p in head.parameters()),
                           'calibration': summary(results(calibration, calibration_scores, floor)),
                           'fresh': {'summary': summary(details), 'rows': details}}
            print(label, floor, arms[label]['fresh']['summary'], flush=True)
    joint = JointAddress(source['hidden']).eval().requires_grad_(False)
    joint.load_state_dict(source['heads']['JointAddress'], strict=True)
    held = examples(HELD_NAMES, HELD_TEXT, HELD_VALUES, SEED + 1)
    reports = replay(held, featurize, configs, joint, tokenizer)
    for name, record in reports.items():
        arms[name]['slot_replay'] = record
        print(name, record['summary'], flush=True)
    if frozen_digest(runtime.organism, runtime.state) != metadata['individual'] or any(sha(p) != s for p, s in files.items()):
        raise ValueError('Source weights or individual state changed')
    output.mkdir(parents=True, exist_ok=False)
    if weights is None:
        atomic_save({'heads': states, 'sources': metadata}, output / 'roles.pt')
        store_json(output / 'result.json', {'sources': metadata, 'arms': arms})
    else:
        expected = json.loads((verify / 'result.json').read_text(encoding='utf-8'))
        actual = {'sources': metadata, 'arms': arms}
        if json.loads(json.dumps(actual)) != expected:
            raise ValueError('Reloaded role head did not reproduce the audit')
        store_json(output / 'result.json', {'identical': True, 'heads': len(states), 'sources': metadata})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'addresses', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--verify', type=Path)
    main(**vars(parser.parse_args()))
