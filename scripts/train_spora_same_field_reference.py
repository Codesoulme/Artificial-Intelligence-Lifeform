"""Match scoped revisions across two working records that share a field type."""

import argparse
from contextlib import ExitStack
from itertools import combinations
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.train_spora_described_revision_span import assess as extract_values, encode as encode_values, make_rows
from scripts.train_spora_revision_intent import JointTurnValueWriter
from scripts.train_spora_revision_reference import training_cases
from scripts.train_spora_separate_revision_reference import (
    ACTIONS, FIELD_TEXT, SEED, UNSEEN_TOPIC, DraftReferenceMatcher, cuts, encode, encode_texts, predictions, tensors,
    train_heads,
)
from scripts.train_spora_observed_episodes import encode_changes, state_for, state_values
from scripts.train_spora_revision_intent import (
    DATA, DESIGN, EPISODES, EXTRA_NAMES, HELD, MODEL, OBSERVATIONS, ROLE_NAMES, AutoTokenizer, atomic_save, evaluate,
    extend_positions, frozen_digest, held_episodes, install, load_full_streamed, load_integrated_runtime,
    load_recurrent_replacements, read, records, sha, store_json,
)


TOPIC_ALIASES = {
    'clinic_booking': (('复诊这事', '医院复查', '门诊预约'), '回去看诊', ('去医院的那趟', '跟诊室约好的事')),
    'repair_pickup': (('维修取件', '拿回修好的东西', '领修理件'), '取回修好的物品', ('修完之后领东西', '去售后拿回那件')),
    'pottery_inventory': (('备货交接', '陶器出货', '仓库发货'), '发货清单', ('把库里的订单送出去', '给收货方配东西')),
    'choir_program': (('合唱演出', '合唱节目', '台上唱歌'), '合唱排演', ('那群人上台一起唱歌', '登台唱曲子')),
    'garden_supply': (('花坛工具', '花园活动', '园艺小组'), '花圃工作', ('几个人去打理苗圃', '给花地干活的那拨人')),
    'film_notice': (('影片放映', '电影通知', '影院安排'), '影厅活动', ('给观众放片子', '放电影的那场')),
}
CONFIRMATIONS = {'train': ACTIONS['train'][0], 'calibration': ACTIONS['calibration'][0], 'held': ACTIONS['held'][0]}
DELAYS = {'train': ACTIONS['train'][1], 'calibration': ACTIONS['calibration'][1], 'held': ACTIONS['held'][1]}
SCOPE = {'train': '我刚才指的是{alias}。', 'calibration': '这里说的是关于{alias}的那件。',
         'held': '先只谈{alias}，另一件等会儿说。'}
AMBIGUOUS = {'train': '刚才提了两件，具体是哪件还没有说明。', 'calibration': '两项都谈到了，尚未单独选定。',
             'held': '提到的两件事现在还没区分清楚。'}


def topic_cue(topic, split, index):
    train, calibration, held = TOPIC_ALIASES[topic]
    return {'train': train[index % len(train)], 'calibration': calibration,
            'held': held[index % len(held)]}[split]


def proposals(cases, names, fixtures):
    grouped = {}
    for case in cases:
        grouped.setdefault(case['topic'], []).append(case)
    result = {}
    for topic, rows in grouped.items():
        result[topic] = {}
        for field, label in names[topic].items():
            pool = sorted({row['initial'][field] for row in rows})
            options = []
            for index, row in enumerate(rows):
                old = row['initial'][field]
                choices = [value for value in pool if value != old]
                candidate = choices[index % len(choices)]
                options.append({'topic': topic, 'field': field, 'value': candidate, 'initial': row['initial'],
                                'name': label, 'theme': fixtures[topic]['label'], 'old': old,
                                'record_key': row['key'], 'first': row['first']})
            result[topic][field] = options
    return result


def build(catalog, split, *, per_pair, key_kind='description'):
    rows = []
    for left, right in combinations(sorted(catalog), 2):
        for field in sorted(catalog[left].keys() & catalog[right].keys()):
            a, b = catalog[left][field], catalog[right][field]
            for index in range(per_pair):
                drafts = (a[index % len(a)], b[(index + 1) % len(b)])
                for order in (drafts, drafts[::-1]):
                    descriptions = tuple(draft_key(draft, key_kind) for draft in order)
                    for target in (0, 1, None):
                        bridge = (AMBIGUOUS[split] if target is None else SCOPE[split].format(
                            alias=topic_cue(order[target]['topic'], split, index)))
                        utterances = (CONFIRMATIONS[split], DELAYS[split])
                        for commands, confirm in zip(utterances, (True, False), strict=True):
                            rows.append({'bridge': bridge, 'utterance': commands[index % len(commands)],
                                         'descriptions': descriptions, 'options': order, 'target': target,
                                         'confirm': confirm})
    return rows


def draft_key(draft, kind):
    if kind == 'observation':
        return draft['record_key']
    if kind == 'topic':
        return draft['theme']
    return f'{draft["theme"]}的{draft["name"]}，{FIELD_TEXT[draft["field"]]}'


def observation_keys(featurizer, sets):
    observations = {}
    for rows in sets.values():
        for row in rows:
            for draft in row['options']:
                key, text = draft['record_key'], draft['first']
                if observations.setdefault(key, text) != text:
                    raise ValueError('A scoped record has conflicting registration turns')
    encoded = encode_texts(featurizer, sorted(set(observations.values())))
    return {key: encoded[text] for key, text in observations.items()}


def copied_candidates(writer, featurizer, cases, schemas):
    # Each original proposal is read separately. The known topic determines which record owns the span.
    rows = make_rows(cases, schemas)
    encoded = encode_values(featurizer, rows, 'identifier')
    return {(row['turn'], row['topic']): row for row in extract_values(writer, encoded, cases, rows, schemas)}


def proposal_text(draft):
    return (f'原先{draft["theme"]}的{draft["name"]}记成{draft["old"]}。'
            f'要不要改用{draft["value"]}，我先征求一下意见。')


def audit(rows, scores, tokenizer, boundaries, copied=None, *, given_field=False):
    scope_cut, action_cut = boundaries
    details = []
    for row, (pick, gap, action) in zip(rows, scores, strict=True):
        drafts = row['options']
        target = row['target'] if row['confirm'] else None
        selected = pick if action > action_cut and gap > scope_cut else None
        initial = {d['topic']: d['initial'] for d in drafts}
        actual = {scope: dict(state) for scope, state in initial.items()}
        expected = {scope: dict(state) for scope, state in initial.items()}
        if target is not None:
            opt = drafts[target]
            expected[opt['topic']][opt['field']] = opt['value']
        if selected is not None:
            opt = drafts[selected]
            if copied is None:
                proposal = {'field': opt['field'], 'text': opt['value']}
            else:
                entry = copied[proposal_text(opt), opt['topic']]
                proposal = entry['given_field' if given_field else 'automatic']
            state = state_for(tokenizer, initial[opt['topic']])
            patch = encode_changes(tokenizer, {proposal['field']: proposal['text']})
            actual[opt['topic']] = state_values(tokenizer, state.update(patch))
        details.append({'topics': [d['topic'] for d in drafts], 'field': drafts[0]['field'], 'bridge': row['bridge'],
                        'utterance': row['utterance'], 'target': target, 'reference': row['target'], 'selected': selected,
                        'top': pick, 'gap': gap, 'action': action, 'correct_state': actual == expected,
                        'expected': expected, 'actual': actual})
    return details


def summary(details):
    return {'turns': len(details), 'should_commit': sum(row['target'] is not None for row in details),
            'right_rank': sum(row['top'] == row['reference'] for row in details),
            'reference_cases': sum(row['reference'] is not None for row in details),
            'right_commit': sum(row['target'] is not None and row['selected'] == row['target'] and row['correct_state']
                                for row in details),
            'wrong_commit': sum(row['selected'] is not None and (row['target'] != row['selected'] or not row['correct_state'])
                                for row in details),
            'state_exact': sum(row['correct_state'] for row in details)}


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    output = args.output
    report, weights, stored_keys = output / 'result.json', output / 'reference.pt', output / 'observation-keys.pt'
    if not args.verify:
        output.mkdir(parents=True, exist_ok=False)
    parent = read(args.reference / 'result.json')
    writer_parent = read(args.writer / 'result.json')
    ref_weights, writer_weights = args.reference / 'heads.pt', args.writer / 'writers.pt'
    paths = (args.checkpoint, DATA, DESIGN, EPISODES, OBSERVATIONS, args.reference / 'result.json', ref_weights,
             args.writer / 'result.json', writer_weights,
             *(folder / name for folder in args.layers for name in ('attention.pt', 'comparison.json')))
    sources = {str(path): sha(path) for path in paths}
    inherited = {Path(path).resolve(): digest for path, digest in parent['sources'].items()}
    if any(sources[str(path)] != inherited.get(path.resolve()) for path in paths
           if path.parent not in (args.reference, args.writer)):
        raise ValueError('Frozen inputs changed')
    if (sources[str(ref_weights)] != parent['weights_sha256'] or sources[str(writer_weights)] != writer_parent['weights_sha256']
            or parent['memory'] != writer_parent['memory']):
        raise ValueError('Parent heads or persistent reads differ')
    expected = read(report) if args.verify else None
    if expected is not None and (expected['sources'] != sources or expected['weights_sha256'] != sha(weights)):
        raise ValueError('Independent replay inputs changed')
    if args.verify and args.observation_key and expected['observation_keys_sha256'] != sha(stored_keys):
        raise ValueError('Stored observation representations changed')

    specs, data, fixtures = read(EPISODES)['cases'], read(DATA), read(OBSERVATIONS)['topics']
    labels = {topic: (*ROLE_NAMES[topic], EXTRA_NAMES[topic]) for topic in EXTRA_NAMES}
    schemas = {spec['case']: dict(zip(spec['fields'], labels[spec['case']], strict=True)) for spec in specs}
    train = training_cases(specs, fixtures, labels, data['values']['train'])
    calibration = training_cases(specs, fixtures, labels, data['values']['held'], calibration=True)
    held = held_episodes({'cases': specs})
    sets = {}
    kind = 'observation' if args.observation_key else ('topic' if args.topic_key else 'description')
    for split, cases, count in (('train', train, 9), ('calibration', calibration, 4), ('held', held, 4)):
        valid = [case for case in cases if split == 'held' or case['topic'] != UNSEEN_TOPIC]
        sets[split] = build(proposals(valid, schemas, fixtures), split, per_pair=count, key_kind=kind)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed()
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    if individual != parent['individual_sha256'] or individual != writer_parent['individual_sha256']:
        raise ValueError('The individual changed')
    extend_positions(decoder, rotary, 2048)
    feature = FrozenTurnFeatures(decoder, tokenizer, depth=20)
    evaluation = {name: rows for name, rows in sets.items() if name != 'train'}
    key_bank = None
    if args.observation_key:
        if args.verify:
            key_bank = torch.load(stored_keys, weights_only=True, map_location='cpu')
        else:
            key_bank = observation_keys(feature, evaluation)
            atomic_save(key_bank, stored_keys)
        required = {key for rows in evaluation.values() for row in rows for key in row['descriptions']}
        if key_bank.keys() != required or any(v.shape != (base.config.hidden_size,) for v in key_bank.values()):
            raise ValueError('Stored observation bank does not match the evaluation records')
    eval_features = encode(feature, evaluation, stored=key_bank)
    hidden = base.config.hidden_size
    original = torch.load(ref_weights, weights_only=True, map_location='cpu')
    action = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 2)).eval().requires_grad_(False)
    action.load_state_dict(original['action'])
    torch.manual_seed(SEED)
    matcher = DraftReferenceMatcher(hidden)
    matcher.load_state_dict(original['reference'])
    baseline = {name: predictions(matcher, action, tensors(rows, eval_features)) for name, rows in evaluation.items()}
    if args.verify:
        matcher.load_state_dict(torch.load(weights, weights_only=True, map_location='cpu'))
    else:
        training = {'train': sets['train']}
        keys = observation_keys(feature, training) if args.observation_key else None
        train_features = encode(feature, training, stored=keys)
        train_heads(matcher, action, tensors(sets['train'], train_features))
        atomic_save(matcher.state_dict(), weights)
    matcher.eval()
    scope = {name: predictions(matcher, action, tensors(rows, eval_features)) for name, rows in evaluation.items()}

    inputs = sorted({(proposal_text(draft), draft['topic'], draft['field'], draft['value'])
                     for row in sets['held'] for draft in row['options']})
    writer = JointTurnValueWriter(64, feature_width=hidden).to(base.device).eval()
    writer.load_state_dict(torch.load(writer_weights, map_location='cpu', weights_only=True)['identifier'])
    copied = copied_candidates(writer, FrozenTurnFeatures(decoder, tokenizer, depth=12), inputs, schemas)
    results = {}
    for name, scored in (('earlier', baseline), ('scoped', scope)):
        limits = cuts(sets['calibration'], scored['calibration'])
        known = audit(sets['held'], scored['held'], tokenizer, limits)
        automatic = audit(sets['held'], scored['held'], tokenizer, limits, copied)
        results[name] = {'limits': list(limits), 'known': summary(known), 'automatic': summary(automatic),
                         'unseen': summary([row for row in known if UNSEEN_TOPIC in row['topics']]),
                         'known_rows': known, 'automatic_rows': automatic}
        print(name, results[name]['known'], results[name]['automatic'], results[name]['unseen'], flush=True)
    layers = load_recurrent_replacements(decoder, args.layers, sources[str(args.checkpoint)])
    with ExitStack() as stack:
        install(stack, layers)
        memory = evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))
    if (memory != parent['memory'] or frozen_digest(runtime.organism, runtime.state) != individual
            or any(sha(path) != sources[str(path)] for path in paths)):
        raise ValueError('Frozen sources or durable memory changed')
    extracts = {'count': len(inputs), 'exact': sum(row['automatic']['field'] == row['expected_field']
                                                  and row['automatic']['text'] == row['expected_value']
                                                  for row in copied.values())}
    result = {'sources': sources, 'individual_sha256': individual, 'weights_sha256': sha(weights),
              'train_turns': len(sets['train']), 'calibration_turns': len(sets['calibration']),
              'value_extraction': extracts, 'arms': results, 'memory': memory}
    if args.observation_key:
        result['observation_keys_sha256'] = sha(stored_keys)
        result['observation_keys'] = len(key_bank)
    if args.verify:
        if any(expected[key] != value for key, value in result.items()):
            raise ValueError('Independent replay did not match')
        held = sets['held']
        oracle = [(row['target'] or 0, float(row['target'] is not None), action_margin)
                  for row, (_, _, action_margin) in zip(held, scope['held'], strict=True)]
        limits = (0., results['scoped']['limits'][1])
        diagnostic = {'count': len(inputs),
                      'given_field_exact': sum(row['given_field']['text'] == row['expected_value'] for row in copied.values()),
                      'automatic_field_exact': sum(row['automatic']['field'] == row['expected_field'] for row in copied.values()),
                      'automatic_exact': extracts['exact'],
                      'given_topic': summary(audit(held, oracle, tokenizer, limits, copied)),
                      'given_topic_and_field': summary(audit(held, oracle, tokenizer, limits, copied, given_field=True)),
                      'proposals': list(copied.values())}
        store_json(output / 'candidate-audit.json', diagnostic)
        print('candidate audit', {key: value for key, value in diagnostic.items() if key != 'proposals'}, flush=True)
        store_json(output / 'reload.json', {'weights_sha256': result['weights_sha256'], 'seconds': perf_counter() - started})
    else:
        store_json(report, {**result, 'seconds': perf_counter() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'reference', 'writer', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('layers', nargs=2, type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--topic-key', action='store_true', help='Match using only the topic representation')
    group.add_argument('--observation-key', action='store_true', help='Match vectors saved from the original registration turns')
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
