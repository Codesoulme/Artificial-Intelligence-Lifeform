"""Compare wider name recall and training with a missing target mention."""

import argparse
import json
from pathlib import Path

from scripts.train_spora_open_name_roles import (
    CALIBRATION, CALIBRATION_NAMES, FORMAT as BASE_FORMAT, FRESH, FRESH_NAMES, HELD_NAMES, HELD_TEXT, HELD_VALUES,
    MODEL, SEED, TRAIN, TRAIN_NAMES, AutoTokenizer, FrozenTurnFeatures, JointAddress, JointNameRoleHead, NameSpanHead,
    atomic_save, decisions, examples, fit, frozen_digest, hidden_batch, load_full_streamed, load_integrated_runtime,
    proposals, replay_profiles, role_batch, role_examples, sha, span_metrics, store_json, summary, torch,
    validate_enrollment,
)


FORMAT = 'spora-missing-name-roles-v1'
CHALLENGE_NAMES = ('乔沅汐', '文楚林', '洪天宜', '黎冬阳', '范书琴', '武宜宁', '辛原可', '桑野明')
CHALLENGE = (
    '订单里属于{subject}的那份改到周日再领。', '约好给{subject}的材料究竟留了几箱？',
    '到集合那天，先查一下给{subject}记的地址。', '接待簿得更新，{subject}改去玻璃门右边等。',
    '以后由{value}处理咨询，记到{subject}的申请表上。', '{subject}档案里负责联络的人请换成{value}。',
    '以{value}为新的联络人，单独更新{subject}的那一行。', '请更新预留单，{subject}那项转交{value}联络。',
)


def choices(names, templates, affixes):
    return role_examples(names, templates, affixes, separate_boundary=True, annotate_roles=True)


def train_profiles(role_states, width, featurize):
    train = choices(TRAIN_NAMES, TRAIN, '荣安云庆岚薇青岳东梦')
    hidden, _ = hidden_batch(featurize, train)
    spans = [sorted(set(row['mentions']) | {(a, b) for a, b, _ in row['candidates']}) for row in train]
    duplicated = [i for i, row in enumerate(train) if len(row['mentions']) > 1 and row['semantic_target'] is not None]
    ids = list(range(len(train))) + duplicated
    rows, (inputs, present) = role_batch([train[i] for i in ids], hidden[ids], [spans[i] for i in ids])
    absent_rows, absent_present = list(rows), present.clone()
    for i in range(len(train), len(rows)):
        target = rows[i]['target']
        absent_present[i, target] = False
        absent_rows[i] = {**rows[i], 'target': None}

    print('Matched continuation examples', len(rows), 'missing target contrasts', len(duplicated), flush=True)
    output = {}
    for mode, samples, mask in (('matched', rows, present), ('missing_target', absent_rows, absent_present)):
        head = JointNameRoleHead(width)
        head.load_state_dict(role_states, strict=True)
        output[mode] = fit(head, inputs, mask, samples, boundary_weight=1., epochs=6)
    return output, len(rows), len(duplicated)


def calibrate_recall(proposer, hidden, lengths, rows):
    stats = {floor: span_metrics(rows, proposals(proposer, hidden, lengths, floor))
             for floor in (-4., -3., -2., -1., 0.)}
    # F2 emphasizes recovering complete names without accepting every substring.
    best = max(stats, key=lambda floor: 5 * stats[floor]['correct'] /
               (4 * stats[floor]['expected'] + stats[floor]['proposed']))
    return best, {str(key): value for key, value in stats.items()}


def score_requests(head, rows, hidden, spans):
    labelled, features = role_batch(rows, hidden, spans)
    scored = decisions(head, labelled, *features)
    output = []
    for row, (candidate, margin) in zip(rows, scored, strict=True):
        span = candidate[:2] if candidate is not None else None
        enrolled = validate_enrollment(span, row['candidates'])
        output.append((span, margin, enrolled[2] if enrolled is not None else None))
    return output


def summarize_requests(rows, spans, scored, floor):
    details = []
    for row, found, (span, margin, eligible) in zip(rows, spans, scored, strict=True):
        owner = eligible if margin > floor else None
        details.append({**row, 'proposed': found, 'winner': span, 'margin': margin if span is not None else None,
                        'owner': owner, 'correct': owner == row['target']})
    return {'summary': summary(details), 'rows': details}


def evaluate_set(proposer, configs, featurize, rows):
    hidden, lengths = hidden_batch(featurize, rows)
    floors = dict.fromkeys(floor for _, floor, _ in configs.values())
    spans = {floor: proposals(proposer, hidden, lengths, floor) for floor in floors}
    result = {}
    for name, (head, floor, margin) in configs.items():
        scored = score_requests(head, rows, hidden, spans[floor])
        result[name] = summarize_requests(rows, spans[floor], scored, margin)
        print(name, result[name]['summary'], flush=True)
    return {'proposals': {str(key): span_metrics(rows, found) for key, found in spans.items()}, 'arms': result}


def main(checkpoint, addresses, prior, output, verify):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    paths = {'checkpoint': checkpoint, 'addresses': addresses, 'prior': prior / 'roles.pt',
             'prior_report': prior / 'result.json'}
    hashes = {name: sha(path) for name, path in paths.items()}
    base = torch.load(addresses, map_location='cpu', weights_only=True)
    if base['format'] != 'spora-entity-field-address-probe-v1' or base['checkpoint_sha256'] != hashes['checkpoint']:
        raise ValueError('Field address weights do not match this model')
    hashes['individual'] = base['individual_sha256']
    previous = torch.load(paths['prior'], map_location='cpu', weights_only=True)
    original_sources = {name: hashes[name] for name in ('checkpoint', 'addresses', 'individual')}
    if previous['format'] != BASE_FORMAT or previous['sources'] != original_sources:
        raise ValueError('Previous role weights do not match this individual')
    stored = torch.load(verify / 'roles.pt', map_location='cpu', weights_only=True) if verify else None
    if stored is not None and (stored['format'] != FORMAT or stored['sources'] != hashes):
        raise ValueError('Adapted role weights do not match this run')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    runtime.decoder.eval().requires_grad_(False)
    if frozen_digest(runtime.organism, runtime.state) != hashes['individual']:
        raise ValueError('Field address weights belong to a different individual')
    featurize = FrozenTurnFeatures(runtime.decoder, tokenizer)
    proposer = NameSpanHead(base['hidden']).eval().requires_grad_(False)
    proposer.load_state_dict(previous['proposer'], strict=True)
    original = JointNameRoleHead(base['hidden']).eval().requires_grad_(False)
    original.load_state_dict(previous['role'], strict=True)
    strict_floor = json.loads(paths['prior_report'].read_text(encoding='utf-8'))['proposal_floor']
    if stored is None:
        adapted, train_count, contrasts = train_profiles(previous['role'], base['hidden'], featurize)
    else:
        adapted, train_count, contrasts = {}, stored['training']['requests'], stored['training']['contrasts']
        for name, state in stored['heads'].items():
            head = JointNameRoleHead(base['hidden']).eval().requires_grad_(False)
            head.load_state_dict(state, strict=True)
            adapted[name] = head
    calibration = choices(CALIBRATION_NAMES, CALIBRATION, '嘉河亭琛柠添')
    hidden, lengths = hidden_batch(featurize, calibration)
    wide, span_calibration = calibrate_recall(proposer, hidden, lengths, calibration)
    wide_spans = proposals(proposer, hidden, lengths, wide)
    learned = adapted['missing_target']
    scored = score_requests(learned, calibration, hidden, wide_spans)
    caution = max(0., *(margin for row, (_, margin, key) in zip(calibration, scored, strict=True)
                        if key is not None and key != row['target']))
    configs = {'previous': (original, strict_floor, 0.), 'wider': (original, wide, 0.),
               'matched': (adapted['matched'], wide, 0.), 'missing_target': (learned, wide, 0.),
               'missing_target_calibrated': (learned, wide, caution)}
    print('Proposal floors', strict_floor, wide, 'learned margin', caution, flush=True)
    development = evaluate_set(proposer, configs, featurize, choices(FRESH_NAMES, FRESH, '飞川宁华星明'))
    challenge = evaluate_set(proposer, configs, featurize, choices(CHALLENGE_NAMES, CHALLENGE, '延竹鸿甫初泉'))
    held = examples(HELD_NAMES, HELD_TEXT, HELD_VALUES, SEED + 1)
    joint = JointAddress(base['hidden']).eval().requires_grad_(False)
    joint.load_state_dict(base['heads']['JointAddress'], strict=True)
    slots = replay_profiles(held, featurize, proposer, configs, joint, tokenizer)
    for name, results in slots.items():
        print(name, results['summary'], flush=True)
    if (any(sha(path) != hashes[name] for name, path in paths.items())
            or frozen_digest(runtime.organism, runtime.state) != hashes['individual']):
        raise ValueError('A source weight or individual state changed')
    training = {'requests': train_count, 'contrasts': contrasts, 'epochs': 6}
    report = {'format': FORMAT, 'sources': hashes, 'training': training, 'span_calibration': span_calibration,
              'thresholds': {name: [floor, margin] for name, (_, floor, margin) in configs.items()},
              'development': development, 'challenge': challenge, 'slot_replay': slots}
    output.mkdir(parents=True, exist_ok=False)
    if stored is None:
        atomic_save({'format': FORMAT, 'sources': hashes, 'training': training,
                     'heads': {name: head.state_dict() for name, head in adapted.items()}}, output / 'roles.pt')
        store_json(output / 'result.json', report)
    else:
        reference = json.loads((verify / 'result.json').read_text(encoding='utf-8'))
        if json.loads(json.dumps(report)) != reference:
            raise ValueError('Reloaded results differ from training results')
        store_json(output / 'result.json', {'identical': True, 'sources': hashes})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'addresses', 'prior', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--verify', type=Path)
    main(**vars(parser.parse_args()))
