"""Measure joint role selection, independent name boundaries, and abstention."""

import argparse
import json
from pathlib import Path

from scripts.train_spora_name_role import (
    CALIBRATION, CALIBRATION_NAMES, F, FRESH, FRESH_NAMES, HELD_NAMES, HELD_TEXT, HELD_VALUES, MODEL, SEED,
    TRAIN, TRAIN_NAMES, AutoTokenizer, FrozenTurnFeatures, JointAddress, atomic_save, examples, feature_batch,
    frozen_digest, load_full_streamed, load_integrated_runtime, replay, role_examples, sha, store_json, summary, torch,
)
from spora.research_name_role import JointNameRoleHead, compare_name_scores


FORMAT = 'spora-joint-name-role-v1'


def pad_candidates(inputs, counts):
    width = max(1, *counts)
    padded = inputs.new_zeros(len(counts), width, *inputs.shape[1:])
    present = torch.arange(width)[None, :] < torch.tensor(counts)[:, None]
    padded[present] = inputs
    return padded, present


def training_targets(rows, present):
    null = present.shape[1]
    targets = torch.full((len(rows),), null, dtype=torch.long)
    complete = torch.zeros_like(present, dtype=torch.float32)
    for i, row in enumerate(rows):
        for j, (_, _, key) in enumerate(row['candidates']):
            complete[i, j] = row['complete'][j]
            if key == row['target']:
                targets[i] = j
    return targets, complete


def fit(head, features, present, rows, *, boundary_weight, epochs=14):
    target, complete = training_targets(rows, present)
    weight = torch.where(target == present.shape[1], 1., 3.)
    rng = torch.Generator().manual_seed(SEED)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.003, weight_decay=.03)
    for _ in range(epochs):
        for indices in torch.randperm(len(rows), generator=rng).split(96):
            active = present[indices]
            optimizer.zero_grad(set_to_none=True)
            decision, boundary = head(features[indices], active)
            target_loss = F.cross_entropy(decision, target[indices], reduction='none')
            boundary_loss = F.binary_cross_entropy_with_logits(
                boundary[active], complete[indices][active], reduction='sum') / active.sum().clamp_min(1)
            loss = (target_loss * weight[indices]).mean() + boundary_weight * boundary_loss
            loss.backward()
            optimizer.step()
    return head.eval().requires_grad_(False)


def decisions(head, rows, padded, present):
    decision, boundary = head(padded, present)
    indices, margins = compare_name_scores(decision, boundary, present)
    details = []
    for row, index, margin in zip(rows, indices.tolist(), margins.tolist(), strict=True):
        candidate = row['candidates'][index] if row['candidates'] else None
        details.append((candidate, margin))
    return details


def apply_floor(rows, choices, floor):
    details = []
    for row, (candidate, margin) in zip(rows, choices, strict=True):
        selected = candidate[2] if candidate is not None and margin > floor else None
        details.append({**row, 'candidate': candidate, 'margin': margin if candidate is not None else None,
                        'selected': selected, 'correct': selected == row['target']})
    return details


def calibrated_floor(rows, choices):
    bad = [margin for row, (candidate, margin) in zip(rows, choices, strict=True)
           if candidate is not None and candidate[2] != row['target']]
    return max(0., *bad)


def resolve(head, floor, rows, inputs, counts):
    choices = decisions(head, rows, *pad_candidates(inputs, counts))
    return [candidate if margin > floor else None for candidate, margin in choices]


def main(checkpoint, addresses, output, verify):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    hashes = {'checkpoint': sha(checkpoint), 'addresses': sha(addresses)}
    source = torch.load(addresses, map_location='cpu', weights_only=True)
    if source['format'] != 'spora-entity-field-address-probe-v1' or source['checkpoint_sha256'] != hashes['checkpoint']:
        raise ValueError('Field address weights do not match this model')
    hashes['individual'] = source['individual_sha256']
    weights = torch.load(verify / 'roles.pt', map_location='cpu', weights_only=True) if verify else None
    if weights is not None and (weights['format'] != FORMAT or weights['sources'] != hashes):
        raise ValueError('Joint role weights do not match these inputs')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    runtime.decoder.eval().requires_grad_(False)
    if frozen_digest(runtime.organism, runtime.state) != hashes['individual']:
        raise ValueError('The field address weights belong to a different individual')
    featurize = FrozenTurnFeatures(runtime.decoder, tokenizer)
    rows_train = role_examples(TRAIN_NAMES, TRAIN, '荣安云庆岚薇青岳东梦', separate_boundary=True)
    rows_cal = role_examples(CALIBRATION_NAMES, CALIBRATION, '嘉河亭琛柠添', separate_boundary=True)
    rows_test = role_examples(FRESH_NAMES, FRESH, '飞川宁华星明', separate_boundary=True)

    def features(rows):
        inputs, _, counts = feature_batch(featurize, rows)
        return pad_candidates(inputs, counts)

    train = features(rows_train) if weights is None else None
    if train is not None:
        print('Joint training pairs', len(rows_train), int(train[1].sum()), flush=True)
    calibration, fresh = features(rows_cal), features(rows_test)
    arms, configs, states = {}, {}, {}
    for name, weight in (('selection', 0.), ('boundary_supervised', 1.)):
        torch.manual_seed(SEED)
        head = JointNameRoleHead(source['hidden'])
        if weights is None:
            head = fit(head, *train, rows_train, boundary_weight=weight)
        else:
            head.load_state_dict(weights['heads'][name], strict=True)
            head.eval().requires_grad_(False)
        states[name] = head.state_dict()
        cal_choices = decisions(head, rows_cal, *calibration)
        test_choices = decisions(head, rows_test, *fresh)
        for policy, floor in (('fixed', 0.), ('calibrated', calibrated_floor(rows_cal, cal_choices))):
            key = name + '_' + policy
            test = apply_floor(rows_test, test_choices, floor)
            arms[key] = {'floor': floor, 'calibration': summary(apply_floor(rows_cal, cal_choices, floor)),
                         'fresh': summary(test), 'rows': test}
            configs[key] = (head, floor)
            print(key, floor, arms[key]['fresh'], flush=True)

    joint = JointAddress(source['hidden']).eval().requires_grad_(False)
    joint.load_state_dict(source['heads']['JointAddress'], strict=True)
    held = examples(HELD_NAMES, HELD_TEXT, HELD_VALUES, SEED + 1)
    for name, value in replay(held, featurize, configs, joint, tokenizer, resolve=resolve).items():
        arms[name]['slot_replay'] = value
        print(name, value['summary'], flush=True)
    if (sha(checkpoint) != hashes['checkpoint'] or sha(addresses) != hashes['addresses']
            or frozen_digest(runtime.organism, runtime.state) != hashes['individual']):
        raise ValueError('Source model or individual changed during the experiment')
    record = {'format': FORMAT, 'sources': hashes, 'train': len(rows_train), 'arms': arms}
    output.mkdir(parents=True, exist_ok=False)
    if weights is None:
        atomic_save({'format': FORMAT, 'sources': hashes, 'heads': states}, output / 'roles.pt')
        store_json(output / 'result.json', record)
    else:
        previous = json.loads((verify / 'result.json').read_text(encoding='utf-8'))
        if json.loads(json.dumps(record)) != previous:
            raise ValueError('Reloaded role model did not reproduce the audit')
        store_json(output / 'result.json', {'identical': True, 'sources': hashes})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'addresses', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--verify', type=Path)
    main(**vars(parser.parse_args()))
