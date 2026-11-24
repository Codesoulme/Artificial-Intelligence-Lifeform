"""Train on diverse correction requests while isolating role selection from name detection."""

import argparse
import json
from pathlib import Path

from scripts.audit_spora_cluener_name_spans import FORMAT as PROPOSER_FORMAT
from scripts.audit_spora_missing_name_roles import (
    CHALLENGE, CHALLENGE_NAMES, HELD_NAMES, HELD_TEXT, HELD_VALUES, MODEL, SEED, TRAIN, TRAIN_NAMES,
    AutoTokenizer, FrozenTurnFeatures, JointAddress, JointNameRoleHead, NameSpanHead, atomic_save, choices, examples,
    fit, frozen_digest, hidden_batch, load_full_streamed, load_integrated_runtime, proposals, role_batch, score_requests,
    sha, span_metrics, store_json, summarize_requests, torch,
)
from scripts.train_spora_open_name_roles import FORMAT as ROLE_FORMAT, replay_proposer_profiles


FORMAT = 'spora-correction-roles-v1'
EPOCHS = 6
FEATURE_BATCH = 64
TRAIN_CORRECTIONS = (
    '{subject}改在玻璃门里面等，给接待台留个新地点。',
    '备注得换一下，碰上{subject}要去西边长廊接。',
    '把原来的场地取消，新地址写到{subject}的记录上。',
    '麻烦{value}通知一下，把{subject}的等候点移到一楼柜台。',
    '下雨了，给{subject}安排的会面改到门厅里。',
    '这个改动只写在{subject}的地点栏，别改时间。',
    '和{subject}那项有关的消息请先交给{value}处理。',
    '现在{value}负责联络，记得覆盖掉{subject}项下的原联系人。',
    '帮我把{subject}这边的联络任务交给{value}。',
    '修一下{subject}的联系人记录，具体找{value}。',
    '这件事的对接换了人，由{value}接手，原单是{subject}的。',
    '需要对接{subject}的那个事项以后归{value}管。',
    '先看看要找{subject}见面该去哪里。',
    '麻烦查一下{subject}的事项现在由谁对接。',
)
HELD_CORRECTIONS = (
    '碰头位置有变，请让{subject}在灰色柜子边等我。',
    '新场地在二楼拐角，更新的是{subject}那张单。',
    '请{value}转告变更，接{subject}的地点移到接车廊。',
    '不用去以前的门口了，替{subject}标一下新的会合点。',
    '把{subject}这边的后续交接人填为{value}。',
    '接下来新联系人是{value}，只涉及{subject}的记录。',
    '{value}接手以后的联络，请把这个写进{subject}的记录。',
    '这次换的是{subject}那项的联系人，今后有事就找{value}。',
    '见到{subject}该往哪个地方走？我想查一下。',
    '能不能看看现在负责{subject}那项的是谁？',
)
NEW_NAMES = ('穆沐夕', '季雨同', '薛桥生', '李澄时', '钱怀秋', '祁敬安', '丁乐宁', '顾冬宜')


def no_change(templates, names, affixes):
    cases = choices(names, templates, affixes)
    return [{**row, 'kind': 'mention_only', 'target': None, 'semantic_target': None} for row in cases
            if row['kind'] in ('valid', 'registered_contact')]


def training_features(rows, width, featurize):
    options = [sorted(set(row['mentions']) | {(a, b) for a, b, _ in row['candidates']}) for row in rows]
    slots = max(1, *(len(spans) for spans in options))
    inputs = torch.zeros(len(rows), slots, 6, width, dtype=torch.float16)
    present = torch.zeros(len(rows), slots, dtype=torch.bool)
    labelled = []
    for start in range(0, len(rows), FEATURE_BATCH):
        end = min(start + FEATURE_BATCH, len(rows))
        hidden, _ = hidden_batch(featurize, rows[start:end])
        labels, (features, mask) = role_batch(rows[start:end], hidden, options[start:end])
        inputs[start:end, :features.shape[1]].copy_(features)
        present[start:end, :mask.shape[1]].copy_(mask)
        labelled.extend(labels)
    return labelled, inputs, present


def train_heads(state, width, featurize, progress, sources):
    mixed = choices(TRAIN_NAMES[:12], TRAIN_CORRECTIONS, '永齐茹立斐容')
    mixed.extend(no_change(('{subject}刚才提到{value}，只是随口说说。',
                            '我只是在转述{subject}说的话，现有记录照旧。'), TRAIN_NAMES[:12], '永齐茹立斐容'))
    parent = choices(TRAIN_NAMES, TRAIN, '荣安云庆岚薇青岳东梦')
    indices = torch.randperm(len(parent), generator=torch.Generator().manual_seed(SEED))[:len(mixed)]
    heads = {}
    progress.mkdir(parents=True, exist_ok=True)
    for label, extras in (('matched', [parent[i] for i in indices.tolist()]), ('corrections', mixed)):
        rows = [*parent, *extras]
        head = JointNameRoleHead(width)
        path = progress / (label + '.pt')
        signature = {'format': FORMAT, 'sources': sources, 'label': label, 'count': len(rows), 'epochs': EPOCHS,
                     'feature_batch': FEATURE_BATCH}
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if saved['signature'] != signature:
                raise ValueError('Partial checkpoint does not match this training job')
            head.load_state_dict(saved['weights'], strict=True)
        else:
            head.load_state_dict(state, strict=True)
            labelled, inputs, present = training_features(rows, width, featurize)
            fit(head, inputs, present, labelled, boundary_weight=1., epochs=EPOCHS)
            del labelled, inputs, present
            atomic_save({'signature': signature, 'weights': head.state_dict()}, path)
        heads[label] = head.eval().requires_grad_(False)
        print('Correction continuation', label, len(rows), len(extras), flush=True)
    return heads, len(parent), len(mixed)


def review(rows, featurize, configs):
    hidden, lengths = hidden_batch(featurize, rows)
    annotated = [row['mentions'] for row in rows]
    keys = dict.fromkeys((proposer, threshold) for proposer, _, threshold, _ in configs.values())
    candidates = {(proposer, threshold): annotated if threshold is None else
                  proposals(proposer, hidden, lengths, threshold) for proposer, threshold in keys}
    scores = {}
    for name, (proposer, role, threshold, margin_floor) in configs.items():
        spans = candidates[proposer, threshold]
        ranked = score_requests(role, rows, hidden, spans)
        reviewed = summarize_requests(rows, spans, ranked, margin_floor)
        picked = [span if margin > margin_floor else None for span, margin, _ in ranked]
        role_correct = sum(span == row['semantic_target'] for row, span in zip(rows, picked, strict=True))
        scores[name] = {'span_metrics': span_metrics(rows, spans), 'role_correct': role_correct, **reviewed}
        print(name, role_correct, '/', len(rows), reviewed['summary'], flush=True)
    return scores


def main(checkpoint, addresses, prior, proposal, output, verify):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    paths = {'checkpoint': checkpoint, 'addresses': addresses, 'prior': prior, 'proposal': proposal / 'proposers.pt',
             'proposal_report': proposal / 'result.json'}
    hashes = {name: sha(path) for name, path in paths.items()}
    base = torch.load(addresses, map_location='cpu', weights_only=True)
    if base['format'] != 'spora-entity-field-address-probe-v1' or base['checkpoint_sha256'] != hashes['checkpoint']:
        raise ValueError('Field address weights do not match the model')
    hashes['individual'] = base['individual_sha256']
    root = torch.load(prior, map_location='cpu', weights_only=True)
    source = {name: hashes[name] for name in ('checkpoint', 'addresses', 'individual')}
    if root['format'] != ROLE_FORMAT or root['sources'] != source:
        raise ValueError('Role weights do not match the individual')
    candidate = torch.load(paths['proposal'], map_location='cpu', weights_only=True)
    conditions = source | {'prior': hashes['prior']}
    if candidate['format'] != PROPOSER_FORMAT or any(candidate['sources'].get(k) != v for k, v in conditions.items()):
        raise ValueError('Name extractor weights do not match the role experiment')
    metadata = json.loads(paths['proposal_report'].read_text(encoding='utf-8'))
    if metadata['format'] != PROPOSER_FORMAT or metadata['sources'] != candidate['sources']:
        raise ValueError('Name extraction report does not match its weights')
    floor = metadata['thresholds']['mixed']
    resumed = torch.load(verify / 'roles.pt', map_location='cpu', weights_only=True) if verify else None
    if resumed is not None and (resumed['format'] != FORMAT or resumed['sources'] != hashes):
        raise ValueError('Adapted role weights do not match this experiment')

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    runtime.decoder.eval().requires_grad_(False)
    if frozen_digest(runtime.organism, runtime.state) != hashes['individual']:
        raise ValueError('Field addresses belong to a different individual')
    featurize = FrozenTurnFeatures(runtime.decoder, tokenizer)
    proposer = NameSpanHead(base['hidden']).eval().requires_grad_(False)
    proposer.load_state_dict(candidate['heads']['mixed'], strict=True)
    original = JointNameRoleHead(base['hidden']).eval().requires_grad_(False)
    original.load_state_dict(root['role'], strict=True)
    if resumed is None:
        progress = output.with_name(output.name + '-progress')
        adapted, shared, changed = train_heads(root['role'], base['hidden'], featurize, progress, hashes)
        training = {'shared': shared, 'changed': changed, 'epochs': EPOCHS}
    else:
        training, adapted = resumed['training'], {}
        for name, state in resumed['heads'].items():
            head = JointNameRoleHead(base['hidden']).eval().requires_grad_(False)
            head.load_state_dict(state, strict=True)
            adapted[name] = head
    profiles = {'prior': original, **adapted}
    configs = {name + '_annotated': (proposer, head, None, 0.) for name, head in profiles.items()}
    configs.update({name + '_proposed': (proposer, head, floor, 0.) for name, head in profiles.items()})
    held = choices(NEW_NAMES, HELD_CORRECTIONS, '严章桃谷河生')
    held.extend(no_change(('{subject}和{value}的近况是闲聊，清单先照原样。',
                           '先不调整{subject}的安排，我只是和你说一声。'), NEW_NAMES, '严章桃谷河生'))
    role_results = review(held, featurize, configs)
    challenge = review(choices(CHALLENGE_NAMES, CHALLENGE, '延竹鸿甫初泉'), featurize, configs)
    joint = JointAddress(base['hidden']).eval().requires_grad_(False)
    joint.load_state_dict(base['heads']['JointAddress'], strict=True)
    replay = examples(HELD_NAMES, HELD_TEXT, HELD_VALUES, SEED + 1)
    slots = replay_proposer_profiles(replay, featurize, configs, joint, tokenizer)
    for name, row in slots.items():
        print('Slots', name, row['summary'], flush=True)
    if (any(sha(path) != hashes[name] for name, path in paths.items())
            or frozen_digest(runtime.organism, runtime.state) != hashes['individual']):
        raise ValueError('One of the source weights or the individual changed')
    record = {'format': FORMAT, 'sources': hashes, 'training': training, 'proposal_floor': floor,
              'corrections': role_results, 'challenge': challenge, 'slot_replay': slots}
    output.mkdir(parents=True, exist_ok=False)
    if resumed is None:
        atomic_save({'format': FORMAT, 'sources': hashes, 'training': training,
                     'heads': {name: head.state_dict() for name, head in adapted.items()}}, output / 'roles.pt')
        store_json(output / 'result.json', record)
    else:
        reference = json.loads((verify / 'result.json').read_text(encoding='utf-8'))
        if json.loads(json.dumps(record)) != reference:
            raise ValueError('Reloaded results differ from the original run')
        store_json(output / 'result.json', {'identical': True, 'sources': hashes})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'addresses', 'prior', 'proposal', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--verify', type=Path)
    main(**vars(parser.parse_args()))
