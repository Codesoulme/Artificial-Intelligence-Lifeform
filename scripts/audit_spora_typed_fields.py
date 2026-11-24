"""Learn field routing independently of the registered subject."""

import argparse
import json
from pathlib import Path
import random

from scripts.audit_spora_correction_roles import FORMAT as ROLE_FORMAT
from scripts.audit_spora_entity_field_address import TRAIN_TEXT, TRAIN_VALUES
from scripts.audit_spora_role_decisions import explicit_events
from scripts.train_spora_open_name_roles import (
    HELD_NAMES, HELD_TEXT, HELD_VALUES, MODEL, SEED, TRAIN_NAMES, AutoTokenizer, FrozenTurnFeatures, JointAddress,
    JointNameRoleHead, atomic_save, examples, frozen_digest, hidden_batch, load_full_streamed,
    load_integrated_runtime, resolve_rows, sha, store_json, torch,
)
from scripts.train_spora_name_role import FIELDS, TokenValue, measure, observations, representations, summarize
from scripts.train_spora_open_name_roles import annotate_group
from spora.research_entity_field_address import FieldReadout, field_slot_keys
from torch.nn import functional as F


FORMAT = 'spora-typed-fields-v1'
FIELD_QUERY = (
    ('{name}要和{peer}碰面，帮我把见面的地方改到{value}。',
     '和{peer}沟通过了，给{name}记的新位置是{value}。', '登记薄上{name}最终是在哪儿会面？'),
    ('{name}那箱材料点错了，按{value}登记。',
     '和{peer}对过数了，{name}实际分到的是{value}。', '给{name}备了多少份材料？'),
    ('地点已经定好，{name}的事有问题请找{value}。',
     '接应位置照旧，{name}这项换由{value}负责沟通。', '请查{name}要找谁来联系。'),
    ('{peer}问过了，{name}的预约改为{value}。',
     '见面地点照旧，{name}的新时间是{value}。', '记录中{name}到底什么时候到？'),
)
TEST_NAMES = ('纳嘉平', '裴望舒', '陶雪岚', '宋竹晖', '莫玉容', '潘凡清', '翁松溪', '施洁妍')
HELD_FIELD_QUERY = (
    ('会合的人是{peer}，麻烦更新{name}那单，改在{value}碰面。',
     '请把{name}的等候点转到{value}，仍找{peer}衔接。', '到哪里才能接到{name}？'),
    ('请把{name}确认的材料规格保留，件数换成{value}。',
     '为{name}做登记，数量请按{value}填写。', '现在发给{name}的材料有多少？'),
    ('{name}会面的地方照原样，负责交接的人改为{value}。',
     '地点由{peer}照看，{name}那项的新接洽人是{value}。', '关于{name}那项，谁来负责接洽？'),
    ('把{name}的就位时间改成{value}，等候区照原安排。',
     '地点没动，{name}要推到{value}开始。', '最迟什么时间要等{name}到场？'),
)
FIELD_PHRASES = ('碰头地点', '材料件数', '负责联络的人', '预约的时段')
HELD_PHRASES = ('接人的位置', '发放的份数', '办事联系人', '到场时刻')
ABSTAIN = ('给{name}改一下记录，具体改哪栏我再想想。',
           '先别改{name}的内容，地点和{peer}的联络信息都照旧。',
           '核对{name}的表，请把地点和联系人一并换掉。')
HELD_ABSTAIN = ('{name}那项该动哪里还没定，等我确认再填。',
                '{name}和{peer}刚讨论过地点，我只转述一下消息。',
                '{name}的时间和领用数量都需要重新登记。')


def data(names, patterns, values, phrases, seed, *, include_base):
    rng = random.Random(seed)
    rows = []
    pair_pattern = ('{name}的单子上有{peer}接应，写了地点、联络人、时间和数量。这次只更新{field}。' if include_base else
                    '{peer}提到了{name}的会合地址、接洽对象、到场时间和份数。改动范围只有{field}。')
    for i, name in enumerate(names):
        peer = names[(i + 1) % len(names)]
        for field, prompts in enumerate(patterns):
            templates = TRAIN_TEXT[field] + prompts if include_base else prompts
            for pattern in templates:
                text = pattern.format(name=name, peer=peer, value=rng.choice(values[field]))
                rows.append({'text': text, 'label': field, 'kind': 'paraphrase'})
            pair = pair_pattern.format(name=name, peer=peer, field=phrases[field])
            rows.append({'text': pair, 'label': field, 'kind': 'paired'})
        for pattern in ABSTAIN if include_base else HELD_ABSTAIN:
            rows.append({'text': pattern.format(name=name, peer=peer), 'label': len(FIELDS), 'kind': 'abstain'})
    return rows


def features(rows, featurize, width):
    padded = torch.zeros(len(rows), max(len(row['text']) for row in rows), width, dtype=torch.float16)
    lengths = torch.tensor([len(row['text']) for row in rows])
    for first in range(0, len(rows), 64):
        hidden, _ = hidden_batch(featurize, rows[first:first + 64])
        padded[first:first + len(hidden), :hidden.shape[1]].copy_(hidden)
    return padded, torch.arange(padded.shape[1])[None, :] < lengths[:, None]


def fit(head, hidden, mask, rows):
    target = torch.tensor([row['label'] for row in rows])
    rng = torch.Generator().manual_seed(SEED)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.004, weight_decay=.03)
    for _ in range(10):
        for indices in torch.randperm(len(rows), generator=rng).split(64):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(hidden[indices], mask[indices]), target[indices])
            loss.backward()
            optimizer.step()
    return head.eval().requires_grad_(False)


def scores(head, hidden, lengths):
    mask = torch.arange(hidden.shape[1])[None, :] < lengths[:, None]
    return head(hidden, mask)


def audit_fields(head, rows, batch):
    predictions = head(*batch).argmax(-1).tolist()
    labels = (*FIELDS, 'abstain')
    results = [{**row, 'prediction': labels[chosen], 'correct': chosen == row['label']}
               for row, chosen in zip(rows, predictions, strict=True)]
    summary = {}
    for kind in ('paraphrase', 'paired', 'abstain'):
        selected = [row for row in results if row['kind'] == kind]
        summary[kind] = {'correct': sum(row['correct'] for row in selected), 'count': len(selected)}
    return {'summary': summary, 'rows': results}


def replay(held, featurize, tokenizer, heads, role, joint):
    size = len(FIELDS) * 4
    groups = [held[start:start + size] for start in range(0, len(held), size)]
    result = {}
    codec = TokenValue(len(tokenizer))
    seed_fields = torch.eye(len(FIELDS)).repeat(4, 1)
    for index, group in enumerate(groups):
        names = [row['name'] for row in group[::len(FIELDS)]]
        foreign = [row['name'] for row in groups[(index + 1) % len(groups)][::len(FIELDS)]]
        original = list(observations(group, foreign, index))
        initial = joint(*representations(featurize, group))[:, 0]
        for variant, events in (('original', original), ('explicit', explicit_events(group, foreign, names, original))):
            rows = annotate_group(group, foreign, events, names)
            hidden, lengths = hidden_batch(featurize, rows)
            spans = [row['mentions'] for row in rows]
            routed = [address for _, address in resolve_rows(role, rows, hidden, spans)]
            confirmed = [next(item for item in row['candidates'] if item[2] == position // len(FIELDS))
                         if kind in ('update', 'query') else address
                         for row, (kind, position, _), address in zip(rows, events, routed, strict=True)]
            keys = {name: field_slot_keys(scores(head, hidden, lengths)) for name, head in heads.items()}
            for mode, mentions in (('role', routed), ('confirmed', confirmed)):
                for name, addresses in (('joint', None), *keys.items()):
                    profile = '_'.join((variant, mode, name))
                    details = measure(group, events, mentions, hidden.float(), initial if addresses is None else seed_fields,
                                      codec, tokenizer, joint, profile, field_keys=addresses,
                                      selector=lambda _mode, _registry, _text, chosen, _gold: chosen)
                    result.setdefault(profile, []).extend(details)
    return {name: {'summary': summarize(rows), 'rows': rows} for name, rows in result.items()}


def main(checkpoint, addresses, roles, output, verify):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    paths = {'checkpoint': checkpoint, 'addresses': addresses, 'roles': roles}
    hashes = {name: sha(path) for name, path in paths.items()}
    base = torch.load(addresses, map_location='cpu', weights_only=True)
    if base['format'] != 'spora-entity-field-address-probe-v1' or base['checkpoint_sha256'] != hashes['checkpoint']:
        raise ValueError('Field address weights do not match this language model')
    hashes['individual'] = base['individual_sha256']
    learned = torch.load(roles, map_location='cpu', weights_only=True)
    if learned['format'] != ROLE_FORMAT or any(learned['sources'][key] != hashes[key] for key in
                                                ('checkpoint', 'addresses', 'individual')):
        raise ValueError('Role weights belong to a different model or individual')
    stored = torch.load(verify / 'fields.pt', map_location='cpu', weights_only=True) if verify else None
    if stored is not None and (stored['format'] != FORMAT or stored['sources'] != hashes):
        raise ValueError('Field readout weights belong to a different experiment')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    runtime.decoder.eval().requires_grad_(False)
    if frozen_digest(runtime.organism, runtime.state) != hashes['individual']:
        raise ValueError('Individual digest mismatch')
    featurize = FrozenTurnFeatures(runtime.decoder, tokenizer)
    width = base['hidden']
    role = JointNameRoleHead(width).eval().requires_grad_(False)
    role.load_state_dict(learned['heads']['matched'], strict=True)
    joint = JointAddress(width).eval().requires_grad_(False)
    joint.load_state_dict(base['heads']['JointAddress'], strict=True)
    training = data(TRAIN_NAMES, FIELD_QUERY, TRAIN_VALUES, FIELD_PHRASES, SEED, include_base=True)
    evaluation = data(TEST_NAMES, HELD_FIELD_QUERY, HELD_VALUES, HELD_PHRASES, SEED + 1, include_base=False)
    train_batch = features(training, featurize, width) if stored is None else None
    held_batch = features(evaluation, featurize, width)
    heads, readout = {}, {}
    for name, attention in (('mean', False), ('attention', True)):
        torch.manual_seed(SEED)
        head = FieldReadout(width, len(FIELDS), attention=attention)
        if stored is None:
            fit(head, *train_batch, training)
        else:
            head.load_state_dict(stored['heads'][name], strict=True)
            head.eval().requires_grad_(False)
        heads[name] = head
        readout[name] = audit_fields(head, evaluation, held_batch)
        print('Field readout', name, readout[name]['summary'], flush=True)
    del train_batch, held_batch
    held = examples(HELD_NAMES, HELD_TEXT, HELD_VALUES, SEED + 1)
    slots = replay(held, featurize, tokenizer, heads, role, joint)
    for name, report in slots.items():
        print('Slots', name, report['summary'], flush=True)
    if (any(sha(path) != hashes[name] for name, path in paths.items())
            or frozen_digest(runtime.organism, runtime.state) != hashes['individual']):
        raise ValueError('Source weights or the individual changed')
    record = {'format': FORMAT, 'sources': hashes, 'training': len(training), 'readout': readout, 'slots': slots}
    output.mkdir(parents=True, exist_ok=False)
    if stored is None:
        atomic_save({'format': FORMAT, 'sources': hashes, 'heads': {name: head.state_dict() for name, head in heads.items()}},
                    output / 'fields.pt')
        store_json(output / 'result.json', record)
    else:
        reference = json.loads((verify / 'result.json').read_text(encoding='utf-8'))
        if json.loads(json.dumps(record)) != reference:
            raise ValueError('Reloaded field readout did not reproduce the audit')
        store_json(output / 'result.json', {'identical': True, 'sources': hashes})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'addresses', 'roles', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--verify', type=Path)
    main(**vars(parser.parse_args()))
