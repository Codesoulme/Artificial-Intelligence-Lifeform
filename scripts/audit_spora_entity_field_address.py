"""Test learned Chinese entity-field addresses under repeated disjoint corrections."""

import argparse
from itertools import product
import json
from pathlib import Path
import random

from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_recurrent_attention import (
    MODEL, AutoTokenizer, frozen_digest, load_full_streamed, load_integrated_runtime, sha, torch,
)
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.identity_fingerprints import IdentityFingerprints
from spora.language_training import atomic_save
from spora.research_entity_field_address import EntityFieldAddress, RecurrentFieldSlots, TokenValue, revise_association
from torch import nn
from torch.nn import functional as F


TRAIN_NAMES = ('程梦宁', '程梦琳', '卢泊远', '卢泊川', '梅初月', '梅初晴', '张庭轩', '张庭禾',
               '金云林', '周书亭', '宁秀语', '喻霁川', '苗岱峰', '苏闻雨', '孔立冬', '陆思桥',
               '左清洲', '水青川', '向怀安', '孟修远', '尤雪竹', '宋明川', '唐谷雨', '朱晚舟')
HELD_NAMES = ('林嘉文', '林嘉雯', '林佳文', '林嘉涵', '池令海', '池令涵', '池凌海', '池霖海',
              '岑熠柏', '岑熠栢', '岑翌柏', '岑翼柏', '余鸿志', '余洪志', '余宏智', '俞鸿志',
              '邵闻箫', '邵闻萧', '邵文箫', '邵文萧')
FIELDS = ('place', 'quantity', 'contact', 'time')
TRAIN_TEXT = (
    ('{name}的集合地点定为{value}。', '请更新{name}的场地记录，改去{value}。', '{name}现在要到哪个地点集合？'),
    ('{name}登记的数量是{value}。', '把{name}项下的件数改成{value}。', '{name}的数量栏现在写多少？'),
    ('{name}这一项由{value}联络。', '请把{name}这条的对接人改为{value}。', '谁是{name}这一项的新联络人？'),
    ('{name}预约时间是{value}。', '{name}的约定时刻调整到{value}。', '{name}现在定在几点？'),
)
HELD_TEXT = (
    ('碰面时请{name}去{value}。', '换个地址，{name}到{value}和大家碰头。', '轮到接{name}，该去哪儿找？'),
    ('给{name}预留{value}材料。', '重新点过了，留给{name}的是{value}。', '出库单上，给{name}的这批有多少？'),
    ('关于{name}的安排，后续有事找{value}。', '涉及{name}的事，以后请和{value}沟通。', '现在有事要问{name}这条安排，该联系谁？'),
    ('请{name}在{value}出发。', '{name}要改到{value}再出门。', '最后定下{name}什么时候动身？'),
)
TRAIN_VALUES = (
    ('东岸月台', '玉竹长廊', '白墙六号厅', '松岭剧场', '桂街书屋', '红砖外廊'),
    ('三卷', '五套', '十一盒', '十二包', '两枚', '九捆'),
    ('傅行远', '钟暮秋', '关惊鸿', '叶芷庭', '闻雨萌', '莫晓诗'),
    ('周六上午八点', '周五下午两点', '今天中午', '明早六点半', '九点十五分', '星期日下午'),
)
HELD_VALUES = (
    ('杉叶工坊', '环形柱廊', '小槐展室', '远山南馆', '群雀天井', '蓝釉车棚'),
    ('二十六册', '十四捆', '十六联', '二十二根', '四十五支', '十八袋'),
    ('殷望竹', '何谷澄', '褚静棠', '韩未央', '蒋露清', '朱怀月'),
    ('凌晨三点十分', '星期四下午一点半', '正午十二点四十', '下周一晚上七点', '傍晚六点四十五', '周三清晨五点'),
)
SEED = 241124


class JointAddress(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.projection = nn.Linear(2 * hidden, 128, bias=False)

    def forward(self, entity, context):
        return F.normalize(self.projection(torch.cat((entity, context), dim=-1)), dim=-1)


def examples(names, patterns, values, seed):
    rng = random.Random(seed)
    rows = []
    for name, field in product(names, range(len(FIELDS))):
        first, update = rng.sample(values[field], 2)
        texts = tuple(template.format(name=name, value=value)
                      for template, value in zip(patterns[field], (first, update, ''), strict=True))
        rows.append({'name': name, 'field': FIELDS[field], 'values': (first, update), 'texts': texts})
    return rows


def representations(features, rows):
    texts = [text for row in rows for text in row['texts']]
    hidden = features(texts).float()
    entities, contexts = [], []
    for i, text in enumerate(texts):
        name = rows[i // 3]['name']
        if text.count(name) != 1:
            raise ValueError('Expected a unique name for the boundary diagnostic')
        start = text.index(name)
        end = start + len(name)
        entities.append(hidden[i, start:end].mean(dim=0))
        contexts.append(hidden[i, :len(text)].mean(dim=0))
    return tuple(F.normalize(torch.stack(items), dim=-1).reshape(len(rows), 3, -1)
                 for items in (entities, contexts))


def similarity_loss(keys):
    # Compare each paraphrase with every address represented in the other two views.
    labels = torch.arange(keys.shape[0])
    loss = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        logits = keys[:, first] @ keys[:, second].T * 12.
        loss.append((F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2)
    return torch.stack(loss).mean()


def fit(head, inputs, steps):
    optimizer = torch.optim.AdamW(head.parameters(), lr=.004, weight_decay=.03)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = similarity_loss(head(*inputs))
        loss.backward()
        optimizer.step()
        if step in (0, steps - 1):
            print('Address training', type(head).__name__, step + 1, float(loss.detach()), flush=True)
    return head.eval().requires_grad_(False)


def geometry(keys, rows):
    actual = keys[:, 2] @ keys[:, 0].T
    count = len(rows)
    target = torch.arange(count)
    same_entity = torch.tensor([[a['name'] == b['name'] for b in rows] for a in rows])
    same_field = torch.tensor([[a['field'] == b['field'] for b in rows] for a in rows])
    diagonal = actual.diagonal()
    correct = actual.argmax(dim=-1) == target
    return {'top1': int(correct.sum()), 'count': count, 'same_key_cosine': float(diagonal.mean()),
            'update_key_cosine': float((keys[:, 1] * keys[:, 0]).sum(dim=-1).mean()),
            'beats_other_field': int((diagonal > actual.masked_fill(~(same_entity & ~same_field), -torch.inf)
                                      .amax(dim=-1)).sum()),
            'beats_other_entity': int((diagonal > actual.masked_fill(~(~same_entity & same_field), -torch.inf)
                                       .amax(dim=-1)).sum())}


def replay(keys, rows, tokenizer):
    codec = TokenValue(len(tokenizer))
    rows_per_group = 4 * len(FIELDS)
    count = len(rows)
    if count % rows_per_group:
        raise ValueError('Every replay requires four entities with all fields')
    events = []
    for start in range(0, count, rows_per_group):
        end = start + rows_per_group
        group = rows[start:end]
        values = [[tokenizer.encode(text, add_special_tokens=False) for text in row['values']] for row in group]
        memory = torch.zeros(keys.shape[-1], codec.width)
        # Touch one field per entity, and rotate which field is changed in each group.
        changed = {owner * len(FIELDS) + (owner + start // rows_per_group) % len(FIELDS) for owner in range(4)}
        for step in range(2):
            positions = range(rows_per_group) if step == 0 else sorted(changed, reverse=True)
            for i in positions:
                memory = revise_association(memory, keys[start + i, step], codec.encode(values[i][step]))
            found = [codec.decode(vector) for vector in keys[start:end, 2] @ memory]
            for i, read in enumerate(found):
                expected = values[i][int(step == 1 and i in changed)]
                events.append({'name': group[i]['name'], 'field': group[i]['field'], 'changed': i in changed,
                               'after_update': step == 1, 'correct': read == expected,
                               'expected': group[i]['values'][int(step == 1 and i in changed)],
                               'read': None if read is None else tokenizer.decode(read)})
    summary = {}
    for name, selected in (
        ('initial', [row for row in events if not row['after_update']]),
        ('revised', [row for row in events if row['after_update'] and row['changed']]),
        ('retained', [row for row in events if row['after_update'] and not row['changed']]),
    ):
        summary[name] = {'correct': sum(row['correct'] for row in selected), 'count': len(selected)}
    summary['matrix_bytes'] = memory.numel() * memory.element_size()
    return {'summary': summary, 'rows': events}


def slot_replay(keys, rows, tokenizer, *, floor, gap, confirm_name=False):
    codec = TokenValue(len(tokenizer))
    per_group = len(FIELDS) * 4
    if len(rows) < 2 * per_group or len(rows) % per_group or len(keys) != len(rows):
        raise ValueError('Expected two or more complete and aligned replay groups')
    scored = dict.fromkeys(('writes_correct', 'writes_wrong', 'writes_skipped', 'revised', 'retained', 'unknown_skipped'), 0)
    details = []
    for start in range(0, len(rows), per_group):
        stop = start + per_group
        addresses = keys[start:stop]
        group = rows[start:stop]
        slots = RecurrentFieldSlots(per_group, keys.shape[-1], codec.width)
        identities = IdentityFingerprints(4) if confirm_name else None
        if identities is not None:
            for owner, record in enumerate(group[::len(FIELDS)]):
                identities.bind(owner, record['name'])

        def scope(name):
            owner = identities.lookup(name) if identities is not None else None
            if owner is None:
                return None if identities is not None else slice(None)
            return slice(owner * len(FIELDS), (owner + 1) * len(FIELDS))

        values = [[tokenizer.encode(text, add_special_tokens=False) for text in row['values']] for row in group]
        for item, (key, _, _) in zip(values, addresses, strict=True):
            slots.allocate(key, codec.encode(item[0]))
        changed = {owner * len(FIELDS) + (owner + start // per_group) % len(FIELDS) for owner in range(4)}
        for index in sorted(changed, reverse=True):
            selected = slots.revise(addresses[index, 1], codec.encode(values[index][1]), floor=floor, gap=gap,
                                    eligible=scope(group[index]['name']))
            outcome = 'writes_skipped' if selected is None else 'writes_correct' if selected == index else 'writes_wrong'
            scored[outcome] += 1
            details.append({'group': start // per_group, 'requested_slot': index, 'written_slot': selected})
        for index, query in enumerate(addresses[:, 2]):
            selected = slots.select(query, floor=floor, gap=gap, eligible=scope(group[index]['name']))
            tokens = codec.decode(slots.values[selected]) if selected is not None else None
            revised = index in changed
            scored['revised' if revised else 'retained'] += selected == index and tokens == values[index][int(revised)]
        foreign = stop % len(rows)
        for query, row in zip(keys[foreign:foreign + per_group, 2], rows[foreign:foreign + per_group], strict=True):
            eligible = scope(row['name'])
            abstained = eligible is None or slots.select(query, floor=floor, gap=gap, eligible=eligible) is None
            scored['unknown_skipped'] += abstained
    identity_bytes = 0 if identities is None else sum(t.numel() * t.element_size() for t in identities.buffers())
    slot_bytes = sum(t.numel() * t.element_size() for t in (slots.addresses, slots.values))
    return {'counts': scored, 'updates': len(rows) // len(FIELDS), 'unmodified': len(rows) * 3 // 4,
            'unknown': len(rows), 'tensor_bytes': slot_bytes + identity_bytes, 'writes': details}


def main(args):
    seed = SEED
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    starting_sha = sha(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(args.checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    runtime.decoder.eval().requires_grad_(False)
    initial_digest = frozen_digest(runtime.organism, runtime.state)
    features = FrozenTurnFeatures(runtime.decoder, tokenizer)
    held = examples(HELD_NAMES, HELD_TEXT, HELD_VALUES, seed + 1)
    held_inputs = representations(features, held)
    path = args.output / 'addresses.pt'
    format_name = 'spora-entity-field-address-probe-v1'
    if args.verify or args.route:
        output = json.loads((args.output / 'result.json').read_text(encoding='utf-8'))
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if (saved['format'] != format_name or saved['checkpoint_sha256'] != starting_sha
                or saved['individual_sha256'] != initial_digest or output['addresses_sha256'] != sha(path)
                or saved['hidden'] != held_inputs[0].shape[-1]):
            raise ValueError('The address checkpoint does not belong to this individual and model')
    else:
        train = examples(TRAIN_NAMES, TRAIN_TEXT, TRAIN_VALUES, seed)
        train_inputs = representations(features, train)
        print('Frozen Chinese features captured:', len(train), len(held), flush=True)
        output = {'checkpoint_sha256': starting_sha, 'individual_sha256': initial_digest, 'train': len(train),
                  'held': len(held), 'steps': args.steps, 'arms': {}}
    heads, routing = {}, {}
    for architecture in (JointAddress, EntityFieldAddress):
        torch.manual_seed(seed)
        name = architecture.__name__
        head = architecture(held_inputs[0].shape[-1])
        if args.verify or args.route:
            head.load_state_dict(saved['heads'][name], strict=True)
            head.eval().requires_grad_(False)
        else:
            head = fit(head, train_inputs, args.steps)
        address = head(*held_inputs).detach()
        result = {'geometry': geometry(address, held), 'replay': replay(address, held, tokenizer),
                  'parameters': sum(weight.numel() for weight in head.parameters())}
        if args.verify or args.route:
            if result != output['arms'][name]:
                raise ValueError(f'The reopened {name} did not reproduce address or replay results')
            if args.route:
                routing[name] = {'nearest': slot_replay(address, held, tokenizer, floor=0., gap=0.),
                                 'gated': slot_replay(address, held, tokenizer, floor=.30, gap=.04),
                                 'confirmed_name': slot_replay(address, held, tokenizer, floor=0., gap=0., confirm_name=True)}
        else:
            output['arms'][name] = result
        heads[name] = head.state_dict()
        print(name, result['geometry'], result['replay']['summary'], flush=True)
    if sha(args.checkpoint) != starting_sha or frozen_digest(runtime.organism, runtime.state) != initial_digest:
        raise ValueError('The language checkpoint or individual changed')
    if args.route:
        store_json(args.route_output or args.output / 'routing.json', {'addresses_sha256': sha(path), 'arms': routing})
        for name, arms in routing.items():
            print(name, {mode: replay['counts'] for mode, replay in arms.items()}, flush=True)
        return
    if args.verify:
        store_json(args.output / 'reload.json', {'identical': True, 'held': len(held), 'architecture_count': len(heads)})
        return
    atomic_save({'format': format_name, 'heads': heads,
                 'checkpoint_sha256': starting_sha, 'individual_sha256': initial_digest,
                 'hidden': held_inputs[0].shape[-1]}, path)
    output['addresses_sha256'] = sha(path)
    store_json(args.output / 'result.json', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--steps', type=int, default=300)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--verify', action='store_true')
    modes.add_argument('--route', action='store_true', help='Reopen a saved address head and measure bounded writes')
    parser.add_argument('--route-output', type=Path)
    options = parser.parse_args()
    if options.steps < 1:
        parser.error('Choose a positive training budget')
    if options.route_output and not options.route:
        parser.error('--route-output requires --route')
    if not (options.verify or options.route):
        options.output.mkdir(parents=True, exist_ok=False)
    main(options)
