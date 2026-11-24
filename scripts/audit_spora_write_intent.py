"""Measure an independent eligibility decision before committing a single-field correction."""

import argparse
import json
from pathlib import Path

from scripts.audit_spora_cluener_name_spans import FORMAT as PROPOSER_FORMAT
from scripts.audit_spora_correction_roles import FORMAT as ROLE_FORMAT
from scripts.audit_spora_role_decisions import explicit_events
from scripts.audit_spora_typed_fields import FORMAT as FIELD_FORMAT, FieldReadout, features, scores
from scripts.train_spora_name_role import FIELDS, TokenValue, measure, observations, summarize
from scripts.train_spora_open_name_roles import (
    HELD_NAMES, HELD_TEXT, HELD_VALUES, MODEL, SEED, TRAIN_NAMES, AutoTokenizer, FrozenTurnFeatures, JointNameRoleHead,
    NameSpanHead, annotate_group, atomic_save, examples, frozen_digest, hidden_batch, load_full_streamed,
    load_integrated_runtime, proposals, resolve_rows, sha, store_json, torch,
)
from spora.research_entity_field_address import field_slot_keys
from torch import nn
from torch.nn import functional as F


FORMAT = 'spora-single-write-intent-v2'
FIELD_NAMES = ('会合地点', '材料数量', '对接联系人', '预约时间')
TRAIN_VALUES = ('南侧廊亭', '二十二箱', '陈柏秋', '周四下午两点')
TEST_VALUES = ('北门窗旁', '十九包', '吴雪初', '星期一中午')
CAL_NAMES = ('匡世平', '关若安', '潘新湘', '苗采文', '庄秋实', '赖怀周')
TEST_NAMES = ('娄闻珂', '路青扬', '汲守明', '吕明舟', '韦启涵', '任雨白', '史文冬', '薛庭深')
TRAIN_INTENT = (
    ('write', '按确认结果执行：{name}的{field}改成{value}。'),
    ('write', '这次核对过了，{name}应该是{value}，请覆盖{field}的旧内容。'),
    ('write', '{name}的{field}最终定为{value}，别再用原来的。'),
    ('read', '帮我查查，{name}的{field}已经是{value}了吗？'),
    ('read', '{name}的{field}写的是不是{value}？先看一下存档。'),
    ('pending', '如果确定{value}，再考虑改{name}的{field}，先不要记。'),
    ('pending', '{name}的{field}可能是{value}，目前还没有落定。'),
    ('keep', '{name}提到{value}了，不过{field}依旧按旧记录走。'),
    ('keep', '撤回{name}的{field}改为{value}的要求，保持现有值。'),
    ('report', '我转述一句，有人建议{name}的{field}填{value}。'),
    ('report', '先聊一聊{name}如果把{field}改为{value}会怎么样。'),
    ('multi', '{name}的{field}换成{value}，{other}也要调整。'),
    ('multi', '给{name}一起重填{field}和{other}，其中一项写{value}。'),
    ('write', '{name}的新{field}是{value}，这次按新的办。'),
    ('write', '刚收到确认，关于{name}，{value}就是更新后的{field}。'),
    ('write', '{name}原来那条{field}有误，正确的是{value}。'),
    ('write', '有个改动，给{name}记的{field}现在以{value}为准。'),
    ('write', '替{name}留个记录：{field}用{value}。'),
    ('write', '{name}的{field}改到{value}，其余项目都保留。'),
    ('write', '关于{name}的安排，{field}已经改成{value}了。'),
    ('write', '已经定了，记下{name}的{field}是{value}。'),
    ('read', '关于{name}，{value}是不是最新的{field}？'),
    ('keep', '{name}的{field}写成{value}是旧提议，刚取消了。'),
)
CAL_INTENT = (
    ('write', '给{name}的表只改一处，{field}填入{value}。'),
    ('read', '请做个检查，{name}的{field}现在是不是{value}？'),
    ('pending', '还没定{name}的{field}是否用{value}，晚一点再答复。'),
    ('keep', '把{name}的{field}改成{value}那条当作没说。'),
    ('report', '{name}曾说过想用{value}作{field}。我只是转告当时的话。'),
    ('multi', '{name}的{field}先按{value}，此外还要一并变动{other}。'),
)
HELD_INTENT = (
    ('write', '改动已经敲定，把{name}表里的{field}换为{value}。'),
    ('write', '{name}这次修正的是{field}，新登记值用{value}，其余照旧。'),
    ('read', '{name}目前的{field}是{value}，还是另有记录？'),
    ('read', '我想核查{name}那条，里面的{field}填的可否是{value}？'),
    ('pending', '打算给{name}的{field}改成{value}，我还在等最终消息。'),
    ('pending', '先搁置{name}的{field}变成{value}这个选项，之后再决定。'),
    ('keep', '原本要给{name}录入{value}，现在取消，{field}留着原值。'),
    ('keep', '就算有人把{value}发过来，也请照旧保留{name}的{field}。'),
    ('report', '提起{name}的{field}时，有人说过{value}，这是会议上的原话。'),
    ('report', '我只是在念纸上的字：把{name}的{field}调整为{value}。'),
    ('multi', '在{name}的记录里一口气更新{field}和{other}，先用了{value}。'),
    ('multi', '本次除了{name}的{field}改为{value}，还必须同时动{other}。'),
)
REPLAY_INTENT = (
    ('read', '{name}的{field}标成{value}没有？麻烦查给我看。'),
    ('pending', '拟给{name}使用{value}作为{field}，请等我决定再录。'),
    ('keep', '先前让给{name}把{field}换为{value}的消息作废。'),
    ('report', '这是别人的提议，他们想让{name}的{field}用{value}。'),
    ('multi', '既要把{name}的{field}调为{value}，也需要立刻改{other}。'),
)
CONTRAST_INTENT = (
    ('write', '那就照这个办，{name}那份的{field}用{value}，其他别动。'),
    ('keep', '刚才说{name}那份的{field}用{value}先不算，旧表先留着。'),
    ('write', '{name}的{field}有了新准信，按{value}执行。'),
    ('pending', '{name}的{field}有人报了{value}，还要等通知落实。'),
    ('write', '我替{name}确认了，拿{value}替换原{field}。'),
    ('report', '他们在征求意见，问能不能拿{value}替换{name}原来的{field}。'),
    ('write', '给{name}更新{field}为{value}，只改这个。'),
    ('multi', '给{name}更新{field}为{value}，顺便把{other}一起更新。'),
)
SCOPE_INTENT = (
    ('write', '刚跟{name}对过，{field}就记{value}，已经确认。'),
    ('pending', '刚跟{name}提过，{field}也许用{value}，还没有确认。'),
    ('write', '{name}那张记录需要更正。最新的{field}确定为{value}。'),
    ('report', '上次讨论有人提出更正{name}那张记录，建议的{field}是{value}。'),
    ('write', '最新一条消息是{name}的{field}改用{value}，把这个变更记下。'),
    ('keep', '最新一条消息撤掉了{name}的{field}改用{value}的决定，原表有效。'),
    ('write', '{name}已定{value}。记录上只需要调整{field}，别的保持原样。'),
    ('multi', '{name}已定{value}。记录上除了{field}，{other}也得跟着调整。'),
    ('write', '{name}的{field}一律按{value}办，刚核实完。'),
    ('pending', '{name}的{field}也许可以按{value}办，还得等最后回复。'),
    ('write', '把{value}作为{name}新的{field}，我确认生效。'),
    ('report', '请听这句建议：把{value}作为{name}新的{field}。建议还在讨论。'),
)


def corpus(names, patterns, values):
    return [{'text': pattern.format(name=name, field=field, value=values[i], other=FIELD_NAMES[(i + 1) % len(FIELDS)]),
             'kind': kind, 'label': int(kind == 'write')}
            for name in names for i, field in enumerate(FIELD_NAMES) for kind, pattern in patterns]


def intent_margin(logits):
    return logits[:, 1] - logits[:, 0]


class RequestIntent(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.decision = nn.Sequential(nn.Linear(2 * width, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, hidden, mask):
        lengths = mask.sum(dim=-1).clamp_min(1)
        hidden = hidden.float()
        mean = hidden.masked_fill(~mask[..., None], 0.).sum(dim=1) / lengths[:, None]
        last = hidden[torch.arange(len(hidden), device=hidden.device), lengths - 1]
        return self.decision(torch.cat((F.normalize(mean, dim=-1), F.normalize(last, dim=-1)), dim=-1))


def fit_intent(head, hidden, mask, rows, *, epochs=10, learning_rate=.002):
    targets = torch.tensor([row['label'] for row in rows])
    weights = targets.bincount(minlength=2).float().reciprocal()
    generator = torch.Generator().manual_seed(SEED)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=.03)
    for _ in range(epochs):
        for indices in torch.randperm(len(rows), generator=generator).split(64):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(hidden[indices], mask[indices]), targets[indices], weight=weights)
            loss.backward()
            optimizer.step()
    return head.eval().requires_grad_(False)


def held_intent(rows, margins, floor):
    output = [{**row, 'margin': score, 'authorized': score > floor,
               'correct': (score > floor) == bool(row['label'])}
              for row, score in zip(rows, margins, strict=True)]
    summary = {kind: {'correct': sum(row['kind'] == kind and row['correct'] for row in output),
                      'count': sum(row['kind'] == kind for row in output)}
               for kind in dict.fromkeys(row['kind'] for row in output)}
    return {'summary': summary, 'rows': output}


def extra_events(group, group_index):
    output = []
    for position, row in enumerate(group):
        kind, pattern = REPLAY_INTENT[(position + group_index) % len(REPLAY_INTENT)]
        field = position % len(FIELDS)
        text = pattern.format(name=row['name'], field=FIELD_NAMES[field], value=row['values'][1],
                              other=FIELD_NAMES[(field + 1) % len(FIELDS)])
        output.append(('intent_' + kind, position, text))
    return output


def annotate_intents(group, foreign, names, events):
    # Extra intents share the positive update's name annotation contract, but are dry-run negatives.
    annotated_events = [('update' if kind.startswith('intent_') else kind, position, text)
                        for kind, position, text in events]
    return annotate_group(group, foreign, annotated_events, names)


def replay(held, featurize, tokenizer, field, role, proposer, proposal_floor, intent, calibrated, intent_features):
    size = len(FIELDS) * 4
    groups = [held[start:start + size] for start in range(0, len(held), size)]
    result = {}
    codec = TokenValue(len(tokenizer))
    initial = torch.eye(len(FIELDS)).repeat(4, 1)
    for group_index, group in enumerate(groups):
        names = [row['name'] for row in group[::len(FIELDS)]]
        foreign = [row['name'] for row in groups[(group_index + 1) % len(groups)][::len(FIELDS)]]
        original = list(observations(group, foreign, group_index))
        for variant, events in (('original', original), ('explicit', explicit_events(group, foreign, names, original))):
            requests = [*events, *extra_events(group, group_index)]
            rows = annotate_intents(group, foreign, names, requests)
            hidden, lengths = hidden_batch(featurize, rows)
            detected = proposals(proposer, hidden, lengths, proposal_floor)
            fields = field_slot_keys(scores(field, hidden, lengths))
            intent_hidden, intent_lengths = (hidden, lengths) if intent_features is featurize else hidden_batch(intent_features, rows)
            margins = intent_margin(scores(intent, intent_hidden, intent_lengths)).tolist()
            for origin, spans in (('annotated', [row['mentions'] for row in rows]), ('automatic', detected)):
                mentions = [candidate for _, candidate in resolve_rows(role, rows, hidden, spans)]
                for mode, floor in (('baseline', None), ('fixed', 0.), ('calibrated', calibrated)):
                    keys = [key if floor is None or kind == 'query' or margin > floor else None
                            for (kind, _, _), key, margin in zip(requests, fields, margins, strict=True)]
                    label = '_'.join((variant, origin, mode))
                    replayed = measure(group, requests, mentions, hidden.float(), initial, codec, tokenizer, None,
                                       label, field_keys=keys,
                                       selector=lambda _mode, _registry, _text, chosen, _gold: chosen)
                    result.setdefault(label, []).extend(replayed)
    output = {}
    for key, rows in result.items():
        extra = [row for row in rows if row['kind'].startswith('intent_')]
        intent_negative = {kind: {'sent_to_slot': sum(row['kind'] == kind and row['slot'] is not None for row in extra),
                                  'count': sum(row['kind'] == kind for row in extra)}
                           for kind in dict.fromkeys(row['kind'] for row in extra)}
        output[key] = {'summary': summarize(rows), 'intent_negatives': intent_negative, 'rows': rows}
        print(key, output[key]['summary'], intent_negative, flush=True)
    return output


def main(checkpoint, field, roles, proposal, output, verify, contrast, scope, intent_depth):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    paths = {'checkpoint': checkpoint, 'field': field, 'roles': roles, 'proposal': proposal / 'proposers.pt',
             'proposal_report': proposal / 'result.json'}
    hashes = {name: sha(path) for name, path in paths.items()}
    if contrast:
        hashes['contrast'] = True
    if scope:
        hashes['scope'] = True
    if intent_depth != 12:
        hashes['intent_depth'] = intent_depth
    field_data = torch.load(field, map_location='cpu', weights_only=True)
    role_data = torch.load(roles, map_location='cpu', weights_only=True)
    proposal_data = torch.load(paths['proposal'], map_location='cpu', weights_only=True)
    hashes['individual'] = field_data['sources']['individual']
    if (field_data['format'] != FIELD_FORMAT or role_data['format'] != ROLE_FORMAT or proposal_data['format'] != PROPOSER_FORMAT
            or any(data['sources'].get(key) != hashes[key] for data in (field_data, role_data, proposal_data)
                   for key in ('checkpoint', 'individual'))
            or field_data['sources']['roles'] != hashes['roles'] or role_data['sources']['proposal'] != hashes['proposal']):
        raise ValueError('Field, role, and name states do not match the requested model')
    proposal_report = json.loads(paths['proposal_report'].read_text(encoding='utf-8'))
    if proposal_report['sources'] != proposal_data['sources'] or role_data['sources']['proposal_report'] != hashes['proposal_report']:
        raise ValueError('Name threshold provenance does not match the role model')
    saved = torch.load(verify / 'intent.pt', map_location='cpu', weights_only=True) if verify else None
    if saved is not None and (saved['format'] != FORMAT or saved['sources'] != hashes):
        raise ValueError('Intent weights do not belong to this model')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    runtime.decoder.eval().requires_grad_(False)
    if frozen_digest(runtime.organism, runtime.state) != hashes['individual']:
        raise ValueError('Field weights do not belong to this individual')
    featurize = FrozenTurnFeatures(runtime.decoder, tokenizer)
    intent_features = featurize if intent_depth == 12 else FrozenTurnFeatures(runtime.decoder, tokenizer, depth=intent_depth)
    width = field_data['heads']['attention']['project.weight'].shape[-1]
    field_head = FieldReadout(width, len(FIELDS), attention=True).eval().requires_grad_(False)
    field_head.load_state_dict(field_data['heads']['attention'], strict=True)
    role = JointNameRoleHead(width).eval().requires_grad_(False)
    role.load_state_dict(role_data['heads']['matched'], strict=True)
    proposer = NameSpanHead(width).eval().requires_grad_(False)
    proposer.load_state_dict(proposal_data['heads']['mixed'], strict=True)
    torch.manual_seed(SEED)
    intent = RequestIntent(width)
    patterns = TRAIN_INTENT + (CONTRAST_INTENT if contrast else ()) + (SCOPE_INTENT if scope else ())
    training = corpus(TRAIN_NAMES, patterns, TRAIN_VALUES)
    calibration = corpus(CAL_NAMES, CAL_INTENT, TEST_VALUES)
    held = corpus(TEST_NAMES, HELD_INTENT, TEST_VALUES)
    if saved is None:
        fit_intent(intent, *features(training, intent_features, width), training)
    else:
        intent.load_state_dict(saved['state'], strict=True)
        intent.eval().requires_grad_(False)
    cal_scores = intent_margin(intent(*features(calibration, intent_features, width))).tolist()
    floor = max(score for row, score in zip(calibration, cal_scores, strict=True) if not row['label'])
    test_scores = intent_margin(intent(*features(held, intent_features, width))).tolist()
    standalone = {'fixed': held_intent(held, test_scores, 0.), 'calibrated': held_intent(held, test_scores, floor)}
    print('Intent threshold', floor, {name: data['summary'] for name, data in standalone.items()}, flush=True)
    held_slots = examples(HELD_NAMES, HELD_TEXT, HELD_VALUES, SEED + 1)
    slots = replay(held_slots, featurize, tokenizer, field_head, role, proposer,
                   proposal_report['thresholds']['mixed'], intent, floor, intent_features)
    if (any(sha(path) != hashes[name] for name, path in paths.items())
            or frozen_digest(runtime.organism, runtime.state) != hashes['individual']):
        raise ValueError('Input weights or the original individual changed')
    record = {'format': FORMAT, 'sources': hashes, 'threshold': floor, 'train': len(training),
              'calibration': held_intent(calibration, cal_scores, floor), 'standalone': standalone, 'slots': slots}
    output.mkdir(parents=True, exist_ok=False)
    if saved is None:
        atomic_save({'format': FORMAT, 'sources': hashes, 'state': intent.state_dict()}, output / 'intent.pt')
        store_json(output / 'result.json', record)
    else:
        reference = json.loads((verify / 'result.json').read_text(encoding='utf-8'))
        if json.loads(json.dumps(record)) != reference:
            raise ValueError('Reloaded intent gate did not reproduce the audit')
        store_json(output / 'result.json', {'identical': True, 'sources': hashes})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'field', 'roles', 'proposal', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--contrast', action='store_true', help='Add paired alternatives that retain the same facts')
    parser.add_argument('--scope', action='store_true', help='Add contrasts for confirmation, quotation, and cancellation')
    parser.add_argument('--intent-depth', type=int, default=12, help='Frozen decoder depth used by the intent head')
    main(**vars(parser.parse_args()))
