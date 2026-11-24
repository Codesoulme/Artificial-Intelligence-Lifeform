"""Test replacing supplied correction values with values read from Chinese turns."""

import argparse
import json
from pathlib import Path
import random

from scripts.audit_spora_decision_transitions import ACTIONS, FORMAT as ACTION_FORMAT, SCENES, replay, test_rows
from scripts.audit_spora_clause_intent import clause_features
from scripts.audit_spora_typed_fields import features
from scripts.audit_spora_entity_field_address import (
    FIELDS, HELD_TEXT, HELD_VALUES, MODEL, SEED, TRAIN_NAMES, TRAIN_TEXT, TRAIN_VALUES, AutoTokenizer,
    FrozenTurnFeatures, frozen_digest, load_full_streamed, load_integrated_runtime, sha, store_json, torch,
)
from spora.language_training import atomic_save
from spora.research_clause_intent import ClauseIntent
from spora.research_revision_value import RevisionValueHead, select_value_spans
from torch.nn import functional as F


FORMAT = 'spora-revision-value-v1'
DEPTH = 12
LABELS = ('会合地点', '材料数量', '对接联系人', '预约时间')
SCENE_FIELDS = ((0, 0), (2, 3), (1, 3), (2, 2), (0, 3), (1, 0))
HELD_NAMES = ('邹晴晚', '路可盈', '文秋南', '江梓方', '屈冰夏', '楚荷濛', '燕尔然', '简韵南')
CAL_NAMES = ('谢清筠', '华寻松', '雷如溪', '阮照青', '屠知渔', '陶近禾', '聂遥林', '洪白山')
TRAIN_PAIR = (
    ('{name}的{field}原来是{old}，现已改为{new}。', '想继续用{old}，不要改为{new}。{name}的{field}照旧。'),
    ('{name}的{field}曾确定成{old}，现在请换到{new}。', '应当核对{name}的{field}是{old}或{new}，暂时别记。'),
    ('给{name}登记{new}。请用这个值更正{field}的旧记录{old}。', '我只转述一下：有人说{name}的{field}从{old}换成{new}。'),
    ('{name}的表上写了{old}，实际应该填{new}，指的是{field}。', '对{name}的{field}，你知道表上是{old}还是{new}吗？'),
)
HELD_PAIR = (
    ('本来给{name}记的是{old}。刚核实完，以{new}作为{field}。', '我问{name}的{field}究竟是{old}还是{new}。'),
    ('新值{new}已经生效，把{name}原来的{field}值{old}替掉。', '想把{name}的{field}改作{new}，但还要等消息，继续留着{old}。'),
    ('不要再照{old}处理，{name}的{field}最后确认是{new}。', '我转述早晨的讨论，有人称{name}的{field}要从{old}换成{new}。'),
)
CAL_PAIR = (
    ('请核对{name}的{field}，旧单填{old}，新登记填{new}。', '暂且只比较{name}的{field}，{old}和{new}都别写进去。'),
    ('本次确定采用{new}，用它替换{name}的{field}原值{old}。', '关于{name}的{field}，听说可能用{new}代替{old}。'),
)


def make_rows(names, templates, values, pairs, seed, *, pending=False):
    rng = random.Random(seed)
    rows = []
    for name in names:
        for index, field in enumerate(LABELS):
            old, new = rng.sample(values[index], 2)
            _, update, query = templates[index]
            rows.extend(({'text': update.format(name=name, value=new), 'field': index, 'value': new, 'kind': 'single'},
                         {'text': query.format(name=name), 'field': index, 'value': None, 'kind': 'reference'}))
            for positive, negative in pairs:
                data = {'name': name, 'field': field, 'old': old, 'new': new}
                rows.extend(({'text': positive.format(**data), 'field': index, 'value': new, 'kind': 'replacement'},
                             {'text': negative.format(**data), 'field': index, 'value': None, 'kind': 'no_value'}))
            if pending:
                rows.extend(({'text': f'{name}的{field}可以考虑用{new}，容我确认以后再定。', 'field': index,
                              'value': new, 'kind': 'proposal'},
                             {'text': f'请按已确认的候选更新{name}的{field}，还是用刚说过的值。', 'field': index,
                              'value': None, 'kind': 'reference'}))
    return rows


def head_inputs(rows, featurize, width):
    hidden, mask = features(rows, featurize, width)
    return hidden, mask, torch.tensor([row['field'] for row in rows])


def fit(head, rows, batch, *, separate_loss):
    hidden = batch[0]
    targets = []
    for row in rows:
        value = row['value']
        if value is None:
            targets.append(0)
            continue
        if row['text'].count(value) != 1:
            raise ValueError('Training value must have one exact span')
        begin = row['text'].index(value)
        targets.append(1 + begin * hidden.shape[1] + begin + len(value) - 1)
    labels = torch.tensor(targets)
    generator = torch.Generator().manual_seed(SEED)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.003, weight_decay=.03)
    for _ in range(10):
        for indices in torch.randperm(len(rows), generator=generator).split(64):
            optimizer.zero_grad(set_to_none=True)
            logits = head(*(item[indices] for item in batch))
            target = labels[indices]
            if separate_loss:
                spans = logits[:, 1:]
                present = target > 0
                span_loss = F.cross_entropy(spans[present], target[present] - 1, reduction='sum')
                gate_loss = F.binary_cross_entropy_with_logits(spans.amax(-1) - logits[:, 0], present.float())
                loss = span_loss / present.sum().clamp_min(1) + gate_loss
            else:
                loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
    return head.eval().requires_grad_(False)


def read_values(head, rows, batch, *, floor=0.):
    spans = select_value_spans(head(*batch), batch[0].shape[1], floor=floor)
    return [None if span is None else row['text'][span[0]:span[1]] for row, span in zip(rows, spans, strict=True)]


def evaluate(head, rows, features, *, floor):
    guessed = read_values(head, rows, features, floor=floor)
    details = [{**row, 'predicted': value, 'correct': value == row['value']}
               for row, value in zip(rows, guessed, strict=True)]
    report = {kind: {'correct': sum(row['kind'] == kind and row['correct'] for row in details),
                     'total': sum(row['kind'] == kind for row in details)} for kind in dict.fromkeys(r['kind'] for r in rows)}
    return {'summary': report, 'rows': details}


def rank_diagnostic(head, rows, batch):
    """Inspect span preferences with the abstention decision factored out."""
    logits = head(*batch)
    width = batch[0].shape[1]
    best = logits[:, 1:].argmax(-1).tolist()
    details = []
    for row, index, scores in zip(rows, best, logits, strict=True):
        begin, end = divmod(index, width)
        selected = row['text'][begin:end + 1]
        details.append({**row, 'top_span': selected, 'top_exact': selected == row['value'],
                        'span_margin': float(scores[index + 1] - scores[0])})
    return details


def sequence_replay(head, value_features, action_state, action_features, hidden, *, floor):
    turns = test_rows(context=True, focus_current=True)
    fields = {scene[0]: pair for scene, pair in zip(SCENES, SCENE_FIELDS, strict=True)}
    values = [{**row, 'text': row['turn'], 'field': fields[row['scene']][row['address']]} for row in turns]
    predicted = read_values(head, values, head_inputs(values, value_features, hidden), floor=floor)
    action_head = ClauseIntent(hidden, recurrent=True, classes=len(ACTIONS)).eval().requires_grad_(False)
    action_head.load_state_dict(action_state, strict=True)
    actions = action_head(*clause_features(action_features, turns, hidden)).argmax(-1).tolist()
    expected_actions = [row['label'] for row in turns]
    report = {}
    for name, choices, value_inputs in (('provided', expected_actions, [row['value'] for row in turns]),
                                       ('value_only', expected_actions, predicted), ('value_and_action', actions, predicted)):
        new_turns = [{**row, 'candidate': value} for row, value in zip(turns, value_inputs, strict=True)]
        report[name] = replay(new_turns, choices)
        print(name, report[name]['summary'], flush=True)
    return report


def main(checkpoint, actions, output, verify, contextual, diagnostic, separate_loss):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    inputs = {'checkpoint': sha(checkpoint), 'actions': sha(actions)}
    saved_actions = torch.load(actions, map_location='cpu', weights_only=True)
    if (saved_actions['format'] != ACTION_FORMAT or saved_actions['sources']['checkpoint'] != inputs['checkpoint']
            or not saved_actions['sources'].get('focus_current')):
        raise ValueError('The action head must use the same model and the current-turn feature configuration')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    if saved_actions['sources']['individual'] != individual:
        raise ValueError('The action head is bound to a different individual')
    sources = inputs | {'individual': individual, 'depth': DEPTH}
    if contextual:
        sources['contextual'] = True
    if separate_loss:
        sources['separate_loss'] = True
    saved = torch.load(verify / 'value.pt', map_location='cpu', weights_only=True) if verify else None
    if saved is not None and (saved['format'] != FORMAT or saved['sources'] != sources):
        raise ValueError('Saved value head belongs to a different model')
    value_features = FrozenTurnFeatures(runtime.decoder, tokenizer, depth=DEPTH)
    action_features = FrozenTurnFeatures(runtime.decoder, tokenizer, depth=saved_actions['sources']['depth'])
    hidden = runtime.decoder.base.config.hidden_size
    torch.manual_seed(SEED)
    head = RevisionValueHead(hidden, len(FIELDS), contextual=contextual)
    training = make_rows(TRAIN_NAMES, TRAIN_TEXT, TRAIN_VALUES, TRAIN_PAIR, SEED, pending=True)
    if saved is None:
        fit(head, training, head_inputs(training, value_features, hidden), separate_loss=separate_loss)
    else:
        head.load_state_dict(saved['state'], strict=True)
        head.eval().requires_grad_(False)
    held = make_rows(HELD_NAMES, HELD_TEXT, HELD_VALUES, HELD_PAIR, SEED + 1)
    held_batch = head_inputs(held, value_features, hidden)
    floor = 0.
    if separate_loss:
        calibration = [row for row in make_rows(CAL_NAMES, TRAIN_TEXT, TRAIN_VALUES, CAL_PAIR, SEED + 2)
                       if row['kind'] in ('replacement', 'no_value')]
        ranked = rank_diagnostic(head, calibration, head_inputs(calibration, value_features, hidden))
        floor = max(row['span_margin'] for row in ranked if row['value'] is None)
        print('Calibration threshold', floor, flush=True)
    value_only = evaluate(head, held, held_batch, floor=floor)
    print('Held value extraction', value_only['summary'], flush=True)
    sequence = sequence_replay(head, value_features, saved_actions['state'], action_features, hidden, floor=floor)
    if (sha(checkpoint) != inputs['checkpoint'] or sha(actions) != inputs['actions']
            or frozen_digest(runtime.organism, runtime.state) != individual):
        raise ValueError('Source weights or individual state changed')
    output.mkdir(parents=True, exist_ok=False)
    result = {'format': FORMAT, 'sources': sources, 'training_count': len(training), 'held': value_only, 'sequence': sequence}
    if separate_loss:
        result['calibrated_floor'] = floor
        result['fixed_floor'] = evaluate(head, held, held_batch, floor=0.)['summary']
    if saved is None:
        atomic_save({'format': FORMAT, 'sources': sources, 'state': head.state_dict()}, output / 'value.pt')
    else:
        expected = json.loads((verify / 'result.json').read_text(encoding='utf-8'))
        if json.loads(json.dumps(result)) != expected:
            raise ValueError('Reloaded value predictions differ from their recorded run')
        result = {'identical': True, 'sources': sources}
    store_json(output / 'result.json', result)
    if diagnostic:
        details = rank_diagnostic(head, held, held_batch)
        counts = {kind: sum(row['kind'] == kind and row['top_exact'] for row in details)
                  for kind in ('single', 'replacement')}
        store_json(output / 'span-ranking.json', {'sources': sources, 'top_exact': counts, 'rows': details})
        print('Top span without abstention', counts, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'actions', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--contextual', action='store_true', help='Let both span endpoints read surrounding characters')
    parser.add_argument('--diagnostic', action='store_true', help='Report the best span before abstention for offline analysis')
    parser.add_argument('--separate-loss', action='store_true', help='Train span ranking independently from abstention')
    main(**vars(parser.parse_args()))
