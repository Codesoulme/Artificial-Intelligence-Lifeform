"""Learn proposal, confirmation, and withdrawal and check their state after each turn."""

import argparse
import json
from pathlib import Path

from scripts.audit_spora_clause_intent import DEPTH, clause_features
from scripts.audit_spora_context_write_intent import (
    MODEL, SEED, TRAIN_NAMES, AutoTokenizer, FrozenTurnFeatures, atomic_save, fit_intent, frozen_digest,
    load_full_streamed, load_integrated_runtime, sha, store_json, torch,
)
from spora.research_clause_intent import ClauseIntent


FORMAT = 'spora-decision-transitions-v2'
ACTIONS = ('other', 'propose', 'confirm', 'withdraw')
FIELDS = ('集合点', '对接人', '到达时间', '领用数量')
VALUES = ('北侧回廊', '尚云', '下午三点', '六盒')
TRAIN = {
    'other': ('帮我读一下{name}那份表上的{field}。', '我只复述原话，说{name}的{field}应填{value}。',
              '知道了，感谢你帮{name}核对{field}。', '我只想看看{name}原先的{field}，别调。',
              '{value}这几个字怎么念？跟{name}的{field}无关。'),
    'propose': ('先把{name}的{field}拟作{value}，我问明白再确认。', '可以考虑给{name}把{field}改成{value}，先别登记。',
                '{name}的{field}暂时建议换成{value}，这只是备选。', '有人报了{value}，可能是{name}的新{field}，有待核对。',
                '给{name}预想一个{field}用{value}的方案，回头定。'),
    'confirm': ('{name}的{field}已经核实，就把刚才提出的那项记上。', '确认执行{name}的那条{field}变更。',
                '不用再等，按刚刚那版更新{name}的{field}。', '把{name}的{field}正式登记为{value}。',
                '刚收到最终消息，{name}的{field}确定为{value}，这次生效。'),
    'withdraw': ('取消{name}的{field}修改，旧记录照常用。', '撤销早些时候给{name}提出的{field}变动。',
                 '那项{name}的{field}更正作废，先维持原来的。', '刚确认不再改{name}的{field}，之前的提议放弃。',
                 '{name}的{field}不用按刚才的提议了，原样。'),
}

# Hand-authored scenes attach the reference action, record address, and value to each utterance.
# A missing supplied value means a confirmation can consume only an existing proposal.
SCENES = (
    ('handover', ('辛漫的接班地点', '乔珂的接班地点'), ('东厅', '停车场'), (
        ('辛漫那班，也许能挪到西门亭，等我跟她讲完再说。', 'propose', 0, '西门亭'),
        ('乔珂的先把南门外当作备选，暂且按旧安排。', 'propose', 1, '南门外'),
        ('两张单上现在分别写的哪里？', 'other', 0, None),
        ('辛漫那项我拍板了，就用刚才说的新地方。', 'confirm', 0, None),
        ('乔珂不用挪了，把刚才那个去南门外的主意搁掉。', 'withdraw', 1, None),
        ('乔珂这次还是按表上写的集合，是哪里？', 'other', 1, None),
    )),
    ('courier', ('单雅的收件人', '严渠的取件时间'), ('黄枝', '周五'), (
        ('单雅这份有个备选，秦峰能帮忙收，我还得征得同意。', 'propose', 0, '秦峰'),
        ('我刚打完电话，严渠的取件日改到星期二了，给他改表。', 'confirm', 1, '星期二'),
        ('我只是报个信，大家听说单雅以后可能都请秦峰帮忙。', 'other', 0, None),
        ('单雅找秦峰那条不用再考虑，旧收件人照用。', 'withdraw', 0, None),
        ('单雅这次已经定了，请把收件人写成向晓。', 'confirm', 0, '向晓'),
    )),
    ('supplies', ('常清的材料数', '常清的到场时间'), ('一捆', '正午'), (
        ('常清那一项件数，八捆先做预案，库房还没有给准信。', 'propose', 0, '八捆'),
        ('常清来馆的时间暂拟为十一点，目前用原时间。', 'propose', 1, '十一点'),
        ('常清的材料方案先跳过，库房说用回旧数。', 'withdraw', 0, None),
        ('库房改口说物料确实有变，常清的数量填九捆。', 'confirm', 0, '九捆'),
        ('我和常清敲定了她之前提出的来馆时间，按那个录。', 'confirm', 1, None),
    )),
    ('replacement', ('毕州的对接人', '郁莲的对接人'), ('沈末', '童秋'), (
        ('毕州的事不妨先考虑覃鸣接手，还没签字。', 'propose', 0, '覃鸣'),
        ('另一个备选是叶可，我先把毕州的预案更新成这个。', 'propose', 0, '叶可'),
        ('郁莲那边我只来查一下目前谁在对接。', 'other', 1, None),
        ('毕州的人选就用更新后的预案，已经通过。', 'confirm', 0, None),
        ('郁莲的新负责人确定为孙佳，登记上。', 'confirm', 1, '孙佳'),
    )),
    ('reversal', ('管竹的终点', '杜乐的取货日'), ('东桥', '周六'), (
        ('杜乐那次领货有人提出改成周二，先留下供商量。', 'propose', 1, '周二'),
        ('杜乐这次已经确认，之前的领货日提案照办。', 'confirm', 1, None),
        ('管竹说想改到广场。我只是把她的意见转给你听。', 'other', 0, None),
        ('杜乐接下来的领取日也可能换成周四，先别用。', 'propose', 1, '周四'),
        ('杜乐周四那条是临时说法，划掉，现行记录保持。', 'withdraw', 1, None),
    )),
    ('interleaved', ('沈葵的材料数', '沈葵的集合点'), ('一组', '北门'), (
        ('沈葵需要的量也许会改为四组，等主管点头。', 'propose', 0, '四组'),
        ('集合点这项确定改到钟楼，写进沈葵那栏。', 'confirm', 1, '钟楼'),
        ('另有人建议沈葵上午早点来，我只是转述。', 'other', 1, None),
        ('关于沈葵的材料，主管同意刚给出的新数量了。', 'confirm', 0, None),
        ('原先北门那个地方先不用恢复，我只是想核对旧值。', 'other', 1, None),
    )),
)


def training_rows():
    return [{'text': pattern.format(name=name, field=field, value=value), 'label': ACTIONS.index(action)}
            for name in TRAIN_NAMES for field, value in zip(FIELDS, VALUES, strict=True)
            for action, patterns in TRAIN.items() for pattern in patterns]


def contextual_training_rows(*, focus_current):
    rows = []
    actions = (
        ('other', '先读给我听一下。', '帮我看当前那栏。'),
        ('propose', '这个先当备选，别急着落表。', '先拟作一个方案，等人答复。'),
        ('confirm', '刚核实过了，就按这个更正。', '已经有人拍板，把刚刚那个写进去。'),
        ('withdraw', '之前提出的调整现在划掉，原样保存。', '我改主意了，刚才那项调整停止。'),
    )
    for i, name in enumerate(TRAIN_NAMES):
        peer = TRAIN_NAMES[(i + 1) % len(TRAIN_NAMES)]
        for field, value in zip(FIELDS, VALUES, strict=True):
            history = f'{name}的{field}考虑改成{value}，还没定。\n{peer}的那单用原安排。'
            for action, *responses in actions:
                explicit = TRAIN[action][i % len(TRAIN[action])].format(name=name, field=field, value=value)
                for utterance in (*responses, explicit):
                    row = {'text': history + '\n' + utterance, 'label': ACTIONS.index(action)}
                    if focus_current:
                        row['focus_start'] = len(history) + 1
                    rows.append(row)
    return rows


def test_rows(*, context, focus_current=False):
    rows = []
    for scene, _, _, turns in SCENES:
        previous = []
        for index, (text, action, address, value) in enumerate(turns):
            context_text = '\n'.join([*previous[-2:], text]) if context else text
            row = {'text': context_text, 'turn': text, 'label': ACTIONS.index(action), 'scene': scene,
                   'index': index, 'address': address, 'value': value}
            if focus_current:
                row['focus_start'] = len(context_text) - len(text)
            rows.append(row)
            previous.append(text)
    return rows


def transition(action, address, value, pending, active):
    if action == 'propose' and value is not None:
        pending[address] = value
    elif action == 'withdraw':
        pending.pop(address, None)
    elif action == 'confirm':
        previous_proposal = pending.pop(address, None)
        confirmed = value if value is not None else previous_proposal
        if confirmed is not None:
            active[address] = confirmed


def replay(rows, actions):
    initials = {name: original for name, _, original, _ in SCENES}
    turns = []
    for row, prediction in zip(rows, actions, strict=True):
        if row['index'] == 0:
            pending, expected_pending = {}, {}
            active = list(initials[row['scene']])
            expected_active = active.copy()
        expected = ACTIONS[row['label']]
        actual = ACTIONS[prediction]
        address, value = row['address'], row['value']
        before = active.copy()
        transition(expected, address, value, expected_pending, expected_active)
        transition(actual, address, row.get('candidate', value), pending, active)
        turns.append({**row, 'actual_action': actual, 'expected_action': expected,
                      'active': active.copy(), 'expected_active': expected_active.copy(),
                      'pending': sorted(pending.items()), 'expected_pending': sorted(expected_pending.items()),
                      'premature_write': expected == 'propose' and before[address] != active[address],
                      'unexpected_write': expected != 'confirm' and before != active,
                      'other_record_preserved': before[1 - address] == active[1 - address]})
    summary = {'turns': len(turns), 'action_correct': sum(r['actual_action'] == r['expected_action'] for r in turns),
               'active_correct': sum(r['active'] == r['expected_active'] for r in turns),
               'pending_correct': sum(r['pending'] == r['expected_pending'] for r in turns),
               'premature_writes': sum(r['premature_write'] for r in turns),
               'unexpected_writes': sum(r['unexpected_write'] for r in turns),
               'other_record_preserved': sum(r['other_record_preserved'] for r in turns),
               'scenes_exact': sum(all(r['active'] == r['expected_active'] and r['pending'] == r['expected_pending']
                                       for r in turns if r['scene'] == name) for name in initials)}
    summary['by_action'] = {action: {'correct': sum(r['actual_action'] == action == r['expected_action'] for r in turns),
                                     'total': sum(r['expected_action'] == action for r in turns)} for action in ACTIONS}
    return {'summary': summary, 'rows': turns}


def main(checkpoint, output, verify, continuation, focus_current):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    checkpoint_sha = sha(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    sources = {'checkpoint': checkpoint_sha, 'individual': individual, 'depth': DEPTH, 'continuation': continuation}
    if focus_current:
        sources['focus_current'] = True
    saved = torch.load(verify / 'head.pt', map_location='cpu', weights_only=True) if verify else None
    if saved is not None and (saved['format'] != FORMAT or saved['sources'] != sources):
        raise ValueError('Action weights belong to a different experiment')
    featurize = FrozenTurnFeatures(runtime.decoder, tokenizer, depth=DEPTH)
    width = runtime.decoder.base.config.hidden_size
    torch.manual_seed(SEED)
    head = ClauseIntent(width, recurrent=True, classes=len(ACTIONS))
    train = training_rows()
    if continuation != 'none':
        extra = contextual_training_rows(focus_current=focus_current)
        if continuation == 'repeat':
            order = torch.randperm(len(train), generator=torch.Generator().manual_seed(SEED)).tolist()
            extra = [train[index] for index in order[:len(extra)]]
        train.extend(extra)
    if saved is None:
        fit_intent(head, *clause_features(featurize, train, width), train, epochs=10, learning_rate=.002)
    else:
        head.load_state_dict(saved['state'], strict=True)
        head.eval().requires_grad_(False)
    report = {}
    for mode in ('latest_turn', 'with_context'):
        rows = test_rows(context=mode == 'with_context', focus_current=focus_current)
        selected = head(*clause_features(featurize, rows, width)).argmax(-1).tolist()
        report[mode] = replay(rows, selected)
        print(mode, report[mode]['summary'], flush=True)
    if sha(checkpoint) != checkpoint_sha or frozen_digest(runtime.organism, runtime.state) != individual:
        raise ValueError('Source model or individual state changed during the replay')
    output.mkdir(parents=True, exist_ok=False)
    result = {'format': FORMAT, 'sources': sources, 'train': len(train), 'results': report}
    if saved is None:
        atomic_save({'format': FORMAT, 'sources': sources, 'state': head.state_dict()}, output / 'head.pt')
    else:
        expected = json.loads((verify / 'result.json').read_text(encoding='utf-8'))
        if json.loads(json.dumps(result)) != expected:
            raise ValueError('Reloaded action decisions differ from the original audit')
        result = {'identical': True, 'sources': sources}
    store_json(output / 'result.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--continuation', choices=('none', 'repeat', 'context'), default='none')
    parser.add_argument('--focus-current', action='store_true', help='Read only the current turn endpoints in the full input')
    main(**vars(parser.parse_args()))
