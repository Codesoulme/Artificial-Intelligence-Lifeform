"""Check whether earlier dialogue changes the decision to write a single field."""

import argparse
import json
from pathlib import Path

from scripts.audit_spora_write_intent import (
    CONTRAST_INTENT, FORMAT, MODEL, SEED, TRAIN_INTENT, TRAIN_NAMES, TRAIN_VALUES, AutoTokenizer, FrozenTurnFeatures,
    RequestIntent, atomic_save, corpus, features, fit_intent, frozen_digest, intent_margin,
    load_full_streamed, load_integrated_runtime, sha, store_json, torch,
)


# Each pair holds the final user message fixed while changing the earlier dialogue.
CASES = (
    ('collect', '给邓禾改的是取货地点，新地方是西楼侧门。', '给邓禾改取货地点和取货时间，两项还要一起处理。',
     '那就把变动写进记录。'),
    ('receipt', '郭清的签收时间核实了，是周二上午。', '郭清的签收时间还没核实，周二上午只是猜的。',
     '按刚才的情况更新吧。'),
    ('place', '我们定了孟雪在走廊尽头碰头，其他不动。', '我们商量过孟雪或许在走廊尽头碰头，还在等回复。',
     '你处理一下刚说的安排。'),
    ('contact', '梁川这条只换联系人，已经定了由顾明接手。', '梁川这条要把联系人和会面地点都换掉。',
     '帮我按刚才说的更新。'),
    ('stock', '和任晓点过数了，给她留的材料更正为十二箱。', '还没点数，有人猜任晓的材料也许是十二箱。',
     '材料那项麻烦按核实的写。'),
    ('replace', '冯棋的安排改好了，只需要把接应人换为潘晴。', '刚收到消息，冯棋改接应人的事取消了。',
     '你把她那条记录处理一下。'),
    ('cancel', '收回最早的暂缓。现在确定贾宁下午六点来，时间要改。', '贾宁本来改到下午六点来，但最后这个更正撤销了。',
     '照最后确认的决定处理。'),
    ('report', '这是明确通知，陆晨的会合位置现在正式改到门厅。', '下面我只是读一段演练台词：“陆晨的会合位置正式改到门厅。”',
     '你看看这句话怎么处理。'),
    ('check', '我刚核对纸质记录，程实要领的实际是十件，旧数字填错了。', '我在核对纸质记录，想知道程实原来要领的是不是十件。',
     '请按我刚说的意思处理。'),
    ('two_fields', '魏书的接应人改为吴立，时间和地点一律照旧。', '魏书的接应人改为吴立，地点也一定要一并更新。',
     '上面那项已经确认，可以处理。'),
    ('provisional', '陈桐的领用份数确认好了，实际是三包。', '陈桐的领用份数没有敲定，有人临时报三包供参考。',
     '确认了再记到她那一栏。'),
    ('read_only', '机器里的赵沅那项日期写错了，实际是星期五，请改日期。',
     '我想知道机器里赵沅那项日期原来是什么，星期五只是我在猜。', '麻烦处理一下这个请求。'),
)
PAIRS = (
    ('刚确认{name}的{field}用{value}，单子上的旧内容已经作废。',
     '{name}的{field}有人提出用{value}，旧内容仍有效，要等批准。', '这条麻烦处理一下。'),
    ('清单上只改{name}的{field}为{value}，其他条目保留。',
     '清单上的{name}那一项，{field}改为{value}，另外{other}也得重填。', '照上面的要求登记。'),
    ('上一条不是提议了，{name}的{field}现在确定为{value}。',
     '还是提议，{name}的{field}暂时可能用{value}，等我通知。', '你按刚刚那条来办。'),
    ('核查完了，{name}的{field}应该用{value}，上次写错。',
     '请帮我核查一下{name}的{field}，想知道是不是{value}。', '表里的事项请按我的意思处理。'),
    ('{name}换成{value}了，更新的是{field}，已经确认。',
     '我只是复述这段例句：“{name}换成{value}了，更新的是{field}。”', '你理解以后再操作。'),
    ('恢复本次修改，给{name}把{field}换成{value}，已经生效。',
     '撤回本次修改，给{name}把{field}换成{value}的安排停止执行。', '按最后定的要求做。'),
)
VALIDATION = (
    ('说定了，{name}新登记的{field}是{value}。', '有人猜{name}登记的{field}是{value}，我还要问。',
     '这份更新请你处理。'),
    ('{name}的{field}必须更正为{value}，其余不改。', '{name}的{field}和另一个栏目都要变，其中报了{value}。',
     '刚才交代的事麻烦办一下。'),
    ('{name}的{field}确实有误，把{value}记为最新值。', '我念的是草稿，里面说{name}的{field}改成{value}。',
     '就处理刚才那一句。'),
)
RECENCY_PAIRS = (
    ('原来暂停了{name}的更正。刚有正式回复，{field}改成{value}，单子可以更新。',
     '原来确认{name}的{field}改成{value}。刚有正式回复，请撤销本次更正。', '接下来按这次通知办。'),
    ('先前没定是否改{name}的{field}，最新决定是改为{value}，立即生效。',
     '先前定的是{name}的{field}改为{value}，最新决定是停止这项变动。', '把最新决定处理掉。'),
    ('起初怀疑{name}的{field}要改，现在查实应该记{value}。',
     '起初确定{name}的{field}记{value}，现在说需要继续核查，现有表先别动。', '麻烦按最后那条来。'),
    ('之前示例里说{name}的{field}用{value}。现在正式确定，按这个值更新。',
     '之前正式通知{name}的{field}用{value}。后来才知道这是举例，变更取消。', '照本轮最新结果执行。'),
    ('曾经说{name}连{other}一起动。范围已经缩小，只把{field}改成{value}。',
     '曾经说只把{name}的{field}改成{value}。范围刚扩大，还要一并改{other}。', '以上是新要求，你处理吧。'),
    ('想知道{name}的{field}是否需要更正，刚已核实，用{value}覆盖旧值。',
     '核实了{name}的{field}可以填{value}，后面我改主意了，只想先看现值。', '照我现在的意思处理。'),
)
RECENCY_VALIDATION = (
    ('曾经取消给{name}改{field}。恢复处理，新的确定值为{value}。',
     '曾经给{name}定了{field}为{value}。刚刚确认该调整作废。', '就按目前说定的情况来。'),
    ('旧消息称{name}的{field}和{other}都要动，最终只调整{field}为{value}。',
     '旧消息称{name}的{field}改为{value}，最终明确{other}也得一起动。', '把上面交代的落实一下。'),
    ('最开始{name}的{field}说是{value}只供参考，刚对完记录，确认应改为这个数。',
     '最开始说要改{name}的{field}为{value}，刚问过还在等答复，先不落定。', '照现在的结论处理。'),
)
RECENCY_CASES = (
    ('reinstated', '薛苇的取货时间原先暂停变更。仓库刚回话，正式改在周六上午，单子同步。',
     '薛苇的取货时间原先改到周六上午。仓库刚回话，那个更改没有获批。', '以仓库最后的意见为准。'),
    ('last_cancelled', '发给宋妍的份数一度取消调整，清点后确认是五盒，请登记这个新数。',
     '发给宋妍的份数一度确认成五盒，现在发现清点有问题，先恢复旧表。', '你跟进一下最后那条。'),
    ('last_scope', '程辉的登记原本要改收到时间及地址，刚确认只把时间推到下午四点。',
     '程辉的登记原本只把时间推到下午四点，通知追加了地址也必须一起更新。', '新版通知你按它处理。'),
    ('only_preview', '殷寒原本只想查地点。现在地点核对完，旧地点填错，换到门卫室。',
     '殷寒原本要把地点改到门卫室。她刚说要先看看旧地点，改动稍后再谈。', '你照刚才的结论操作。'),
    ('handover', '最早许榆的接应联系人没定。刚宣布由小周接手，更新联络表。',
     '最早说由小周接应许榆。刚知道那是预案，负责人还没定。', '上面那个情况处理一下。'),
    ('quoted', '演练时说胡琪要去二楼。现在发布正式通知，集合点改二楼，更新表上的地点。',
     '通知草稿写胡琪要去二楼。正式通知还没发，二楼是演练例子。', '按最后确定的信息办理。'),
    ('latest_count', '陶理的参数表先别改，这条消息已撤回。现在确定材料换为九包，数量那栏改一下。',
     '陶理的材料曾确认换为九包。刚收到消息，先别再改参数表了。', '最新消息请处理。'),
    ('only_location', '崔悦的出发安排曾要求换时间和路线，照刚确认的，只改集合地点为南门。',
     '崔悦的安排曾要求只把地点改南门。照刚确认的，出发时间也要一起改。', '请按最终范围登记。'),
)


def dialogue_text(earlier, final):
    return '用户：' + earlier + '\n助手：我听到了。\n用户：' + final


def contexts(names, patterns):
    fields = ('见面点', '要发的份数', '接洽人', '到达时间')
    return [{'text': dialogue_text(earlier.format(name=name, field=field, value=TRAIN_VALUES[i],
                                                  other=fields[(i + 1) % len(fields)]), final), 'label': label}
            for name in names for i, field in enumerate(fields) for positive, negative, final in patterns
            for label, earlier in ((1, positive), (0, negative))]


def audit(head, floor, featurize, width, cases=CASES):
    cases = [{'pair': name, 'label': label, 'earlier': earlier, 'final': final}
             for name, positive, negative, final in cases for label, earlier in ((True, positive), (False, negative))]
    results = {}
    for mode in ('final', 'dialogue'):
        requests = [{'text': row['final'] if mode == 'final' else dialogue_text(row['earlier'], row['final'])}
                    for row in cases]
        margins = intent_margin(head(*features(requests, featurize, width))).tolist()
        rows = [{**case, 'margin': margin, 'allowed': margin > floor}
                for case, margin in zip(cases, margins, strict=True)]
        counts = {'write_correct': sum(row['allowed'] and row['label'] for row in rows),
                  'reject_correct': sum(not row['allowed'] and not row['label'] for row in rows),
                  'pairs_correct': sum(a['allowed'] and not b['allowed'] for a, b in zip(rows[::2], rows[1::2], strict=True)),
                  'positive_ranked_higher': sum(a['margin'] > b['margin'] for a, b in zip(rows[::2], rows[1::2], strict=True))}
        results[mode] = {'counts': counts, 'rows': rows}
        print(mode, counts, flush=True)
    return results


def continue_intent(state, featurize, width, output, hashes, recency, resume):
    contextual = contexts(TRAIN_NAMES[:12], PAIRS + (RECENCY_PAIRS if recency else ()))
    request_rows = corpus(TRAIN_NAMES[12:], TRAIN_INTENT + CONTRAST_INTENT, TRAIN_VALUES)
    generator = torch.Generator().manual_seed(SEED)
    order = []
    while len(order) < 2 * len(contextual):
        order.extend(torch.randperm(len(request_rows), generator=generator).tolist())
    shared = [request_rows[i] for i in order[:len(contextual)]]
    single_turn = [request_rows[i] for i in order[len(contextual):2 * len(contextual)]]
    calibration_cases = VALIDATION + (RECENCY_VALIDATION if recency else ())
    validation = contexts(('舒朝扬', '杜英河', '罗晓月', '顾树明'), calibration_cases)
    val_features = features(validation, featurize, width)
    previously_saved = torch.load(resume.with_suffix('.pt'), map_location='cpu', weights_only=True) if resume else None
    if previously_saved is not None and previously_saved['sources'] != hashes:
        raise ValueError('Continued weights belong to a different intent experiment')
    arms, saved = {}, {}
    for name, supplement in (('continued_single', single_turn), ('continued_dialogue', contextual)):
        rows = shared + supplement
        head = RequestIntent(width)
        if previously_saved is None:
            head.load_state_dict(state, strict=True)
            fit_intent(head, *features(rows, featurize, width), rows, epochs=6, learning_rate=.001)
        else:
            head.load_state_dict(previously_saved['heads'][name], strict=True)
            head.eval().requires_grad_(False)
        margins = intent_margin(head(*val_features)).tolist()
        floor = max(score for row, score in zip(validation, margins, strict=True) if not row['label'])
        arms[name] = {'threshold': floor, 'training': len(rows), 'results': audit(head, floor, featurize, width),
                      'calibration_positive': sum(row['label'] and score > floor
                                                  for row, score in zip(validation, margins, strict=True))}
        if recency:
            arms[name]['unseen_recency'] = audit(head, floor, featurize, width, RECENCY_CASES)
        saved[name] = head.state_dict()
    if previously_saved is None:
        atomic_save({'sources': hashes, 'heads': saved}, output.with_suffix('.pt'))
    return arms


def main(checkpoint, experiment, output, train, recency, verify):
    if output.exists():
        raise FileExistsError(output)
    if train and output.with_suffix('.pt').exists():
        raise FileExistsError(output.with_suffix('.pt'))
    torch.set_num_threads(4)
    paths = {'checkpoint': checkpoint, 'weights': experiment / 'intent.pt', 'report': experiment / 'result.json'}
    hashes = {name: sha(path) for name, path in paths.items()}
    saved = torch.load(paths['weights'], map_location='cpu', weights_only=True)
    report = json.loads(paths['report'].read_text(encoding='utf-8'))
    if (report['format'] != FORMAT or saved['format'] != FORMAT or report['sources'] != saved['sources']
            or saved['sources']['checkpoint'] != hashes['checkpoint']):
        raise ValueError('Intent weights and report must come from this model')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    runtime.decoder.eval().requires_grad_(False)
    before = frozen_digest(runtime.organism, runtime.state)
    if before != saved['sources']['individual']:
        raise ValueError('Intent weights belong to a different individual')
    depth = saved['sources'].get('intent_depth', 12)
    featurize = FrozenTurnFeatures(runtime.decoder, tokenizer, depth=depth)
    width = saved['state']['decision.0.weight'].shape[-1] // 2
    head = RequestIntent(width).eval().requires_grad_(False)
    head.load_state_dict(saved['state'], strict=True)
    results = audit(head, report['threshold'], featurize, width)
    output.parent.mkdir(parents=True, exist_ok=True)
    fresh = audit(head, report['threshold'], featurize, width, RECENCY_CASES) if recency else {}
    continued = continue_intent(saved['state'], featurize, width, output, hashes, recency, verify) if train or verify else {}
    if any(sha(path) != hashes[name] for name, path in paths.items()) or before != frozen_digest(runtime.organism, runtime.state):
        raise ValueError('Source or individual state changed')
    result = {'sources': hashes, 'depth': depth, 'threshold': report['threshold'], 'results': results,
              'unseen_recency': fresh, 'continued': continued}
    if verify is not None:
        expected = json.loads(verify.read_text(encoding='utf-8'))
        if result != expected:
            raise ValueError('Reloaded dialogue decisions differ')
        result = {'identical': True, 'report_sha256': sha(verify), 'weights_sha256': sha(verify.with_suffix('.pt'))}
    store_json(output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'experiment', 'output'):
        parser.add_argument(name, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--train', action='store_true', help='Compare contextual continuation with equal-size single turns')
    mode.add_argument('--verify', type=Path, help='Reload and verify a saved comparison')
    parser.add_argument('--recency', action='store_true', help='Check changes to decisions later in the dialogue')
    main(**vars(parser.parse_args()))
