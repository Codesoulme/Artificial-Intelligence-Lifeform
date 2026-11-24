"""Measure last-decision handling and exact revision application in SPORA."""

import argparse
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

from torch.nn import functional as F

from scripts.audit_spora_clause_intent import clause_features
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.train_spora_revision_intent import (
    DATA, DESIGN, EPISODES, EXTRA_NAMES, FIELDS, HELD, MODEL, OBSERVATIONS, ROLE_NAMES, TEST, AutoTokenizer,
    JointTurnValueWriter, atomic_save, evaluate, examples, extend_positions, frozen_digest, held_episodes, install,
    load_full_streamed, load_integrated_runtime, load_recurrent_replacements, packed, read, records, sha, store_json,
    torch, training_groups, value_span,
)
from spora.research_clause_intent import ClauseIntent
from spora.research_observation_episode import ObservationEpisode
from spora.research_revision_draft import RevisionDecision as Decision, RevisionDraft
from scripts.train_spora_observed_episodes import encode_changes, state_for, state_values


SEED = 26184
PAIRS = ((Decision.PROPOSE, Decision.CONFIRM), (Decision.CONFIRM, Decision.WITHDRAW),
         (Decision.CONFIRM, Decision.OTHER), (Decision.PROPOSE, Decision.OTHER),
         (Decision.CONFIRM, Decision.PROPOSE), (Decision.PROPOSE, Decision.WITHDRAW))
FOLLOWUP = {
    Decision.CONFIRM: ('已核验，那个备选现在生效。', '之前没定，现在确定采用刚说的方案。',
                       '这项正式定下，按上面的新值执行。', '最后收到同意，先前的提议成了最终意见。',
                       '不再只是备选，已经批准办理。'),
    Decision.WITHDRAW: ('收回改动，照旧。', '这次变更的指示取消，原安排继续。',
                        '刚才的变动不执行。', '同意的那条撤回，先前记录还是有效。',
                        '后来查到不该变，撤掉修改。'),
    Decision.PROPOSE: ('重开讨论，刚才确定的新值现在只有参考意义。', '先搁置，最终版本再等通知。',
                       '把上一条降为预案，还没正式通过。', '等重新核对，不能先算已决定。',
                       '改动还需要复查，暂时只把它留作候选。'),
    Decision.OTHER: ('我确认收到这条提示，先聊点别的。', '另外，外面可能要下雨。',
                     '谢谢，你能看到我刚发的字吗？', '另有件事，你知道今天星期几吗？',
                     '这段字有点小，我换台设备看。'),
}
CAL_FOLLOWUP = {
    Decision.CONFIRM: ('刚回话，说那个备选已正式获准。', '最新答复确定用提到的替代安排。'),
    Decision.WITHDRAW: ('刚回话说替换作废，原安排作数。', '最终没有采纳变更，撤销刚才的说法。'),
    Decision.PROPOSE: ('最新消息要求再议，暂缓落实那个新值。', '他们正重新评估，这一项还没形成决议。'),
    Decision.OTHER: ('另外提醒你，午饭在食堂吃。', '我刚收到消息，和这个安排无关。'),
}
HELD_FOLLOWUP = {
    Decision.CONFIRM: ('等到了负责人的批复，就照那个替代方案落定。', '前面报的备选转成正式安排，按它执行。'),
    Decision.WITHDRAW: ('说要用新方案的事我收回，这一栏恢复此前记录。', '刚改口，不用那个替代值，变更不再有效。'),
    Decision.PROPOSE: ('现在算不上定案，先把变动留在讨论稿，正式表别更新。', '把已经说好的调整先缓一缓，等重新拍板。'),
    Decision.OTHER: ('另一个事情我同意了，顺便问问今天天气。', '对了，我收到你的回复了，咱们另开一个话题。'),
}
APPLY = ('按最后的意见处理这件事。', '请根据最后决定处理。', '事情就按目前有效的意见办。')


def inputs(groups, followup, split):
    out = []
    for i, group in enumerate(groups):
        for before, after in PAIRS:
            initial = group['texts'][0 if before == Decision.CONFIRM else 1]
            last = followup[after][i % len(followup[after])]
            complete = '\n'.join((initial, last, APPLY[split]))
            accepted = after == Decision.CONFIRM or (before == Decision.CONFIRM and after == Decision.OTHER)
            out.append({'group': i, 'first': initial, 'last': last, 'dialogue': complete, 'field': group['field'],
                        'candidate': group['candidate'], 'first_step': int(before), 'last_step': int(after),
                        'expected': accepted})
    return out


def training_rows(groups, episodes):
    events = {}
    for group in groups:
        field = FIELDS.index(group['field'])
        for index, action in enumerate((Decision.CONFIRM, Decision.PROPOSE, Decision.WITHDRAW)):
            events[group['texts'][index]] = (int(action), field)
    for episode in episodes:
        events[episode['last']] = (episode['last_step'], -100)
    return [(text, action, field) for text, (action, field) in events.items()]


def encode(featurize, sets, width):
    texts = sorted({row[name] for group in sets for row in group for name in ('first', 'last', 'dialogue')})
    hidden, masks = clause_features(featurize, [{'text': text} for text in texts], width)
    return {text: (h, m) for text, h, m in zip(texts, hidden, masks, strict=True)}


def fit(rows, features, width, output_size, *, with_field):
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    model = ClauseIntent(width, recurrent=True, classes=output_size + (len(FIELDS) if with_field else 0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.02)
    for epoch in range(12):
        for batch in torch.randperm(len(rows), generator=generator).split(48):
            text, target, field = zip(*(rows[i] for i in batch.tolist()), strict=True)
            raw = model(*packed(features, text))
            loss = F.cross_entropy(raw[:, :output_size], torch.tensor(target))
            expected = torch.tensor(field)
            selected = expected.ne(-100)
            if with_field and selected.any():
                loss = loss + F.cross_entropy(raw[selected, output_size:], expected[selected])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch in (3, 7, 11):
            print('event' if with_field else 'dialogue', 'epoch', epoch + 1, round(float(loss.detach()), 5),
                  flush=True)
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def forecast(events, direct, features, cases, writer, chars, spans):
    first = events(*packed(features, [row['first'] for row in cases]))
    last = events(*packed(features, [row['last'] for row in cases]))
    decision_score = direct(*packed(features, [row['dialogue'] for row in cases]))
    results = []
    for row, a, b, d in zip(cases, first, last, decision_score, strict=True):
        initial, tail = (Decision(int(vector[:4].argmax())) for vector in (a, b))
        field = FIELDS[int(a[4:].argmax())]
        key = row['first'], field
        if key not in spans:
            spans[key] = value_span(writer, chars, row['first'], field)
        value = spans[key]
        draft = RevisionDraft(field).observe(initial, field=field, value=value).observe(tail)
        results.append({**row, 'predicted_field': field, 'predicted_value': value, 'event_first': int(initial),
                        'event_last': int(tail), 'structured_write': bool(draft.patch()),
                        'dialogue_margin': float(d[1] - d[0])})
    return results


def compare(rows):
    return {mode: {'commit': sum(row['expected'] and row[mode] for row in rows),
                   'false_commit': sum(not row['expected'] and row[mode] for row in rows),
                   'same_group_all_correct': sum(all(row['expected'] == row[mode] for row in rows[i:i + len(PAIRS)])
                                                 for i in range(0, len(rows), len(PAIRS))),
                   'complete_commit': sum(row['expected'] and row[mode] and
                                          (row['field'], row['candidate']) ==
                                          (row['predicted_field'], row['predicted_value']) for row in rows)}
            for mode in ('dialogue_write', 'structured_write')}


def finalize(rows, floor):
    out = [{**row, 'dialogue_write': row['dialogue_margin'] > floor} for row in rows]
    return {'rows': out, 'summary': compare(out)}


def state_audit(rows, episodes, tokenizer):
    written = []
    for index, case in enumerate(episodes):
        if not rows[index * len(PAIRS)]['group'] == index:
            raise ValueError('Replay episodes and held rows became misaligned')
        start = ObservationEpisode(state_for(tokenizer, case['initial']), case['topic'], torch.zeros(1))
        for row in rows[index * len(PAIRS):(index + 1) * len(PAIRS)]:
            field = row['predicted_field']
            value = row['predicted_value']
            slots = encode_changes(tokenizer, {field: value})
            expected = case['expected'] if row['expected'] else case['initial']
            outcomes = {}
            for mode in ('dialogue_write', 'structured_write'):
                after = start.apply(slots) if row[mode] else start
                actual = state_values(tokenizer, after.state)
                outcomes[mode] = {'exact': actual == expected, 'changed': start.state != after.state,
                                  'revision': after.revision, 'state': actual}
            written.append({'key': case['key'], 'expected_write': row['expected'], 'expected': expected,
                            'arms': outcomes})
    return written


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    candidate = args.output / 'decisions.pt'
    result_path = args.output / 'result.json'
    restore_heads = args.verify or args.resume_heads
    if restore_heads:
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        if args.resume_heads and result_path.exists():
            raise FileExistsError(result_path)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    study = read(args.study / 'result.json')
    weights = args.study / 'revision-writers.pt'
    paths = (args.checkpoint, DATA, DESIGN, EPISODES, OBSERVATIONS, args.study / 'result.json', weights,
             *(layer / name for layer in args.layers for name in ('attention.pt', 'comparison.json')))
    protected = {str(path): sha(path) for path in paths}
    original = {Path(p).resolve(): digest for p, digest in study['sources'].items()}
    for path in paths:
        if path.parent != args.study and protected[str(path)] != original.get(path.resolve()):
            raise ValueError(f'Frozen input changed: {path.name}')
    if protected[str(weights)] != study['weights_sha256']:
        raise ValueError('Frozen span writer changed')
    specimens = read(EPISODES)['cases']
    fixture = read(OBSERVATIONS)['topics']
    domains = {row['case']: row['fields'] for row in specimens}
    names = {name: (*ROLE_NAMES[name], EXTRA_NAMES[name]) for name in EXTRA_NAMES}
    source = read(DATA)
    train = training_groups(source, specimens, fixture, names)
    cal = training_groups(source, specimens, fixture, names, calibration=True)
    held = held_episodes({'cases': specimens})
    groups = examples(held, domains, names, TEST)
    for group, case in zip(groups, held, strict=True):
        group['texts'][0] = case['update']
    train_set, cal_set, held_set = (inputs(*params) for params in
                                    ((train, FOLLOWUP, 0), (cal, CAL_FOLLOWUP, 1), (groups, HELD_FOLLOWUP, 2)))
    splits = [{row[name] for row in arm for name in ('first', 'last', 'dialogue')}
              for arm in (train_set, cal_set, held_set)]
    if any(a & b for i, a in enumerate(splits) for b in splits[i + 1:]):
        raise ValueError('Dialogue training, calibration and evaluation overlap')
    event_rows = training_rows(train, train_set)
    direct_rows = [(row['dialogue'], int(row['expected']), -100) for row in train_set]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    # Keep the output projection on the model device from the first load.  Staging
    # it on CPU and moving it back later creates a second large tensor at the exact
    # point where the recurrent replacements and evaluation heads are loaded.
    base = load_full_streamed(stage_output_projection=False)
    native_rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    if individual != study['individual_sha256']:
        raise ValueError('This experiment uses another individual')
    width = base.config.hidden_size
    extend_positions(decoder, native_rotary, 2048)
    features_at_turn = FrozenTurnFeatures(decoder, tokenizer, depth=20)
    features = encode(features_at_turn, (train_set, cal_set, held_set,
                                        [{'first': text, 'last': text, 'dialogue': text} for text, _, _ in event_rows]),
                      width)
    if args.verify:
        expected = read(result_path)
        if expected['sources'] != protected or expected['decisions_sha256'] != sha(candidate):
            raise ValueError('The replay inputs changed')
    if restore_heads:
        saved = torch.load(candidate, map_location='cpu', weights_only=True)
        event = ClauseIntent(width, recurrent=True, classes=4 + len(FIELDS))
        direct = ClauseIntent(width, recurrent=True)
        for name, model in (('event', event), ('direct', direct)):
            model.load_state_dict(saved[name])
            model.eval().requires_grad_(False)
    else:
        event = fit(event_rows, features, width, 4, with_field=True)
        direct = fit(direct_rows, features, width, 2, with_field=False)
        atomic_save({'event': event.state_dict(), 'direct': direct.state_dict()}, candidate)
    writer = JointTurnValueWriter(64, feature_width=width).to(base.device)
    writer.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True)['joint']['weights'])
    writer.eval().requires_grad_(False)
    chars = FrozenTurnFeatures(decoder, tokenizer, depth=12)
    spans = {}
    calibration = forecast(event, direct, features, cal_set, writer, chars, spans)
    blind = forecast(event, direct, features, held_set, writer, chars, spans)
    floor = max(row['dialogue_margin'] for row in calibration if not row['expected'])
    metrics = {'calibration': finalize(calibration, floor), 'held': finalize(blind, floor)}
    apply = state_audit(metrics['held']['rows'], held, tokenizer)
    print('held:', metrics['held']['summary'], flush=True)
    print('state correctness:', {mode: sum(row['arms'][mode]['exact'] for row in apply)
                                 for mode in ('dialogue_write', 'structured_write')}, flush=True)
    torch.cuda.empty_cache()
    layers = load_recurrent_replacements(decoder, args.layers, protected[str(args.checkpoint)])
    with ExitStack() as stack:
        install(stack, layers)
        memory = evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))
    if (memory != study['memory'] or frozen_digest(runtime.organism, runtime.state) != individual
            or any(sha(Path(p)) != digest for p, digest in protected.items())):
        raise ValueError('A frozen input or the durable memory replay changed')
    result = {'sources': protected, 'individual_sha256': individual, 'decisions_sha256': sha(candidate),
              'event_training': len(event_rows), 'direct_training': len(direct_rows), 'floor': floor,
              'metrics': metrics, 'state_replay': apply, 'memory': memory}
    if args.verify:
        if any(expected[key] != value for key, value in result.items()):
            raise ValueError('Independent reload did not reproduce')
        store_json(args.output / 'reload.json', {'decisions_sha256': sha(candidate), 'seconds': perf_counter() - started})
    else:
        store_json(result_path, {**result, 'seconds': perf_counter() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'study', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('layers', nargs=2, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify', action='store_true')
    mode.add_argument('--resume-heads', action='store_true', help='Evaluate the saved heads after an interrupted run')
    main(parser.parse_args())
