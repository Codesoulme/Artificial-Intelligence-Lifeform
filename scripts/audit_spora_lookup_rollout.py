"""Train lookup state transitions and follow them through actual version reads."""

import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from scripts import audit_spora_lookup_transitions as transitions
from scripts.audit_spora_confirmed_revocation import classify, fit_probe
from scripts.audit_spora_recurrent_lookup_scope import ASK, READ, Episode
from scripts.spora_version_chat import VersionReplyActions, VersionStore, artifacts, base, compress_dimension, local_model
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_version_store import MemoryAddress


KEEP, ADD, REMOVE = transitions.KEEP, transitions.ADD, transitions.REMOVE
OPERATIONS = ((KEEP, KEEP), (KEEP, ADD), (REMOVE, REMOVE), (REMOVE, ADD), (ADD, KEEP), (KEEP, REMOVE))
FRAMES = {
    'train': (
        ('照着目前的单子来。', '把{b}也加进来。', '我把查阅全部叫停。', '原有的都作废，单查{b}。',
         '查阅名单另加{a}。', '只删去{b}，别的照原样。'),
        ('眼下的选择全部保留。', '旧安排不动，顺便核实{b}。', '先撤掉全部待查项。',
         '其他都放下，留{b}这一项。', '再给我补查{a}。', '我决定不核查{b}，其余不动。'),
        ('保持已有的查询范围。', '此前查的还是查，再看看{b}。', '无论选了几条，这轮一起撤回。',
         '别管刚提过什么了，改查{b}。', '我想一并核对{a}。', '请从查询范围移除{b}，其余照办。'),
    ),
    'calibration': (
        ('手上列的项目继续。', '在原有那些之外补看{b}。', '所有请求打住，之后再说。',
         '只要{b}，剩下的一律省略。', '可否合并查一下{a}？', '范围里只划掉{b}。'),
    ),
    'held': (
        ('刚才定的范围不改。', '原先说的仍有效，麻烦算上{b}。', '这轮一笔都不用往下翻了。',
         '此前的选择都收回，我想确认的是{b}。', '把{a}算进待办，跟现有的一起。',
         '刚提到的{b}不列在内，其他按原先的来。'),
        ('接着按手头的安排。', '现在的不用动，顺手把{b}并上。', '我暂时不用调阅任何一条。',
         '原来的安排清掉，由{b}顶替。', '我还想翻一翻{a}的记录。',
         '不用处理{b}，没点到的照查。'),
    ),
}
PATHS = ((1, 2, 0, 3, 0), (3, 4, 5, 0, 2), (1, 5, 1, 2, 4))
QUESTIONS = ('首版。', '当前版。', '上一版。')
RESET_TRAIN = (
    ('先别核对任何一份档案。', '先别停止核对任何一份档案。'),
    ('全部待查项目今天一起叫停。', '全部待查项目今天一起继续。'),
    ('整个查阅我撤回，重新提出再办。', '整个查阅仍照原计划办。'),
    ('别人希望我继续，但我决定所有记录都先不看。', '别人希望我先不看，但我决定所有记录都继续看。'),
    ('「照原样查」只是例子，我现在想全部取消。', '「全部取消」只是例子，我现在想照原样查。'),
    ('我自己去找，档里哪一项也不用给我翻。', '我自己去找的是别的，这些档案照查。'),
    ('本轮档案一个都先别读。', '本轮档案先别漏读任何一项。'),
    ('查阅现在全结束，没查完的也别做。', '查阅继续，没查完的接着做。'),
    ('请把全部核对任务撤掉。', '只撤掉其中一件核对任务。'),
    ('这次不查了，一个都不留。', '这次不查原项了，换成另一项。'),
)
RESET_CAL = (
    ('我现在不打算核查任何记录。', '我现在不打算停下任何核查。'),
    ('同事说照常，我决定这次统统放下。', '同事说统统放下，我决定这次照常。'),
    ('手头所有查阅到此为止。', '手头所有查阅继续推进。'),
    ('先收回整份名单，不做调阅。', '先收回名单里末尾那项，别的继续。'),
)
RESET_HELD = (
    ('universal', '眼下我一条档案也不需要调。', '眼下我每条档案都需要调，别停。'),
    ('negation', '查阅一个也别接着做。', '查阅一个也别给我停掉。'),
    ('speaker', '她想让你继续翻档，但我的意思是本次全部结束。', '她想让你结束翻档，但我的意思是本次全部照旧。'),
    ('quote', '「照原定的来」是我举例，实际这一轮全都不做了。', '「这一轮全都不做了」是我举例，实际照原定的来。'),
    ('scope', '原先提的每个查询一并划去。', '只把{b}划去，其他继续。'),
    ('switch', '之前那些事项一律作罢，无须接着查。', '之前那些事项一律作罢，改成查{b}。'),
)


def case_groups():
    return {split: [(tuple(MemoryAddress(*entry[i:i + 2]) for i in (0, 2)), forms)
                    for entry, forms in product(transitions.PEOPLE[split], FRAMES[split])]
            for split in FRAMES}


def cases_for(groups):
    rows = []
    for addresses, forms in groups:
        names = dict(zip(('a', 'b'), map(transitions.describe, addresses), strict=True))
        rows.extend(Episode('', addresses, (text.format(**names),), QUESTIONS[0], ASK, -1) for text in forms)
    return rows


def fit_operations(head, data, steps):
    targets = torch.tensor(OPERATIONS).repeat(len(data) // len(OPERATIONS), 1)
    optimizer = torch.optim.AdamW(head.operation.parameters(), lr=.001)
    head.train()
    for _ in range(steps):
        loss = F.cross_entropy(head.operation(data).flatten(0, 1), targets.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    head.eval()


def state_action(active):
    return READ + active.long().argmax().item() if active.sum().item() == 1 else ASK


def reset_examples(pairs):
    return [(text, label) for cancel, control in pairs for text, label in ((cancel, 1), (control, 0))]


def reset_threshold(rows):
    def utility(cutoff):
        found = [row for row in rows if row['predicted'] == 1 and row['confidence'] >= cutoff]
        good = sum(row['expected'] == 1 for row in found)
        bad = len(found) - good
        return good - 2 * bad, -bad, -cutoff

    return max((.5, .65, .8, .9, .95, .99, 1.01), key=utility)


def apply_reset(operations, rows, cutoff):
    accepted = torch.tensor([row['predicted'] == 1 and row['confidence'] >= cutoff for row in rows])
    return operations.masked_fill(accepted[:, None], REMOVE)


def reset_cases():
    cases, labels = [], []
    controls = [(KEEP, KEEP)] * 4 + [(KEEP, REMOVE), (REMOVE, ADD)]
    for entry in transitions.PEOPLE['held']:
        addresses = tuple(MemoryAddress(*entry[i:i + 2]) for i in (0, 2))
        for (family, cancel, control), operations in zip(RESET_HELD, controls, strict=True):
            cases.extend(Episode(family, addresses, (text.format(b=transitions.describe(addresses[1])),),
                                 QUESTIONS[2], ASK, -1) for text in (cancel, control))
            labels.extend(((REMOVE, REMOVE), operations))
    return cases, torch.tensor(labels)


def replay_resets(cases, targets, predicted):
    rows, plans, states = [], [], []
    for row, target, guess in zip(cases, targets, predicted, strict=True):
        for initial in ((False, False), (True, False), (False, True), (True, True)):
            active = torch.tensor(initial)
            expected = transitions.apply_operations(active, target)
            actual = transitions.apply_operations(active, guess)
            wanted = state_action(expected)
            rows.append(Episode(row.scope, row.addresses, row.events, row.question, wanted,
                                2 if wanted >= READ else -1))
            plans.append((state_action(actual), 2))
            states.append(dict(initial=list(initial), expected_state=expected.tolist(), actual_state=actual.tolist(),
                               cancel=target.tolist() == [REMOVE, REMOVE]))
    return rows, plans, states


@torch.inference_mode()
def rollout(rows, operations):
    evaluation, decisions, trace = [], [], []
    width = len(OPERATIONS)
    for block in range(0, len(rows), width):
        cases, predictions = rows[block:block + width], operations[block:block + width]
        for path in PATHS:
            expected = actual = torch.tensor([True, False])
            dialogue = [f'先查{transitions.describe(cases[0].addresses[0])}。']
            trace.append([])
            for step, index in enumerate(path):
                dialogue.extend(cases[index].events)
                expected = transitions.apply_operations(expected, torch.tensor(OPERATIONS[index]))
                actual = transitions.apply_operations(actual, predictions[index])
                version = (step + block // width) % len(QUESTIONS)
                action = state_action(expected)
                evaluation.append(Episode(str(index), cases[index].addresses, tuple(dialogue), QUESTIONS[version], action,
                                          version if action >= READ else -1))
                decisions.append((state_action(actual), version))
                trace[-1].append(dict(operation=index, expected=expected.tolist(), actual=actual.tolist(),
                                      predicted_operation=predictions[index].tolist()))
                dialogue.append(QUESTIONS[version])
    return evaluation, decisions, trace


def main(args):
    start = perf_counter()
    torch.set_num_threads(2)
    manifest = json.loads((args.prior / 'result.json').read_text(encoding='utf-8'))
    archive = args.prior / 'three-versions.sqlite3'
    checkpoint = args.prior / 'clause.pt'
    reader_path, normalizer, writer, binding = artifacts(args.reader, args.normalizer)
    paths = (archive, checkpoint, reader_path, normalizer, writer, args.recurrent[0],
             *(folder / 'attention.pt' for folder in args.recurrent[1:]))
    sources = {path: base.sha(path) for path in paths}
    expected = {Path(path).resolve(): digest for path, digest in manifest['sources'].items()}
    expected[archive.resolve()] = manifest['archive_sha256']
    expected[checkpoint.resolve()] = manifest['results']['clause']['checkpoint_sha256']
    if any(expected.get(path.resolve()) != digest for path, digest in sources.items()):
        raise ValueError('A model or archive differs from the source experiment')
    values = {int(number): value for number, value in manifest['values'].items()}
    cases = {name: cases_for(group) for name, group in case_groups().items()}
    if args.global_reset:
        cases['reset'], reset_labels = reset_cases()
    if any(value in text for group in cases.values() for row in group for text in row.events for value in values.values()):
        raise ValueError('A query exposes a value in the archive')
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    results = {}
    with local_model(args.recurrent) as model, VersionStore(archive, binding) as store:
        extract = transitions.FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer)
        features = transitions.transition_features(extract, cases, utterance_features=True, clause_features=True)
        inputs = {split: history[:, 0] for split, (_, history, _) in features.items()}
        head = transitions.LookupTransitions(model.config.hidden_size, operation_width=inputs['train'].shape[-1])
        head.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        head.eval()
        weights = torch.load(reader_path, map_location='cpu', weights_only=True)
        reader = VersionReplyActions(model.config.hidden_size, weights['scaffold_tokens'], slots=2)
        reader.load_state_dict(weights, strict=True)
        reader.eval().to(model.device)
        embedding = model.get_input_embeddings().weight.detach()
        codebook = F.normalize(embedding.index_select(1, compress_dimension(embedding.shape[1]).to(model.device)).float(), dim=-1)
        reset_report = {}
        arms = ('previous', 'state_updates')
        if args.global_reset:
            arms += ('global_reset',)
            reset_head = fit_probe(extract, reset_examples(RESET_TRAIN), 2, clauses=True)
            reset_path = args.output / 'global-reset.pt'
            atomic_save(reset_head.state_dict(), reset_path)
            reset_head.load_state_dict(torch.load(reset_path, map_location='cpu', weights_only=True), strict=True)
            calibrated = classify(reset_head, extract, reset_examples(RESET_CAL), clauses=True)
            cutoff = reset_threshold(calibrated)
            predictions = {}
            for split in ('calibration', 'held', 'reset'):
                labels = (reset_labels.eq(REMOVE).all(1) if split == 'reset' else
                          torch.arange(len(cases[split])).remainder(len(OPERATIONS)).eq(2))
                examples = [(row.events[0], int(label)) for row, label in zip(cases[split], labels, strict=True)]
                predictions[split] = classify(reset_head, extract, examples, clauses=True)
            reset_report = dict(cutoff=cutoff, calibration=calibrated, predictions=predictions,
                                checkpoint_sha256=base.sha(reset_path))
        for name in arms:
            if name == 'state_updates':
                fit_operations(head, inputs['train'], args.steps)
                output = args.output / 'state-updates.pt'
                atomic_save(head.state_dict(), output)
                head.load_state_dict(torch.load(output, map_location='cpu', weights_only=True), strict=True)
            with torch.inference_mode():
                metrics = {}
                for split in ('calibration', 'held'):
                    labels = torch.tensor(OPERATIONS).repeat(len(inputs[split]) // len(OPERATIONS), 1)
                    predicted = head.operation(inputs[split]).argmax(-1)
                    if name == 'global_reset':
                        predicted = apply_reset(predicted, predictions[split], cutoff)
                    metrics[split] = dict(correct=predicted.eq(labels).all(1).sum().item(), total=len(labels))
                    if split == 'held':
                        rows, plans, trace = rollout(cases[split], predicted)
                result = transitions.evaluate(rows, plans, store.catalog(), store, model, tokenizer, codebook, reader, values=values)
                if args.global_reset:
                    reset_ops = head.operation(inputs['reset']).argmax(-1)
                    if name == 'global_reset':
                        reset_ops = apply_reset(reset_ops, predictions['reset'], cutoff)
                    reset_rows, reset_plans, states = replay_resets(cases['reset'], reset_labels, reset_ops)
                    reset_result = transitions.evaluate(reset_rows, reset_plans, store.catalog(), store, model,
                                                        tokenizer, codebook, reader, values=values)
                    for state, row in zip(states, reset_result['rows'], strict=True):
                        row.update(state)
                    result['reset_followup'] = reset_result
            empty = Counter()
            for state, row in zip((state for dialogue in trace for state in dialogue), result['rows'], strict=True):
                if not any(state['expected']):
                    empty.update(total=1, read=row['selected'] is not None)
            result.update(operations=metrics, trace=trace, empty_scope=dict(empty),
                          complete_dialogues=sum(all(turn['actual'] == turn['expected'] for turn in dialogue) for dialogue in trace))
            results[name] = result
            print(name, metrics, result['summary'], empty, result['complete_dialogues'], '/', len(trace), flush=True)
            if args.global_reset:
                print('reset followup', name, result['reset_followup']['summary'], flush=True)
    if any(base.sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('A protected model or archive changed')
    store_json(args.output / 'result.json', dict(results=results, training=len(cases['train']), steps=args.steps,
                                               reset=reset_report,
                                               sources={str(path): digest for path, digest in sources.items()},
                                               seconds=perf_counter() - start))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('prior', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--recurrent', type=Path, nargs=3, required=True)
    cli.add_argument('--reader', type=Path, default=Path('runs/spora-dual-version-reply-v1'))
    cli.add_argument('--normalizer', type=Path, default=Path('runs/spora-version-normalizer-v1'))
    cli.add_argument('--steps', type=int, default=120)
    cli.add_argument('--global-reset', action='store_true')
    args = cli.parse_args()
    if args.steps < 1:
        cli.error('--steps must be positive')
    main(args)
