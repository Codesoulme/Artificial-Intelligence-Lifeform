"""Compare cancellation decisions and verify clarification with actual version reads."""

import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from scripts import audit_spora_lookup_rollout as rollout
from scripts.audit_spora_confirmed_revocation import fit_probe, classify
from scripts.audit_spora_lookup_transitions import FrozenTurnFeatures, LookupTransitions, evaluate, transition_features
from scripts.audit_spora_recurrent_lookup_scope import ASK, Episode, PEOPLE, READ
from scripts.spora_version_chat import VersionReplyActions, VersionStore, artifacts, base, compress_dimension, local_model
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_version_store import MemoryAddress


PROCEED, CANCEL, CLARIFY = range(3)
TRAIN_MORE = (
    ('核查任务别延续了，一件也不做。', '核查任务别停下来，一件也不能少。', '这些核查是不是该先停？'),
    ('不用再看任何一条记录。', '不用停止查看任何一条记录。', '听说不用看了，咱们是不是也这样？'),
    ('本轮就到这儿，我不再请求翻档。', '本轮接着查，我没有要求停止。', '终止的话会不会更好？我没决定。'),
    ('我已决定把这些检查悉数撤销。', '我已决定让这些检查全部继续。', '这些检查或许撤掉，或许照办。'),
    ('不必把名单中任何事项交给你核实。', '名单里的事项都要你核实，先别收尾。', '你是说名单里的事项都先略过吗？'),
    ('对方说全部不要查，我也赞同他的停查决定。', '对方说全部不要查，但这轮我决定照查。', '对方说全部不要查，我还在考虑。'),
    ('从本次起不再调出这一批档案。', '这个批次的档案接着调取。', '我有点拿不准这批档案要不要继续调。'),
    ('方才那批我全部撤回。', '方才那批以外，再查另一项记录。', '你觉得要撤回的是全批还是一项？'),
)
CALIBRATION = (
    ('我决定本轮一概不再调档。', '我决定本轮一概照原计划。', '也许本轮先搁置，你怎么看？'),
    ('到这里为止，不要翻查余下的。', '不要到这里为止，余下的也翻查。', '余下那些是不是先不动比较好？'),
    ('他说保持原要求，但我不同意，全部收回。', '他说全部收回，但我不同意，保持原要求。', '他说全部收回，这事我没有定。'),
    ('哪笔都无需再调阅了。', '哪笔都仍需调阅，继续。', '有一笔要撤掉，具体是哪笔我还没想好。'),
)
HELD = (
    ('negation', '每一笔核验都先中止。', '每一笔核验都不得中止。', '每一笔核验都得中止吗？'),
    ('universal', '这回没有哪份资料还需要翻出来。', '这一回没有哪份资料可以漏查。', '这一回究竟还有哪份需要翻，我没想定。'),
    ('speaker', '搭档倾向照旧，我的决定是把全部调档叫停。', '搭档倾向叫停，我的决定是全部按原样调档。', '搭档倾向叫停，我暂未答应。'),
    ('quote', '「继续办理」只是引文，实际把查询统统作废。', '「查询统统作废」只是引文，实际继续办理。', '原话是「查询统统作废」，我只是在复述。'),
    ('scope', '这批查询整个失效，别再处理。', '这批查询仍算数，请按原来的处理。', '先把其中一笔排除，具体哪笔还不确定。'),
    ('conditional', '不用等了，我现在确定全撤。', '不用等了，我现在确定全留。', '如果今天确认不了，可能就要全撤。'),
)
MATCHED_TRAIN = (
    ('所有调档都应停办。', '所有调档都不应停办。', '所有调档都应停办吗？'),
    ('别让每项核查继续。', '别让每项核查停止。', '是否该让每项核查停止？'),
    ('这批记录统统不用查。', '这批记录统统不得漏查。', '这批记录是不是统统不用查？'),
    ('每份待查材料都别翻了。', '每份待查材料都别忘了翻。', '每份待查材料都别翻了吗？'),
    ('同事说全部停办，我同意。', '同事说全部停办，我不同意。', '同事说全部停办，我还没表态。'),
    ('旁人建议接着查，我决定全部停下。', '旁人建议停下，我决定全部接着查。', '旁人建议停下，我尚未决定。'),
    ('「整批终止」是我的最终决定。', '「整批终止」是他的话，我决定照旧。', '「整批终止」只是我念出的原文。'),
    ('有人说不能停查，我的意思是可以停查并且现在停。', '有人说可以停查，我的意思是不能停查。', '有人说可以停查，我暂不发表意见。'),
)
MATCHED_HELD = (
    ('must', '这些检索都须终止。', '这些检索都不许终止。', '这些检索都须终止么？'),
    ('any', '待查的卷宗一份都别开。', '待查的卷宗一份都别忘了开。', '待查的卷宗是不是一份都别开？'),
    ('endorsement', '组长提出先把全部查询作废，我接受。', '组长提出先把全部查询作废，我反对，照办。', '组长提出先把全部查询作废，我还没有意见。'),
    ('attribution', '旁边的人主张保留，不过我批准撤销整单。', '旁边的人主张撤销，不过我批准保留整单。', '旁边的人主张撤销，我只是转达。'),
    ('quotation', '「照原样做」是讨论材料，我的决定是停止全部。', '「停止全部」是讨论材料，我的决定是照原样做。', '材料里的说法是「停止全部」，这事我还没拍板。'),
    ('condition', '条件已经满足，现在把整批都停下。', '条件没有满足，现在这批接着查。', '假如条件满足，是不是就把整批都停下？'),
)
RESOLUTION = (
    ('cancel', '我确定这一次整批都不再查询了。', CANCEL),
    ('proceed', '确认维持当前查阅范围，请接着办。', PROCEED),
)


def labeled(triples):
    return [(text, label) for texts in triples
            for text, label in zip(texts, (CANCEL, PROCEED, CLARIFY), strict=True)]


def matched_summary(cases, picks):
    comparisons = tuple(zip(cases, picks, strict=True))
    counts = Counter()
    for (_, _, wanted), actual in comparisons:
        counts.update(correct=wanted == actual, false_cancel=wanted != CANCEL and actual == CANCEL,
                      unnecessary_wait=wanted != CLARIFY and actual == CLARIFY)
    counts.update(total=len(comparisons), complete_triples=sum(
        all(wanted == actual for (_, _, wanted), actual in comparisons[begin:begin + 3])
        for begin in range(0, len(comparisons), 3)))
    return counts


def decision(row, cutoff):
    return row['predicted'] if row['confidence'] >= cutoff else CLARIFY


def calibrate(rows):
    def utility(cutoff):
        pairs = [(row['expected'], decision(row, cutoff)) for row in rows]
        correct = sum(expected == guess for expected, guess in pairs)
        wrong_commit = sum(expected != guess and guess != CLARIFY for expected, guess in pairs)
        return correct - 2 * wrong_commit, -wrong_commit, -cutoff

    return max((.34, .5, .65, .8, .9, .95, .99, 1.01), key=utility)


def step(active, waiting, label, operations):
    if label == CLARIFY:
        return active, True
    if label == CANCEL:
        return torch.zeros_like(active), False
    if waiting:
        return active, False
    return rollout.transitions.apply_operations(active, operations), False


def followups(examples, predictions, pairs, ordinary_ops, resolutions):
    episodes, plans, trace = [], [], []
    for (family, text, expected), predicted, operations in zip(examples, predictions, ordinary_ops, strict=True):
        replies = resolutions if expected == CLARIFY else (('none', '', None, None),)
        for (addresses, ops), initial, (reply_name, reply, wanted, guessed) in product(
                zip(pairs, operations, strict=True), ((False, False), (True, False), (False, True), (True, True)), replies):
            active = torch.tensor(initial)
            keep = torch.tensor([rollout.KEEP, rollout.KEEP])
            expected_state, expected_wait = step(active, False, expected, keep)
            actual_state, actual_wait = step(active, False, predicted, ops)
            turns = [((), expected_state, expected_wait, actual_state, actual_wait)]
            if wanted is not None:
                after = step(expected_state, expected_wait, wanted, keep)
                changed = step(actual_state, actual_wait, guessed, keep)
                turns.append(((reply,), *after, *changed))
            for extra, target, target_wait, actual, waiting in turns:
                expected_action = ASK if target_wait else rollout.state_action(target)
                action = ASK if waiting else rollout.state_action(actual)
                episodes.append(Episode(family, addresses, (text, *extra), '上一版。', expected_action,
                                        2 if expected_action >= READ else -1))
                plans.append((action, 2))
                trace.append(dict(kind='after_reply' if extra else 'initial', reply=reply_name if extra else 'none',
                                  expected_intent=expected, predicted_intent=predicted, initial=list(initial),
                                  expected_state=target.tolist(), actual_state=actual.tolist(),
                                  expected_wait=target_wait, actual_wait=waiting))
    return episodes, plans, trace


def main(args):
    start = perf_counter()
    torch.set_num_threads(2)
    prior = json.loads((args.prior / 'result.json').read_text(encoding='utf-8'))
    archive_meta = json.loads((args.archive / 'result.json').read_text(encoding='utf-8'))
    archive = args.archive / 'three-versions.sqlite3'
    binary_path = args.prior / 'global-reset.pt'
    actions_path = args.prior / 'state-updates.pt'
    reader_path, normalizer, writer, binding = artifacts(args.reader, args.normalizer)
    recurrent = (args.recurrent[0], *(folder / 'attention.pt' for folder in args.recurrent[1:]))
    checked = (archive, binary_path, actions_path, reader_path, normalizer, writer, *recurrent)
    sources = {path: base.sha(path) for path in checked}
    expected = {Path(path).resolve(): digest for path, digest in prior['sources'].items()}
    expected[binary_path.resolve()] = prior['reset']['checkpoint_sha256']
    if sources[archive] != archive_meta['archive_sha256'] or any(
            expected.get(path.resolve()) != digest for path, digest in sources.items() if path != actions_path):
        raise ValueError('The previous experiment inputs do not match')

    pairs = [tuple(MemoryAddress(*entry[index:index + 2]) for index in (0, 2)) for entry in PEOPLE['held']]
    train = rollout.reset_examples(rollout.RESET_TRAIN) + labeled(TRAIN_MORE)
    calibration = labeled(CALIBRATION)
    held = [(family, text, label) for family, *texts in HELD for text, label in labeled([texts])]
    previous_size = len(held)
    if args.matched:
        held.extend((family, text, label) for family, *texts in MATCHED_HELD for text, label in labeled([texts]))
    held_labeled = [(text, label) for _, text, label in held]
    answers = {int(number): value for number, value in archive_meta['values'].items()}
    if any(value in text for text, _ in held_labeled for value in answers.values()):
        raise ValueError('A held prompt contains an archived value')
    augmented = train + labeled(MATCHED_TRAIN) if args.matched else train
    if {text for text, _ in augmented} & {text for text, _ in calibration + held_labeled}:
        raise ValueError('Training wording overlaps evaluation')
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    results = {}
    with local_model(args.recurrent) as model, VersionStore(archive, binding) as store:
        extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer)
        probes = {}
        for name, training in (('three_way', train), ('matched', augmented)) if args.matched else (('three_way', train),):
            probe = fit_probe(extract, training, 3, clauses=True)
            path = args.output / f'{name.replace("_", "-")}.pt'
            atomic_save(probe.state_dict(), path)
            probe.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
            cal = classify(probe, extract, calibration, clauses=True)
            probes[name] = probe, calibrate(cal), cal, base.sha(path)

        values = [Episode(family, addresses, (text,), '上一版。', ASK, -1)
                  for (family, text, _), addresses in product(held, pairs)]
        _, history, _ = transition_features(extract, {'held': values}, utterance_features=True,
                                            clause_features=True)['held']
        operations = LookupTransitions(model.config.hidden_size, operation_width=history.shape[-1])
        operations.load_state_dict(torch.load(actions_path, map_location='cpu', weights_only=True), strict=True)
        binary = torch.nn.Sequential(torch.nn.LayerNorm(history.shape[-1]),
                                    torch.nn.Linear(history.shape[-1], 32), torch.nn.GELU(), torch.nn.Linear(32, 2))
        binary.load_state_dict(torch.load(binary_path, map_location='cpu', weights_only=True), strict=True)
        with torch.inference_mode():
            ordinary_ops = operations.eval().operation(history[:, 0]).argmax(-1).unflatten(0, (len(held), len(pairs)))
        weights = torch.load(reader_path, map_location='cpu', weights_only=True)
        reader = VersionReplyActions(model.config.hidden_size, weights['scaffold_tokens'], slots=2)
        reader.load_state_dict(weights, strict=True)
        reader.eval().to(model.device)
        embedding = model.get_input_embeddings().weight.detach()
        columns = compress_dimension(embedding.shape[1]).to(model.device)
        codebook = F.normalize(embedding.index_select(1, columns).float(), dim=-1)
        arms = [('binary', binary.eval(), prior['reset']['cutoff'])]
        arms.extend((name, probe, cutoff) for name, (probe, cutoff, _, _) in probes.items())
        for name, head, cutoff in arms:
            classified = classify(head, extract, held_labeled, clauses=True)
            replies = classify(head, extract, [(text, label) for _, text, label in RESOLUTION], clauses=True)
            if name == 'binary':
                picks = [CANCEL if row['predicted'] == CANCEL and row['confidence'] >= cutoff else PROCEED
                         for row in classified + replies]
            else:
                picks = [decision(row, cutoff) for row in classified + replies]
            resolutions = [(*row, pick) for row, pick in zip(RESOLUTION, picks[len(held):], strict=True)]
            examples, plans, trace = followups(held, picks[:len(held)], pairs, ordinary_ops, resolutions)
            result = evaluate(examples, plans, store.catalog(), store, model, tokenizer, codebook, reader, values=answers)
            grouped = {}
            for row, state in zip(result['rows'], trace, strict=True):
                row.update(state)
                group = grouped.setdefault('after_reply' if state['kind'] == 'after_reply'
                                           else ('continue', 'cancel', 'clarify')[state['expected_intent']], Counter())
                group.update(total=1, range_correct=state['actual_state'] == state['expected_state'],
                             wait_correct=state['actual_wait'] == state['expected_wait'],
                             unexpected_read=row['selected'] is not None and row['expected_record'] is None,
                             missed_read=row['selected'] is None and row['expected_record'] is not None)
            result.update(groups=grouped, cutoff=cutoff, classifications=classified, resolutions=resolutions, picks=picks)
            if args.matched:
                result['intent'] = {
                    'prior': matched_summary(held[:previous_size], picks[:previous_size]),
                    'fresh': matched_summary(held[previous_size:], picks[previous_size:len(held)]),
                }
                print('intent', name, result['intent'], flush=True)
            results[name] = result
            print(name, cutoff, result['summary'], grouped, resolutions, flush=True)
    if any(base.sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('A model input or archive changed')
    _, _, calibration_rows, checkpoint_sha256 = probes['three_way']
    store_json(args.output / 'result.json', dict(results=results, training=len(train), calibration=calibration_rows,
                                               checkpoint_sha256=checkpoint_sha256,
                                               matched_training=len(augmented) if args.matched else None,
                                               matched_calibration=probes['matched'][2] if args.matched else None,
                                               matched_sha256=probes['matched'][3] if args.matched else None,
                                               sources={str(path): digest for path, digest in sources.items()},
                                               seconds=perf_counter() - start))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('archive', type=Path)
    cli.add_argument('prior', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--recurrent', type=Path, nargs=3, required=True)
    cli.add_argument('--reader', type=Path, default=Path('runs/spora-dual-version-reply-v1'))
    cli.add_argument('--normalizer', type=Path, default=Path('runs/spora-version-normalizer-v1'))
    cli.add_argument('--matched', action='store_true')
    main(cli.parse_args())
