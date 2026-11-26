"""Audit context-dependent confirmation replies against stored version records."""

import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from scripts.audit_spora_confirmed_revocation import classify, fit_probe
from scripts.audit_spora_lookup_reset_clarification import CANCEL, CLARIFY, PROCEED, calibrate, decision, step
from scripts.audit_spora_lookup_rollout import KEEP, state_action
from scripts.audit_spora_recurrent_lookup_scope import ASK, READ, Episode, PEOPLE, evaluate
from scripts.spora_version_chat import FrozenTurnFeatures, VersionReplyActions, VersionStore, artifacts, base, compress_dimension, local_model
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_version_store import MemoryAddress


AGREE, DISAGREE = 3, 4

# Each question pair switches what an affirmative reply authorizes.
QUESTIONS = {
    'train': (('您是要停止这轮查阅吗？', '您是要继续这轮查阅吗？'),
              ('现在把整组查询撤销，对吗？', '现在把整组查询保留，对吗？'),
              ('您的决定是清空待查范围吗？', '您的决定是沿用待查范围吗？')),
    'calibration': (('您的意思是结束当前调阅吗？', '您的意思是维持当前调阅吗？'),
                    ('前面提的请求都要撤掉，是吗？', '前面提的请求都要接着办，是吗？')),
    'held': (('手头这批档案还要照查，您确认吗？', '手头这批档案就不再查了，您确认吗？'),
             ('刚定下的查询事项一并作罢吗？', '刚定下的查询事项照原样执行吗？')),
    'fresh': (('您的决定是把当前整批查询收回吗？', '您的决定是让当前整批查询照常吗？'),
              ('这些记录都先别翻，对吗？', '这些记录都按原要求翻，对吗？')),
    'speaker': (('现在把清单上的查阅全停下来，可以吗？', '现在把清单上的查阅全做下去，可以吗？'),),
}
REPLIES = {
    'train': (('对，就照你说的。', '不，我的意思正好相反。', '先等等，我还得考虑。'),
              ('是，没错。', '不是，反过来。', '稍后再告诉你。'),
              ('我确认。', '我否认你说的那个决定。', '刚刚别人说过，算不算我还不知道。')),
    'calibration': (('嗯，正是。', '不对，请按相反的来。', '这件事还没有敲定。'),
                    ('没说错，我同意。', '恰恰不是你说的那样。', '请再等等。')),
    'held': (('是的。', '说反了。', '容我再想一想。'),
             ('就按你刚说的办。', '我选另一种。', '他说「同意」，我没有表态。')),
    'fresh': (('没错，就是这个意思。', '不对，我要的是反面的决定。', '先别定，容我斟酌。'),
              ('你理解对了。', '你理解反了。', '同伴同意了，我还在考虑。')),
    'speaker': (('她已经赞成，我也同意你刚问的。', '她已经赞成，我却不同意你刚问的。',
                 '她已经赞成，我本人还在犹豫。'),
                ('他在反对，不过我赞成你这个提议。', '他在反对，我也不接受你这个提议。',
                 '他在反对，我只是把他的意见告诉你。'),
                ('那是别人的话，我的答复是可以。', '那是别人的话，我的答复是不可以。',
                 '那是别人的话，我自己还没有答案。'),
                ('我想明白了，照你提的做。', '我想明白了，按你提的反面做。',
                 '再容我想想，先别替我拍板。')),
}
EXPLICIT = {
    'train': (('不管刚才怎么问，我现在决定停止查阅。', CANCEL),
              ('不用撤销，按手头范围继续。', PROCEED),
              ('整个这次我不再需要了。', CANCEL),
              ('这些记录还是接着调吧。', PROCEED)),
    'calibration': (('撤回我这次全部核查要求。', CANCEL), ('请依照原有名单核查。', PROCEED)),
    'held': (('我确认整批都不再调档。', CANCEL), ('一笔都不撤，接着查。', PROCEED),
             ('谁都不用查了，这就是我的决定。', CANCEL), ('把原来定的全留下，继续办理。', PROCEED)),
    'fresh': (('我已决定终止本次所有查阅。', CANCEL), ('我授权按原范围继续核对。', PROCEED),
              ('把现有的请求全撤回吧。', CANCEL), ('每个原定事项都照常办理。', PROCEED)),
    'speaker': (('别人说要照查，最后我决定这一次全部撤回。', CANCEL),
                ('别人说要撤回，最后我决定全部照常查。', PROCEED)),
}
SPEAKER_TRAIN = (
    ('同事点头了，我也赞同你的建议。', '同事点头了，我的答复是否定的。', '同事点头了，我还不能给你答复。'),
    ('旁人说不，但我答应你的提议。', '旁人说不，我同样拒绝这个提议。', '旁人说不，这不代表我已有决定。'),
    ('我已考虑过，接受你的方案。', '我已考虑过，不接受你的方案。', '我得再考虑，暂时不要执行。'),
    ('他提了他的想法，我本人说行。', '他提了他的想法，我本人说不行。', '他提了他的想法，我本人没表态。'),
)
SPEAKER_CALIBRATION = (
    ('朋友赞成，我也认可你刚才提出的。', '朋友赞成，但我不认可你刚才提出的。', '朋友赞成，我还没有定下来。'),
    ('搭档说不，我这里答复可以。', '搭档说不，我这里也拒绝。', '搭档说不，我只是在传话。'),
)


def cases(split):
    result = []
    for question, agreement in question_actions(split).items():
        opposite = CANCEL if agreement == PROCEED else PROCEED
        for replies in REPLIES[split]:
            result.extend((question, reply, label, kind) for (reply, label), kind in zip(
                zip(replies, (agreement, opposite, CLARIFY), strict=True), ('yes', 'no', 'unclear'), strict=True))
        result.extend((question, reply, label, 'explicit') for reply, label in EXPLICIT[split])
    return result


def examples(rows, contextual):
    return [(f'孢子：{question}\n用户：{reply}' if contextual else reply, label)
            for question, reply, label, _ in rows]


def response_examples(rows):
    relative = {'yes': AGREE, 'no': DISAGREE}
    return list(dict.fromkeys((reply, relative.get(kind, label)) for _, reply, label, kind in rows))


def speaker_examples(triples):
    return [(reply, label) for triple in triples
            for reply, label in zip(triple, (AGREE, DISAGREE, CLARIFY), strict=True)]


def question_actions(split):
    actions = {}
    for index, pair in enumerate(QUESTIONS[split]):
        labels = (PROCEED, CANCEL) if split == 'held' and index == 0 else (CANCEL, PROCEED)
        actions.update(zip(pair, labels, strict=True))
    return actions


def resolve_response(proposed, response):
    if response == AGREE:
        return proposed
    if response == DISAGREE:
        return CANCEL if proposed == PROCEED else PROCEED
    return response


def replay(rows, picks, pairs):
    episodes, plans, trace = [], [], []
    keep = torch.full((2,), KEEP)
    for (question, reply, label, kind), predicted in zip(rows, picks, strict=True):
        for addresses, opening in product(pairs, ((False, False), (True, False), (False, True), (True, True))):
            active = torch.tensor(opening)
            expected, target_wait = step(active, True, label, keep)
            current, waiting = step(active, True, predicted, keep)
            want = ASK if target_wait else state_action(expected)
            plan = ASK if waiting else state_action(current)
            episodes.append(Episode(kind, addresses, (question, reply), '上一版。', want, 2 if want >= READ else -1))
            plans.append((plan, 2))
            trace.append(dict(label=label, predicted=predicted, opening=list(opening), pending=waiting,
                              scope=current.tolist(), expected_scope=expected.tolist()))
    return episodes, plans, trace


def main(args):
    start = perf_counter()
    torch.set_num_threads(2)
    prior = json.loads((args.prior / 'result.json').read_text(encoding='utf-8'))
    archive_meta = json.loads((args.archive / 'result.json').read_text(encoding='utf-8'))
    archive = args.archive / 'three-versions.sqlite3'
    prior_head = args.prior / 'matched.pt'
    reader_path, normalizer, writer, binding = artifacts(args.reader, args.normalizer)
    paths = (archive, prior_head, reader_path, normalizer, writer, args.recurrent[0],
             *(directory / 'attention.pt' for directory in args.recurrent[1:]))
    sources = {path: base.sha(path) for path in paths}
    expected = {Path(path).resolve(): digest for path, digest in prior['sources'].items()}
    expected[prior_head.resolve()] = prior['matched_sha256']
    if sources[archive] != archive_meta['archive_sha256'] or any(
            expected.get(path.resolve()) != digest for path, digest in sources.items()):
        raise ValueError('An input differs from the preceding experiment')
    splits = ('train', 'calibration', 'held', 'fresh') if args.stored_action else ('train', 'calibration', 'held')
    if args.speaker_contrasts:
        splits += ('speaker',)
    groups = {name: cases(name) for name in splits}
    distinct_replies = [{reply for _, reply, _, _ in rows} for rows in groups.values()]
    if args.speaker_contrasts:
        distinct_replies[0].update(reply for reply, _ in speaker_examples(SPEAKER_TRAIN))
        distinct_replies[1].update(reply for reply, _ in speaker_examples(SPEAKER_CALIBRATION))
    if any(left & right for index, left in enumerate(distinct_replies) for right in distinct_replies[index + 1:]):
        raise ValueError('Confirmation replies must be disjoint across splits')
    addresses = [tuple(MemoryAddress(*entry[index:index + 2]) for index in (0, 2)) for entry in PEOPLE['held']]
    values = {int(number): text for number, text in archive_meta['values'].items()}
    if any(value in part for rows in groups.values() for question, reply, _, _ in rows
           for part in (question, reply) for value in values.values()):
        raise ValueError('A prompt contains a stored value')
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    results = {}
    with local_model(args.recurrent) as model, VersionStore(archive, binding) as store:
        decoder = SimpleNamespace(base=model)
        shallow = FrozenTurnFeatures(decoder, tokenizer)
        width = model.config.hidden_size
        baseline = nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, 32), nn.GELU(), nn.Linear(32, 3))
        baseline.load_state_dict(torch.load(prior_head, weights_only=True, map_location='cpu'), strict=True)
        weights = torch.load(reader_path, weights_only=True, map_location='cpu')
        reader = VersionReplyActions(width, weights['scaffold_tokens'], slots=2)
        reader.load_state_dict(weights, strict=True)
        reader.eval().to(model.device)
        embedding = model.get_input_embeddings().weight.detach()
        indices = compress_dimension(embedding.shape[1]).to(embedding.device)
        codebook = F.normalize(embedding.index_select(1, indices).float(), dim=-1)
        arms = (('baseline', False, 12), ('reply_only', False, 12),
                ('with_question', True, 12), ('recurrent', True, 23))
        if args.stored_action:
            arms = (('stored_action_12', False, 12), ('stored_action_23', False, 23))
        if args.speaker_contrasts:
            arms = (('stored_action_12', False, 12), ('speaker_contrasts', False, 12))
        for name, contextual, depth in arms:
            extract = shallow if depth == 12 else FrozenTurnFeatures(decoder, tokenizer, depth=depth, single_sequence=True)
            checkpoint = prior_head if name == 'baseline' else args.output / f'{name}.pt'
            if name == 'baseline':
                head = baseline.eval()
            else:
                training = response_examples(groups['train']) if args.stored_action else examples(groups['train'], contextual)
                if name == 'speaker_contrasts':
                    training += speaker_examples(SPEAKER_TRAIN)
                head = fit_probe(extract, training, 5 if args.stored_action else 3, clauses=True)
                atomic_save(head.state_dict(), checkpoint)
                head.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'), strict=True)
            calibration_input = (response_examples(groups['calibration']) if args.stored_action else
                                 examples(groups['calibration'], contextual))
            if args.speaker_contrasts:
                calibration_input += speaker_examples(SPEAKER_CALIBRATION)
            calibration = classify(head, extract, calibration_input, clauses=True)
            cutoff = prior['results']['matched']['cutoff'] if name == 'baseline' else calibrate(calibration)
            for split in splits[2:]:
                rows = groups[split]
                if args.stored_action:
                    held = classify(head, extract, response_examples(rows), clauses=True)
                    replies = {row['text']: decision(row, cutoff) for row in held}
                    proposals = question_actions(split)
                    picks = [resolve_response(proposals[question], replies[reply]) for question, reply, _, _ in rows]
                else:
                    held = classify(head, extract, examples(rows, contextual), clauses=True)
                    picks = [decision(row, cutoff) for row in held]
                result = evaluate_choices(rows, picks, addresses, store, model, tokenizer, codebook, reader, values)
                result.update(cutoff=cutoff, calibration=calibration, held=held, picks=picks,
                              checkpoint_sha256=base.sha(checkpoint))
                results[f'{name}_{split}' if args.stored_action else name] = result
                print(name, split, cutoff, result['decisions'], result['totals'], result['summary'], flush=True)
    if any(base.sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('A model input or archive changed')
    training_size = len(response_examples(groups['train'])) if args.stored_action else len(groups['train'])
    store_json(args.output / 'result.json', dict(results=results, training=training_size,
                                               augmented_training=training_size + len(speaker_examples(SPEAKER_TRAIN))
                                               if args.speaker_contrasts else None,
                                               sources={str(path): digest for path, digest in sources.items()},
                                               seconds=perf_counter() - start))


def evaluate_choices(rows, picks, addresses, store, model, tokenizer, codebook, reader, values):
    episodes, plans, trace = replay(rows, picks, addresses)
    result = evaluate(episodes, plans, store.catalog(), store, model, tokenizer, codebook, reader, values=values)
    totals = Counter()
    direct = {}
    for (_, _, label, kind), predicted in zip(rows, picks, strict=True):
        direct.setdefault(kind, Counter()).update(total=1, correct=label == predicted,
                                                  false_cancel=predicted == CANCEL and label != CANCEL)
    for row, state in zip(result['rows'], trace, strict=True):
        row.update(state)
        cancellation = state['label'] == CANCEL
        cleared = not any(state['scope']) and not state['pending']
        nonempty = cancellation and any(state['opening'])
        totals.update(cancel_total=cancellation, cleared=cancellation and cleared,
                      nonempty_cancel=nonempty, cleared_nonempty=nonempty and cleared,
                      read_after_cancel=cancellation and row['selected'] is not None,
                      unclear_total=state['label'] == CLARIFY,
                      waiting_unclear=state['label'] == CLARIFY and state['pending'])
    result.update(totals=totals, decisions=direct)
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('archive', type=Path)
    cli.add_argument('prior', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--recurrent', type=Path, nargs=3, required=True)
    cli.add_argument('--reader', type=Path, default=Path('runs/spora-dual-version-reply-v1'))
    cli.add_argument('--normalizer', type=Path, default=Path('runs/spora-version-normalizer-v1'))
    cli.add_argument('--stored-action', action='store_true')
    cli.add_argument('--speaker-contrasts', action='store_true')
    args = cli.parse_args()
    if args.speaker_contrasts and not args.stored_action:
        cli.error('--speaker-contrasts requires --stored-action')
    main(args)
