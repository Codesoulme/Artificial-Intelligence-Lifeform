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


# Each question pair switches what an affirmative reply authorizes.
QUESTIONS = {
    'train': (('您是要停止这轮查阅吗？', '您是要继续这轮查阅吗？'),
              ('现在把整组查询撤销，对吗？', '现在把整组查询保留，对吗？'),
              ('您的决定是清空待查范围吗？', '您的决定是沿用待查范围吗？')),
    'calibration': (('您的意思是结束当前调阅吗？', '您的意思是维持当前调阅吗？'),
                    ('前面提的请求都要撤掉，是吗？', '前面提的请求都要接着办，是吗？')),
    'held': (('手头这批档案还要照查，您确认吗？', '手头这批档案就不再查了，您确认吗？'),
             ('刚定下的查询事项一并作罢吗？', '刚定下的查询事项照原样执行吗？')),
}
REPLIES = {
    'train': (('对，就照你说的。', '不，我的意思正好相反。', '先等等，我还得考虑。'),
              ('是，没错。', '不是，反过来。', '稍后再告诉你。'),
              ('我确认。', '我否认你说的那个决定。', '刚刚别人说过，算不算我还不知道。')),
    'calibration': (('嗯，正是。', '不对，请按相反的来。', '这件事还没有敲定。'),
                    ('没说错，我同意。', '恰恰不是你说的那样。', '请再等等。')),
    'held': (('是的。', '说反了。', '容我再想一想。'),
             ('就按你刚说的办。', '我选另一种。', '他说「同意」，我没有表态。')),
}
EXPLICIT = {
    'train': (('不管刚才怎么问，我现在决定停止查阅。', CANCEL),
              ('不用撤销，按手头范围继续。', PROCEED),
              ('整个这次我不再需要了。', CANCEL),
              ('这些记录还是接着调吧。', PROCEED)),
    'calibration': (('撤回我这次全部核查要求。', CANCEL), ('请依照原有名单核查。', PROCEED)),
    'held': (('我确认整批都不再调档。', CANCEL), ('一笔都不撤，接着查。', PROCEED),
             ('谁都不用查了，这就是我的决定。', CANCEL), ('把原来定的全留下，继续办理。', PROCEED)),
}


def cases(split):
    result = []
    for index, pair in enumerate(QUESTIONS[split]):
        affirmative = (PROCEED, CANCEL) if split == 'held' and index == 0 else (CANCEL, PROCEED)
        for question, agreement in zip(pair, affirmative, strict=True):
            opposite = CANCEL if agreement == PROCEED else PROCEED
            for replies in REPLIES[split]:
                result.extend((question, reply, label, kind) for (reply, label), kind in zip(
                    zip(replies, (agreement, opposite, CLARIFY), strict=True), ('yes', 'no', 'unclear'), strict=True))
            result.extend((question, reply, label, 'explicit') for reply, label in EXPLICIT[split])
    return result


def examples(rows, contextual):
    return [(f'孢子：{question}\n用户：{reply}' if contextual else reply, label)
            for question, reply, label, _ in rows]


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
    groups = {name: cases(name) for name in QUESTIONS}
    distinct_replies = [{reply for _, reply, _, _ in rows} for rows in groups.values()]
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
        for name, contextual, depth in (('baseline', False, 12), ('reply_only', False, 12),
                                        ('with_question', True, 12), ('recurrent', True, 23)):
            extract = shallow if depth == 12 else FrozenTurnFeatures(decoder, tokenizer, depth=depth, single_sequence=True)
            checkpoint = prior_head if name == 'baseline' else args.output / f'{name}.pt'
            if name == 'baseline':
                head = baseline.eval()
            else:
                head = fit_probe(extract, examples(groups['train'], contextual), 3, clauses=True)
                atomic_save(head.state_dict(), checkpoint)
                head.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'), strict=True)
            calibration = classify(head, extract, examples(groups['calibration'], contextual), clauses=True)
            cutoff = prior['results']['matched']['cutoff'] if name == 'baseline' else calibrate(calibration)
            held = classify(head, extract, examples(groups['held'], contextual), clauses=True)
            picks = [decision(row, cutoff) for row in held]
            episodes, plans, trace = replay(groups['held'], picks, addresses)
            result = evaluate(episodes, plans, store.catalog(), store, model, tokenizer, codebook, reader, values=values)
            totals = Counter()
            direct = {}
            for (_, _, label, kind), predicted in zip(groups['held'], picks, strict=True):
                direct.setdefault(kind, Counter()).update(total=1, correct=label == predicted,
                                                          false_cancel=predicted == CANCEL and label != CANCEL)
            for row, state in zip(result['rows'], trace, strict=True):
                row.update(state)
                totals.update(cancel_total=state['label'] == CANCEL,
                              cleared=state['label'] == CANCEL and not any(state['scope']) and not state['pending'],
                              read_after_cancel=state['label'] == CANCEL and row['selected'] is not None,
                              unclear_total=state['label'] == CLARIFY,
                              waiting_unclear=state['label'] == CLARIFY and state['pending'])
            result.update(cutoff=cutoff, totals=totals, decisions=direct, calibration=calibration, held=held,
                          picks=picks, checkpoint_sha256=base.sha(checkpoint))
            results[name] = result
            print(name, cutoff, direct, totals, result['summary'], flush=True)
    if any(base.sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('A model input or archive changed')
    store_json(args.output / 'result.json', dict(results=results, training=len(groups['train']),
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
    main(cli.parse_args())
