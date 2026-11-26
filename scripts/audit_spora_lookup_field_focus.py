"""Compare whole-request addressing with learned clause selection on matched distractors."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from scripts import audit_spora_dense_joint_address as dense
from scripts import audit_spora_spoken_query_plan as plan
from scripts.audit_spora_switch_address_ranking import full_mask, split_outcomes, split_scores
from scripts.spora_version_chat import FrozenTurnFeatures, base, local_model
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_version_store import MemoryAddress


TRAIN = (
    ('请告诉我{cue}', '{cue}这项我自己去确认', '{owner}选来看的那一册', '{owner}让我拿过去的物件'),
    ('我想从记录里核实{cue}', '{cue}先搁下不用报', '{owner}预定阅读的作品', '{owner}请我捎去的那件'),
    ('请把{cue}的名字报给我', '我没打算翻{cue}的记录', '{owner}留下的读物', '{owner}让我准备的东西'),
    ('需要确认{cue}', '先暂时略过{cue}', '{owner}要借的读物', '{owner}嘱咐带上的用具'),
    ('我只想查{cue}', '{cue}下次再处理', '{owner}看中的那本书', '{owner}托人携带的那一项'),
    ('我要核对{cue}', '{cue}我会另行打听', '{owner}打算翻阅的那卷', '{owner}说要备好的器具'),
)
HELD = (
    ('帮我回顾一下{cue}', '{cue}就留到改天吧', '{owner}打算读的那部', '{owner}让我随身捎上的东西'),
    ('我得知道{cue}叫什么', '{cue}这次我会直接问本人', '{owner}书架上要找的那一本', '{owner}叮嘱我别忘拿的那样'),
    ('我想弄清{cue}', '先不要管{cue}', '{owner}先前圈中的那本读物', '{owner}出门之前要找的物品'),
)
DEFERRED = (
    ('{owner}打算读的那部就留到改天吧', '{owner}让我随身捎上的东西我会亲自问'),
    ('我会另外去确认{owner}书架上要找的那一本', '先别报{owner}叮嘱我拿的那样'),
    ('我暂时不用知道{owner}选中的读物', '至于{owner}要找的物品也先放着'),
    ('把「{owner}打算读哪部」这句话改礼貌些', '不要回答里面的问题'),
)
CALIBRATION = (
    ('告诉我档案里写定的{cue}', '这回先不必说明{cue}', '{owner}挑去阅览的那册', '{owner}交代带去的那件物品'),
    ('核实一下{cue}', '{cue}我准备自己问清楚', '{owner}借阅清单里选的作品', '{owner}途中得用的东西'),
)
TRANSFER = (
    ('报一下{cue}在档案里的名字', '这次不用翻出{cue}', '{owner}写下准备阅读的那本', '{owner}出门托我带的那件'),
    ('说说先前登记的{cue}', '{cue}先留着，等我下回再问', '{owner}托我从书架取的那卷', '{owner}请我随手捎去的用具'),
)


def paired_cases(owners, families):
    rows = []
    for owner in owners:
        for ask, defer, book, item in families:
            cues = (book.format(owner=owner), item.format(owner=owner))
            for reverse in (False, True):
                for target, field in enumerate(('books', 'items')):
                    parts = (ask.format(cue=cues[target]), defer.format(cue=cues[1 - target]))
                    rows.append((parts[::-1] if reverse else parts, int(reverse), MemoryAddress(owner, field)))
    return rows


def requests(rows):
    return [('，'.join(parts) + '。', address) for parts, _, address in rows]


def no_single_cases(owners, families):
    rows = []
    for owner in owners:
        for ask, defer, book, item in families:
            cues = (book.format(owner=owner), item.format(owner=owner))
            for frame in (defer, ask):
                parts = tuple(frame.format(cue=cue) for cue in cues)
                rows.append((parts, -1, None))
    return rows


def encode_clauses(extract, rows, owners):
    clauses = list(dict.fromkeys(text for parts, _, _ in rows for text in parts))
    states, _, _ = dense.align(extract, clauses, owners)
    vectors = dict(zip(clauses, states, strict=True))
    return torch.stack([torch.stack([vectors[part] for part in parts]) for parts, _, _ in rows])


def fit_focus(features, rows, *, independent=False):
    torch.manual_seed(dense.SEED)
    head = nn.Sequential(nn.LayerNorm(features.shape[-1]), nn.Linear(features.shape[-1], 32), nn.GELU(), nn.Linear(32, 1))
    targets = torch.tensor([index for _, index, _ in rows])
    if independent:
        targets = targets[:, None].eq(torch.arange(features.shape[1])).float()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    for _ in range(140):
        scores = head(features).squeeze(-1)
        loss = (F.binary_cross_entropy_with_logits(scores, targets) if independent
                else F.cross_entropy(scores, targets))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return head.eval()


def pairs_correct(outcomes):
    rows = outcomes['rows']
    return sum(left['unmasked'] == left['expected'] and right['unmasked'] == right['expected']
               for left, right in zip(rows[::2], rows[1::2], strict=True))


def single_clause(scores, threshold):
    allowed = scores >= threshold
    count = allowed.sum(1)
    return torch.where(count == 1, allowed.long().argmax(1), -1)


def calibration_cut(scores, rows):
    expected = torch.tensor([index for _, index, _ in rows])

    def utility(cut):
        selected = single_clause(scores, cut)
        correct = (selected.eq(expected) & expected.ge(0)).sum().item()
        incorrect = (selected.ne(expected) & selected.ge(0)).sum().item()
        return correct - 2 * incorrect, -incorrect, -cut

    return max((.5, .65, .8, .9, .95, .99, 1.01), key=utility)


@torch.inference_mode()
def single_evidence(extract, rows, focus, owners, cutoff):
    cases = requests(rows)
    full = dense.align(extract, [text for text, _ in cases], owners)
    masks = torch.stack([full_mask(names) for names in full[2]])
    absent = dense.subtract_targets(masks, cases, full[2])
    confidence = focus(encode_clauses(extract, rows, owners)).squeeze(-1).sigmoid()
    selected = single_clause(confidence, cutoff).tolist()
    # A placeholder is encoded for skipped requests but never used as a proposal.
    chosen = [parts[max(0, index)] for (parts, _, _), index in zip(rows, selected, strict=True)]
    local = dense.align(extract, chosen, owners)
    return cases, full, masks, absent, local, confidence, selected


def score_single(rows, evidence, pointer, intent, field_cut):
    cases, full, masks, absent, local, confidence, selected = evidence
    ranking = split_scores(pointer, intent, local)
    scores = [(dense.index_of(dense.addressed(slot, local_names), names), probability, float(index >= 0))
              for (slot, probability, _), local_names, names, index in zip(ranking, local[2], full[2], selected, strict=True)]
    outcome = split_outcomes(cases, full, masks, absent, scores, (.5, field_cut))
    targets = [index for _, index, _ in rows]
    outcome.update(intent_correct=sum(a == b for a, b in zip(selected, targets, strict=True)),
                   selected=selected, targets=targets, clause_confidence=confidence.tolist())
    return outcome


def evaluate_single(extract, rows, focus, pointer, intent, owners, cutoff, field_cut):
    evidence = single_evidence(extract, rows, focus, owners, cutoff)
    return score_single(rows, evidence, pointer, intent, field_cut)


@torch.inference_mode()
def compare(extract, rows, focus, pointer, intent, thresholds, owners, *, deferred=False):
    cases = requests(rows)
    full = dense.align(extract, [text for text, _ in cases], owners)
    masks = torch.stack([full_mask(names) for names in full[2]])
    absent = dense.subtract_targets(masks, cases, full[2])
    evidence = encode_clauses(extract, rows, owners)
    selection = focus(evidence).squeeze(-1).argmax(-1).tolist()
    oracle = [index for _, index, _ in rows]
    results = {}
    baseline_scores = split_scores(pointer, intent, full)
    modes = [('whole', None), ('learned', selection), ('local_intent', selection)]
    if not deferred:
        modes.append(('oracle', oracle))
    for mode, indexes in modes:
        scores = baseline_scores
        if indexes is not None:
            chosen = [parts[index] for (parts, _, _), index in zip(rows, indexes, strict=True)]
            local = dense.align(extract, chosen, owners)
            evidence_scores = split_scores(pointer, intent, local)
            scores = [(dense.index_of(dense.addressed(slot, local_names), full_names), confidence,
                       local_intent if mode == 'local_intent' else original[2])
                      for (slot, confidence, local_intent), local_names, full_names, original in
                      zip(evidence_scores, local[2], full[2], baseline_scores, strict=True)]
        outcome = split_outcomes(cases, full, masks, absent, scores, thresholds)
        if not deferred:
            outcome['correct_pairs'] = pairs_correct(outcome)
        results[mode] = outcome
    result = dict(results=results, choices=selection, total=len(rows))
    if not deferred:
        result.update(focus_correct=sum(a == b for a, b in zip(selection, oracle, strict=True)),
                      expected_choices=oracle)
    return result


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    manifest = json.loads((args.baseline / 'result.json').read_text(encoding='utf-8'))
    field = args.baseline / 'field.pt'
    intent_path = args.baseline / 'intent.pt'
    backbone = (args.recurrent[0], *(folder / 'attention.pt' for folder in args.recurrent[1:]))
    sources = {path: base.sha(path) for path in (field, intent_path, *backbone)}
    config = manifest['separate_training']
    expected = {Path(path).resolve(): digest for path, digest in manifest['sources'].items()}
    if (sources[field] != config['field_sha256'] or sources[intent_path] != config['intent_sha256']
            or any(expected.get(path.resolve()) != sources[path] for path in backbone)):
        raise ValueError('Saved heads and recurrent backbone do not match the baseline')
    train_owners = ('窦明川', '戚时月')
    held_owners = ('夏宁瑶', '郭回霜')
    train, held = paired_cases(train_owners, TRAIN), paired_cases(held_owners, HELD)
    normalized = [{plan.anonymize(text, names)[0] for text, _ in requests(rows)}
                  for rows, names in ((train, train_owners), (held, held_owners))]
    if normalized[0] & normalized[1]:
        raise ValueError('Training and held-out wording overlap')
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    with local_model(args.recurrent) as model:
        extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=plan.DEPTH)
        width = model.config.hidden_size
        pointer = dense.JointPointer(width).eval()
        intent = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 32), nn.GELU(), nn.Linear(32, 1)).eval()
        pointer.load_state_dict(torch.load(field, map_location='cpu', weights_only=True), strict=True)
        intent.load_state_dict(torch.load(intent_path, map_location='cpu', weights_only=True), strict=True)
        focus = fit_focus(encode_clauses(extract, train, train_owners), train)
        checkpoint = args.output / 'focus.pt'
        atomic_save(focus.state_dict(), checkpoint)
        focus.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        result = compare(extract, held, focus, pointer, intent, config['cutoffs'], held_owners)
        deferred = [(tuple(part.format(owner=owner) for part in parts), -1, None)
                    for owner in held_owners for parts in DEFERRED]
        controls = compare(extract, deferred, focus, pointer, intent, config['cutoffs'], held_owners, deferred=True)
        cal_owners = ('庄临修', '乔文荃')
        calibration = [*paired_cases(cal_owners, CALIBRATION), *no_single_cases(cal_owners, CALIBRATION)]
        calibration_vectors = encode_clauses(extract, calibration, cal_owners)
        independent = fit_focus(encode_clauses(extract, train, train_owners), train, independent=True)
        independent_path = args.output / 'independent.pt'
        atomic_save(independent.state_dict(), independent_path)
        independent.load_state_dict(torch.load(independent_path, map_location='cpu', weights_only=True), strict=True)
        with torch.inference_mode():
            cutoff = calibration_cut(independent(calibration_vectors).squeeze(-1).sigmoid(), calibration)
        transfer = [*paired_cases(held_owners, TRANSFER), *no_single_cases(held_owners, TRANSFER)]
        independent_results = {name: evaluate_single(extract, cases, independent, pointer, intent, owners,
                                                       cutoff, config['cutoffs'][1])
                               for name, cases, owners in (('calibration', calibration, cal_owners),
                                                           ('held', held, held_owners),
                                                           ('deferred', deferred, held_owners),
                                                           ('transfer', transfer, held_owners))}
    if any(base.sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('A protected model input changed')
    print('focus', result['focus_correct'], '/', result['total'], flush=True)
    for mode, outcome in result['results'].items():
        print(mode, outcome['correct_pairs'], outcome['counts'], flush=True)
    for mode, outcome in controls['results'].items():
        print('deferred', mode, outcome['counts'], flush=True)
    for mode, outcome in independent_results.items():
        print('independent', cutoff, mode, outcome['intent_correct'], outcome['counts'], flush=True)
    store_json(args.output / 'result.json', dict(**result, deferred=controls['results'], training=len(train),
                                               independent=independent_results, independent_cutoff=cutoff,
                                               independent_sha256=base.sha(independent_path),
                                               seconds=perf_counter() - started,
                                               checkpoint_sha256=base.sha(checkpoint),
                                               sources={str(path): digest for path, digest in sources.items()}))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('baseline', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--recurrent', type=Path, nargs=3, required=True)
    main(cli.parse_args())
