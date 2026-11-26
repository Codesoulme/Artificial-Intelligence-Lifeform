"""Audit saved address rankers on switched requests and catalog omissions."""

import argparse
from collections import Counter
from itertools import islice, product
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from scripts import audit_spora_dense_joint_address as dense
from scripts import audit_spora_spoken_query_plan as plan
from scripts.audit_spora_spoken_switch_cli import SCENARIOS
from scripts.spora_version_chat import (FIELD_IDS, SWITCH_LOOKUP, FrozenTurnFeatures, VersionStore, artifacts,
                                        base, local_model)
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.research_version_dialogue_scope import eligible_spoken_lookup
from spora.research_version_store import MemoryAddress
from spora.language_training import atomic_save


ARCHIVE_FRESH = (
    ('跟唐砚平见面，之前说定让我到哪儿等？', '唐砚平', 'places'),
    ('叶澄依交代出门顺手捎上什么来着？', '叶澄依', 'items'),
    ('顾遥青托我买来喝的是哪一款？', '顾遥青', 'drinks'),
    ('岑蕙安原来相中的是哪部读物？', '岑蕙安', 'books'),
    ('岑蕙安那册先放一边，顾遥青让我找的是哪本？', '顾遥青', 'books'),
    ('碰头这事不问唐砚平了，叶澄依要我顺带拿什么？', '叶澄依', 'items'),
    ('唐砚平原先说要喝哪一款？', None, None),  # This person has no beverage record.
    ('唐砚坪和我说好在哪儿会合？', None, None),  # The near name is absent.
    ('唐砚平的去处或岑蕙安那本，先等我决定。', None, None),
    ('把「叶澄依让我带什么」改成客气的问法。', None, None),
)

NEAR_FRESH = (
    ('夏宁瑶那杯我不负责，夏宁遥说过让我买哪种喝的？', '夏宁遥', 'drinks'),
    ('夏宁遥的取书时间另约，我问的是夏宁瑶挑中的读物。', '夏宁瑶', 'books'),
    ('要去见郭回霜，她说的集合位置在哪？郭回双那边已经问到了。', '郭回霜', 'places'),
    ('跟郭回霜无关，郭回双请我带过去的是哪样？', '郭回双', 'items'),
    ('夏宁遥还是夏宁瑶的都行？先让我确认到底是谁。', None, None),
    ('请翻译「郭回霜让我带哪样」。', None, None),
    ('夏宁摇跟我约的会合地点在哪儿？', None, None),
)

TRANSFER = (
    ('我和唐砚平碰面该往哪儿去来着？', '唐砚平', 'places'),
    ('叶澄依叫我出门时帮忙带上啥？', '叶澄依', 'items'),
    ('要给顾遥青买喝的，册子里说的是哪一种？', '顾遥青', 'drinks'),
    ('岑蕙安那件事过会儿再谈，顾遥青上次挑的那本是哪部？', '顾遥青', 'books'),
    ('帮我核实叶澄依准备读的那一本。', None, None),
    ('用更客气的语气改写「唐砚平和我约在哪儿」这句。', None, None),
    ('岑蕙安还是顾遥青？我还没选好查谁那本。', None, None),
    ('唐砚坪提过约在哪碰面？', None, None),
)

INTENT_NEED = {
    '唐砚平原先说要喝哪一款？', '唐砚坪和我说好在哪儿会合？',
    '夏宁摇跟我约的会合地点在哪儿？', '帮我核实叶澄依准备读的那一本。',
    '唐砚坪提过约在哪碰面？',
}

# Each family asks about four fields while keeping the speaker and sentence frame fixed.
MATCHED_TRAIN = (
    ('我替{owner}准备，想核实{cue}。', ('杯里该装哪一种', '会合时该到哪个去处', '得带去的物件是哪件', '选来阅读的书是哪册')),
    ('上回替{owner}记下的，{cue}？', ('说要点来喝的品种', '约定等人的位置', '托我携带的那样物品', '打算翻阅的那一卷')),
    ('我想重看{owner}的安排，具体是{cue}。', ('饮用的那款叫什么', '碰面的地址是哪里', '要我捎上的器具是什么', '指定的阅读材料是哪部')),
    ('请帮我确认{owner}{cue}。', ('喝东西挑哪样', '在哪里等对方过来', '出门前要备好什么物件', '准备看哪本作品')),
    ('{owner}那一项请报一下，{cue}？', ('说好端来哪种饮料', '到什么地方集合', '出发时手上要拿哪样', '事先定下的读物叫什么')),
)
MATCHED_HELD = (
    ('关于{owner}，先前定好的{cue}来着？', ('要啜饮的是哪款', '碰头应该走到哪处', '让我捎过去的是什么', '拿来读的是什么书')),
    ('我按{owner}原来的话办，{cue}？', ('得给倒哪种喝的', '要去什么位置等', '手边要准备哪一个东西', '书架上该取走哪一本')),
    ('向你确认{owner}先前那笔，{cue}？', ('端上桌给喝的是哪一样', '两个人在哪会面', '对方嘱咐随身带什么', '定的那部阅读作品是什么')),
)


def matched_cases(names, families):
    return [(frame.format(owner=name, cue=cue), MemoryAddress(name, field))
            for name, (frame, cues) in product(names, families)
            for field, cue in zip(plan.FIELDS_ORDER, cues, strict=True)]

ADDRESS_FORMS = (
    ('{owner}{field}？', True),
    ('我这会想核实{owner}{field}。', True),
    ('{other}先不用管，我现在问{owner}{field}。', True),
    ('先别查{owner}{field}，我会自己问。', False),
    ('请把「{owner}{field}」改写得简短点。', False),
    ('我只是提到{owner}{field}，不用调档。', False),
)
CAL_FORMS = (
    ('{owner}{field}这事我得核对一下。', True),
    ('我想问{owner}{field}，跟{other}无关。', True),
    ('{owner}{field}我稍后亲自确认，你先别报。', False),
    ('是{owner}还是{other}？我还没决定要查谁。', False),
)


def training_cases(names, aliases, forms):
    rows = []
    for order, (field, phrases), (frame, reads) in product(range(2), zip(plan.FIELDS_ORDER, aliases, strict=True), forms):
        owner, other = names[::1 if order == 0 else -1]
        address = MemoryAddress(owner, field) if reads else None
        rows.extend((frame.format(owner=owner, other=other, field=phrase), address) for phrase in phrases)
    return list(dict.fromkeys(rows))


def addressed_rows(rows):
    return [(text, None if owner is None else MemoryAddress(owner, field)) for text, owner, field in rows]


def regression_rows():
    result = []
    for _, question, label, _, _ in SCENARIOS:
        if label:
            owner, field = label.split('的', 1)
            result.append((SWITCH_LOOKUP.fullmatch(question)['target'], MemoryAddress(owner, FIELD_IDS[field])))
    return result


def evidence(extract, cases, catalog):
    questions = [text for text, _ in cases]
    enrolled = {address.name for address in catalog}
    inputs = dense.align(extract, questions, enrolled)
    masks = torch.stack([dense.mask_for(names, catalog) for names in inputs[2]])
    absent = dense.subtract_targets(masks, cases, inputs[2])
    return inputs, masks, absent


def choice(prediction, names, threshold):
    slot, confidence = prediction
    return dense.addressed(slot, names) if confidence >= threshold else None


def evaluate(cases, inputs, predictions, removed, threshold):
    counts = Counter()
    rows = []
    for (question, target), names, raw, missing in zip(cases, inputs[2], predictions, removed, strict=True):
        allowed = eligible_spoken_lookup(question)
        proposed = choice(raw, names, threshold) if allowed else None
        substitute = choice(missing, names, threshold) if allowed else None
        counts.update(positive=target is not None, negative=target is None, correct=target is not None and proposed == target,
                      wrong=target is not None and proposed is not None and proposed != target,
                      unnecessary=target is None and proposed is not None,
                      substitute=target is not None and substitute is not None)
        rows.append(dict(question=question, expected=None if target is None else [target.name, target.field],
                         candidate=None if proposed is None else [proposed.name, proposed.field],
                         raw=raw, removed=missing,
                         after_removal=None if substitute is None else [substitute.name, substitute.field]))
    return dict(counts=counts, rows=rows)


def audit(extract, old, old_cut, heads, thresholds, cases, catalog):
    inputs, masks, absent = evidence(extract, cases, catalog)
    results = {}
    for name, head in heads.items():
        options = dict(local=name != 'global')
        results[name] = evaluate(cases, inputs, dense.proposals(head, inputs, masks, **options),
                                 dense.proposals(head, inputs, absent, **options), thresholds[name])
    old_features, names = plan.vectors(extract, [text for text, _ in cases], {address.name for address in catalog}, 'user-span')
    if names != inputs[2]:
        raise ValueError('The two address encoders disagreed on mentioned names')
    addresses = [plan.proposed_address(prediction, owners, old_cut)
                 for prediction, owners in zip(plan.predict(old, old_features), names, strict=True)]
    old_slots = [dense.index_of(address, owners) for address, owners in zip(addresses, names, strict=True)]
    present = [(slot if slot < dense.NONE and mask[slot] else dense.NONE, 1.)
               for slot, mask in zip(old_slots, masks, strict=True)]
    dropped = [(slot if slot < dense.NONE and mask[slot] else dense.NONE, 1.)
               for slot, mask in zip(old_slots, absent, strict=True)]
    results['old'] = evaluate(cases, inputs, present, dropped, 0.)
    return results


def train_local(extract):
    names, cal_names = ('马澄文', '章其安'), ('陶知书', '周越溪')
    train = training_cases(names, dense.ALIASES, ADDRESS_FORMS)
    cal = training_cases(cal_names, dense.CAL_ALIASES, CAL_FORMS)
    state, spans, owners = dense.align(extract, [text for text, _ in train], names)
    head = dense.fit(state, spans, dense.supervised(train, owners, len(train)), local=True)
    catalog = {MemoryAddress(owner, field) for owner, field in product(cal_names, plan.FIELDS_ORDER)}
    inputs, masks, absent = evidence(extract, cal, catalog)
    present = dense.proposals(head, inputs, masks, local=True)
    missing = dense.proposals(head, inputs, absent, local=True)
    cutoff = dense.calibrate(cal, present, missing, inputs[2])
    return head, cutoff, dict(training=len(train), calibration=len(cal),
                              outcomes=evaluate(cal, inputs, present, missing, cutoff))


def full_mask(names):
    addresses = {MemoryAddress(owner, field) for owner, field in product(names, plan.FIELDS_ORDER)}
    return dense.mask_for(names, addresses)


def fit_split(train, inputs, *, fixed_intent=None, steps=None, pointer_factory=dense.JointPointer):
    states, spans, names = inputs
    torch.manual_seed(dense.SEED)
    pointer = pointer_factory(states.shape[-1])
    intent = nn.Sequential(nn.LayerNorm(states.shape[-1]), nn.Linear(states.shape[-1], 32),
                           nn.GELU(), nn.Linear(32, 1))
    if fixed_intent is not None:
        intent.load_state_dict(fixed_intent)
        intent.requires_grad_(False)
    for parameter in pointer.none.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([*pointer.parameters(), *intent.parameters()], lr=.002)
    targets = torch.tensor([dense.index_of(target, owners) for (_, target), owners in zip(train, names, strict=True)])
    reads = targets.ne(dense.NONE)
    masks = torch.stack([full_mask(owners) for owners in names])
    rng = torch.Generator().manual_seed(dense.SEED)
    batches = (batch for _ in range(16) for batch in torch.randperm(len(train), generator=rng).split(32))
    if steps is not None:
        batches = islice(batches, steps)
    for batch in batches:
        positive = batch[reads[batch]]
        logits = pointer(states[positive], spans[positive], masks[positive])[:, :dense.NONE]
        loss = F.cross_entropy(logits, targets[positive], reduction='sum') / max(1, len(positive))
        if fixed_intent is None:
            loss += F.binary_cross_entropy_with_logits(intent(states[batch]).squeeze(-1), reads[batch].float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return pointer.eval(), intent.eval()


@torch.inference_mode()
def split_scores(pointer, intent, inputs):
    states, spans, names = inputs
    masks = torch.stack([full_mask(owners) for owners in names])
    # Keep an explicit empty option only on rows without any named candidate.
    logits = pointer(states, spans, masks)
    logits[:, dense.NONE] = torch.where(masks.any(1), -torch.inf, 0.)
    confidence, slots = logits.softmax(-1).max(-1)
    return list(zip(slots.tolist(), confidence.tolist(), intent(states).squeeze(-1).sigmoid().tolist(), strict=True))


def apply_split(scores, masks, cuts):
    intent_cut, field_cut = cuts
    return [(slot if intent >= intent_cut and confidence >= field_cut and slot < dense.NONE and mask[slot]
             else dense.NONE, 1.) for (slot, confidence, intent), mask in zip(scores, masks, strict=True)]


def split_outcomes(cases, inputs, masks, absent, scores, cuts):
    base = evaluate(cases, inputs, apply_split(scores, masks, cuts), apply_split(scores, absent, cuts), 0.)
    counts = base['counts']
    for row, (text, target), names, (slot, confidence, intent) in zip(base['rows'], cases, inputs[2], scores, strict=True):
        address = dense.addressed(slot, names)
        expected_intent = target is not None or text in INTENT_NEED
        row.update(unmasked=None if address is None else [address.name, address.field], field_confidence=confidence,
                   intent_confidence=intent, expected_intent=expected_intent)
        counts.update(field_correct=target is not None and address is not None and address.field == target.field,
                      owner_correct=target is not None and address is not None and address.name == target.name,
                      unmasked_correct=target is not None and address == target,
                      intent_positive=expected_intent, intent_negative=not expected_intent,
                      intent_hit=expected_intent and intent >= cuts[0], intent_false=not expected_intent and intent >= cuts[0])
    return base


def matched_family_totals(rows):
    width = len(plan.FIELDS_ORDER)
    if len(rows) % width:
        raise ValueError('A matched family must contain all fields')
    families = [rows[start:start + width] for start in range(0, len(rows), width)]
    return dict(total=len(families), exact=sum(all(row['unmasked'] == row['expected'] for row in family)
                                               for family in families))


def train_split(extract):
    names, cal_names = ('马澄文', '章其安'), ('陶知书', '周越溪')
    train = training_cases(names, dense.ALIASES, ADDRESS_FORMS)
    cal = training_cases(cal_names, dense.CAL_ALIASES, CAL_FORMS)
    pointer, intent = fit_split(train, dense.align(extract, [text for text, _ in train], names))
    catalog = {MemoryAddress(owner, field) for owner, field in product(cal_names, plan.FIELDS_ORDER)}
    inputs, masks, absent = evidence(extract, cal, catalog)
    scores = split_scores(pointer, intent, inputs)

    def utility(cuts):
        outcome = split_outcomes(cal, inputs, masks, absent, scores, cuts)['counts']
        bad = outcome['wrong'] + outcome['unnecessary'] + outcome['substitute']
        return outcome['correct'] - 2 * bad, -bad, -sum(cuts)

    options = (.5, .65, .8, .9, .95, .99, 1.01)
    cuts = max(product(options, repeat=2), key=utility)
    return (pointer, intent), cuts, dict(training=len(train), calibration=len(cal), cutoffs=cuts,
                                        outcomes=split_outcomes(cal, inputs, masks, absent, scores, cuts))


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    archive = args.archive / 'three-versions.sqlite3'
    archive_report = json.loads((args.archive / 'result.json').read_text(encoding='utf-8'))
    old_path = args.planner / 'public.pt'
    old_report = json.loads((args.planner / 'result.json').read_text(encoding='utf-8'))
    ranker_report = json.loads((args.ranker / 'result.json').read_text(encoding='utf-8'))
    head_paths = {name: args.ranker / f'{name}.pt' for name in ('global', 'local')}
    backbone = (args.recurrent[0], *(folder / 'attention.pt' for folder in args.recurrent[1:]))
    sources = {path: base.sha(path) for path in (archive, old_path, *head_paths.values(), *backbone)}
    if (sources[archive] != archive_report['archive_sha256']
            or sources[old_path] != old_report['results']['public']['probe_sha256']
            or any(sources[path] != ranker_report['sha256'][name] for name, path in head_paths.items())):
        raise ValueError('An input differs from its experiment report')
    for report in (archive_report, old_report, ranker_report):
        indexed = {Path(path).resolve(): digest for path, digest in report['sources'].items()}
        if any(indexed.get(path.resolve()) != sources[path] for path in backbone):
            raise ValueError('Model weights do not match the saved rankers')
    args.output.mkdir(parents=True, exist_ok=False)
    _, _, _, binding = artifacts(Path('runs/spora-dual-version-reply-v1'), Path('runs/spora-version-normalizer-v1'))
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    with local_model(args.recurrent) as model, VersionStore(archive, binding) as store:
        extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=plan.DEPTH)
        old = plan.QueryPlanProbe(model.config.hidden_size).eval()
        old.load_state_dict(torch.load(old_path, map_location='cpu', weights_only=True))
        old_cut = old_report.get('address_proposal', {}).get('threshold', old_report['results']['public']['threshold'])
        heads = {name: dense.JointPointer(model.config.hidden_size).eval() for name in head_paths}
        for name, head in heads.items():
            head.load_state_dict(torch.load(head_paths[name], map_location='cpu', weights_only=True))
        catalog = set(store.catalog())
        near = {MemoryAddress(owner, field) for owner, field, _ in dense.RECORDS}
        cutoffs = dict(ranker_report['thresholds'])
        training = None
        if args.retrain:
            heads['retrained'], cutoffs['retrained'], training = train_local(extract)
            destination = args.output / 'retrained.pt'
            atomic_save(heads['retrained'].state_dict(), destination)
            heads['retrained'].load_state_dict(torch.load(destination, map_location='cpu', weights_only=True), strict=True)
            training.update(checkpoint_sha256=base.sha(destination), cutoff=cutoffs['retrained'])
        split_training = None
        if args.separate:
            (pointer, intent), split_cuts, split_training = train_split(extract)
            for name, head in (('field', pointer), ('intent', intent)):
                destination = args.output / f'{name}.pt'
                atomic_save(head.state_dict(), destination)
                head.load_state_dict(torch.load(destination, map_location='cpu', weights_only=True), strict=True)
                split_training[f'{name}_sha256'] = base.sha(destination)
        matched_training = None
        suites = [('regression', regression_rows(), catalog), ('fresh', addressed_rows(ARCHIVE_FRESH), catalog),
                  ('similar_names', addressed_rows(NEAR_FRESH), near), ('transfer', addressed_rows(TRANSFER), catalog)]
        if args.matched:
            train_names = ('马澄文', '章其安')
            held_names = ('郭回霜', '夏宁瑶')
            original = training_cases(train_names, dense.ALIASES, ADDRESS_FORMS)
            extra = matched_cases(train_names, MATCHED_TRAIN)
            held = matched_cases(held_names, MATCHED_HELD)
            train = [*original, *extra]
            normalized = [{plan.anonymize(text, names)[0] for text, _ in rows}
                          for rows, names in ((train, train_names), (held, held_names))]
            if normalized[0] & normalized[1]:
                raise ValueError('Matched field training and test prompts overlap')
            steps = 16 * ((len(original) + 31) // 32)
            matched, _ = fit_split(train, dense.align(extract, [text for text, _ in train], train_names),
                                   fixed_intent=intent.state_dict(), steps=steps)
            destination = args.output / 'matched-field.pt'
            atomic_save(matched.state_dict(), destination)
            matched.load_state_dict(torch.load(destination, map_location='cpu', weights_only=True), strict=True)
            matched_training = dict(training=len(train), added=len(extra), steps=steps, checkpoint_sha256=base.sha(destination))
            suites.append(('matched_fields', held, near))
        results = {}
        for mode, cases, candidates in suites:
            results[mode] = audit(extract, old, old_cut, heads, cutoffs, cases, candidates)
            if args.separate:
                inputs, masks, absent = evidence(extract, cases, candidates)
                results[mode]['separate'] = split_outcomes(cases, inputs, masks, absent,
                                                          split_scores(pointer, intent, inputs), split_cuts)
                if args.matched:
                    results[mode]['matched'] = split_outcomes(cases, inputs, masks, absent,
                                                             split_scores(matched, intent, inputs), split_cuts)
                    if mode == 'matched_fields':
                        for arm in ('separate', 'matched'):
                            outcome = results[mode][arm]
                            outcome['families'] = matched_family_totals(outcome['rows'])
            print(mode, {name: outcome['counts'] for name, outcome in results[mode].items()}, flush=True)
    if any(base.sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('An input changed during the audit')
    store_json(args.output / 'result.json', dict(results=results, training=training, separate_training=split_training,
                                               matched_training=matched_training,
                                               sources={str(path): digest for path, digest in sources.items()},
                                               seconds=perf_counter() - started))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('archive', type=Path)
    cli.add_argument('planner', type=Path)
    cli.add_argument('ranker', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--recurrent', type=Path, nargs=3, required=True)
    cli.add_argument('--retrain', action='store_true')
    cli.add_argument('--separate', action='store_true')
    cli.add_argument('--matched', action='store_true')
    args = cli.parse_args()
    if args.matched and not args.separate:
        cli.error('--matched requires --separate')
    main(args)
