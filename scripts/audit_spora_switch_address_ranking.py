"""Audit saved address rankers on switched requests and catalog omissions."""

import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch

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
        results = {}
        for mode, cases, candidates in (('regression', regression_rows(), catalog),
                                        ('fresh', addressed_rows(ARCHIVE_FRESH), catalog),
                                        ('similar_names', addressed_rows(NEAR_FRESH), near),
                                        ('transfer', addressed_rows(TRANSFER), catalog)):
            results[mode] = audit(extract, old, old_cut, heads, cutoffs, cases, candidates)
            print(mode, {name: outcome['counts'] for name, outcome in results[mode].items()}, flush=True)
    if any(base.sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('An input changed during the audit')
    store_json(args.output / 'result.json', dict(results=results, training=training,
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
    main(cli.parse_args())
