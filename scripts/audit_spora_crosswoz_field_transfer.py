"""Test natural dialogue supervision for SPORA's joint memory pointer."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
from time import perf_counter
from types import SimpleNamespace
from zipfile import ZipFile

import torch
from torch import nn
from torch.nn import functional as F

from scripts import audit_spora_dense_joint_address as dense
from scripts import audit_spora_spoken_query_plan as plan
from scripts.audit_spora_crosswoz_positive_fields import PINNED, requested
from scripts.audit_spora_natural_lookup_cli import digest
from scripts.audit_spora_spoken_query_data import corpus_sample
from scripts.spora_version_chat import FrozenTurnFeatures, VersionReplyActions, VersionStore, artifacts, base, local_model
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_version_store import MemoryAddress


SEED = 882901
EPOCHS = 16
QUOTAS = {'train': (480, 480, 192), 'val': (120, 120, 48), 'test': (120, 120, 48)}
FRESH = (
    ('夏宁遥那杯先别管。我要给夏宁瑶带杯喝的，她点的是哪个？', 0, 'drinks'),
    ('夏宁瑶带的东西问错了。我是想问跟夏宁遥见面要去哪儿。', 1, 'places'),
    ('夏宁遥选的书我会另外打听。夏宁瑶让捎过去的那件东西叫什么？', 0, 'items'),
    ('本来想核对夏宁瑶的书，后来改要夏宁遥那本，麻烦报一下书名。', 1, 'books'),
    ('郭回双的约见先不问。到店要给郭回霜买哪杯喝的？', 2, 'drinks'),
    ('郭回霜先不见了，郭回双和我说好去哪儿碰头来着？', 3, 'places'),
    ('刚才说郭回双口误，帮我记起郭回霜让我带走的那件物品。', 2, 'items'),
    ('我知道郭回霜会喝什么。你只要告诉我郭回双挑的那册书是哪本。', 3, 'books'),
    ('夏宁瑶和夏宁遥的安排我自己问。先帮我想想晚饭吃啥。', None, None),
    ('郭回霜和郭回双要拿的东西先搁着，我问的是怎么收纳。', None, None),
    ('夏宁瑶的书我已问清楚了，我只想听你推荐点新的。', None, None),
    ('郭回双在哪儿见面这件事已经临时取消。郭回霜那边也不用再问。', None, None),
)


def fingerprint(text):
    return sha256(text.encode('utf-8')).hexdigest()


def context(messages, index):
    text = messages[index]['content'].strip()
    if not 2 <= len(text) <= 160:
        return None
    previous = messages[index - 1]['content'].strip()[-120:] if index and messages[index - 1]['role'] == 'sys' else ''
    return f'上一条回复：{previous}\n用户：{text}' if previous else f'用户：{text}'


def public_rows(archive, manifest, split, excluded):
    with ZipFile(archive) as archive_file:
        dialogues = json.loads(archive_file.read(f'{split}.json'))
    source = json.loads(manifest.read_text(encoding='utf-8'))
    if source['source_archive_sha256'] != PINNED[split]:
        raise ValueError(f'{split} manifest refers to a different corpus')
    buckets = [[], [], []]
    for item in source['rows']:
        identifier, index = item['dialogue'], item['turn']
        messages = dialogues[identifier]['messages']
        gold = requested(messages[index])
        if (messages[index]['role'] != 'usr' or [list(pair) for pair in gold] != item['requested']
                or fingerprint(messages[index]['content']) != item['utterance_sha256']):
            raise ValueError('A manifest item no longer matches its original annotation')
        prompt = context(messages, index)
        if prompt:
            bucket = 0 if 'multiple_fields' in item['flags'] else 1
            buckets[bucket].append((identifier, index, prompt, tuple(f'{domain}/{field}' for domain, field in gold)))
    for identifier, record in dialogues.items():
        messages = record['messages']
        for index, turn in enumerate(messages):
            if turn['role'] == 'usr' and not requested(turn) and (prompt := context(messages, index)):
                buckets[2].append((identifier, index, prompt, ()))
    result = []
    for bucket, quota in zip(buckets, QUOTAS[split], strict=True):
        bucket.sort(key=lambda row: fingerprint(f'{SEED}/{split}/{row[0]}/{row[1]}'))
        selected = 0
        for row in bucket:
            key = fingerprint(row[2])
            if key in excluded:
                continue
            result.append(row)
            excluded.add(key)
            selected += 1
            if selected == quota:
                break
        if selected != quota:
            raise ValueError(f'Could not fill the {split} sampler budget')
    return result


def multilabel(cases, labels):
    indices = {label: i for i, label in enumerate(labels)}
    target = torch.zeros(len(cases), len(labels))
    for i, (_, _, _, fields) in enumerate(cases):
        for name in fields:
            if name not in indices:
                raise ValueError(f'Tag has no label in the frozen vocabulary: {name}')
            target[i, indices[name]] = 1.
    return target


def fit(person, supervision, public, public_labels, *, mode):
    torch.manual_seed(dense.SEED)
    person_rng = torch.Generator().manual_seed(dense.SEED)
    public_rng = torch.Generator().manual_seed(SEED)
    states, spans, _ = person
    indices, masks, targets = supervision
    pointer = dense.JointPointer(states.shape[-1])
    aux = nn.Linear(pointer.output.in_features, public_labels.shape[-1])
    opt = torch.optim.AdamW([*pointer.parameters(), *aux.parameters()], lr=.002)
    frequency = public_labels.sum(0)
    weights = ((len(public_labels) - frequency) / frequency.clamp_min(1)).sqrt().clamp(1, 12)
    if mode == 'shuffled':
        order = list(range(len(public_labels)))
        random.Random(SEED).shuffle(order)
        public_labels = public_labels[order]
    steps = 0
    for _ in range(EPOCHS):
        for batch in torch.randperm(len(targets), generator=person_rng).split(32):
            rows = indices[batch]
            loss = F.cross_entropy(pointer(states[rows], spans[rows], masks[batch]), targets[batch])
            support = torch.randint(len(public), (len(batch),), generator=public_rng)
            if mode != 'personal':
                scores = aux(pointer.shared(public[support]))
                loss = loss + F.binary_cross_entropy_with_logits(scores, public_labels[support], pos_weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
    return pointer.eval(), aux.eval(), steps


def public_score(probs, gold, cutoff):
    guesses = probs >= cutoff
    gold = gold.bool()
    tp = int((guesses & gold).sum())
    fp = int((guesses & ~gold).sum())
    fn = int((~guesses & gold).sum())
    no_request = ~gold.any(-1)
    return {'micro_f1': 2 * tp / max(2 * tp + fp + fn, 1), 'exact': int((guesses == gold).all(-1).sum()),
            'multi_exact': int(((guesses == gold).all(-1) & (gold.sum(-1) > 1)).sum()),
            'none_correct': int((no_request & ~guesses.any(-1)).sum()), 'tp': tp, 'fp': fp, 'fn': fn}


@torch.inference_mode()
def public_measure(pointer, aux, validation, test, targets):
    val = aux(pointer.shared(validation)).sigmoid()
    thresholds = (.15, .25, .35, .45, .5, .6, .7, .8)
    cutoff = max(thresholds, key=lambda value: (public_score(val, targets['val'], value)['micro_f1'], value))
    scores = aux(pointer.shared(test)).sigmoid()
    return {'threshold': cutoff, 'validation': public_score(val, targets['val'], cutoff),
            'test': public_score(scores, targets['test'], cutoff)}


def make_cases(rows):
    return [(text, None if owner is None else MemoryAddress(dense.NAMES[owner], field)) for text, owner, field in rows]


def masks_for(inputs, addresses):
    return torch.stack([dense.mask_for(names, addresses) for names in inputs[2]])


def main(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    manifest_path = args.manifest / 'result.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    labels = tuple(manifest['fields']['train'])
    selected = {}
    bound = {Path(__file__): digest(Path(__file__)), manifest_path: digest(manifest_path)}
    excluded = set()
    for split, sha in PINNED.items():
        pack = args.corpus / f'{split}.json.zip'
        rows = args.manifest / f'{split}.json'
        bound[pack], bound[rows] = digest(pack), digest(rows)
        if bound[pack] != sha or bound[rows] != manifest['splits'][split]['manifest_sha256']:
            raise ValueError(f'{split} source does not match the audited corpus split')
        selected[split] = public_rows(pack, rows, split, excluded)
    targets = {split: multilabel(rows, labels) for split, rows in selected.items()}
    database = args.profile / 'profile' / 'versions.sqlite3'
    profile_report = json.loads((args.profile / 'result.json').read_text(encoding='utf-8'))
    prior_sources = {Path(path).resolve(): value for path, value in profile_report['sources'].items()}
    for path in (database, args.recurrent[0], *(folder / 'attention.pt' for folder in args.recurrent[1:])):
        bound[path] = digest(path)
        if bound[path] != prior_sources[path.resolve()]:
            raise ValueError('The dense profile or recurrent backbone changed')
    reader_file, _, _, binding = artifacts(args.reader, args.normalizer)
    bound[reader_file] = digest(reader_file)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    train_names, cal_names = ('程理新', '贺梦川'), ('周靖岭', '颜曦音')
    personal = dense.practice(dense.TRAIN, dense.ALIASES, train_names)
    calibration = dense.practice(dense.CAL, dense.CAL_ALIASES, cal_names)
    external = corpus_sample(args.ordinary, 'train', set())
    personal_all = (*personal, *((text, None) for _, _, _, text in external))
    held = {'previous': make_cases(dense.HELD), 'fresh': make_cases(FRESH)}
    prepared = {plan.anonymize(text, train_names)[0] for text, _ in personal_all}
    seen = {plan.anonymize(text, dense.NAMES)[0] for cases in held.values() for text, _ in cases}
    if prepared & seen:
        raise ValueError('Personal holdout has a training duplicate')
    output, trained = {}, {}
    with local_model(args.recurrent) as model, VersionStore(database, binding) as store:
        extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=plan.DEPTH)
        public = {}
        for split, rows in selected.items():
            public[split] = plan.vectors(extract, [text for _, _, text, _ in rows], (), 'user-span')[0]
            print('public features', split, len(rows), flush=True)
        inputs = dense.align(extract, [text for text, _ in personal_all], train_names)
        supervision = dense.supervised(personal_all, inputs[2], len(personal))
        cal = dense.align(extract, [text for text, _ in calibration], cal_names)
        cal_catalog = {MemoryAddress(name, field) for name in cal_names for field in plan.FIELDS_ORDER}
        cal_masks = masks_for(cal, cal_catalog)
        cal_missing = dense.subtract_targets(cal_masks, calibration, cal[2])
        held_inputs = {name: dense.align(extract, [text for text, _ in cases], dense.NAMES) for name, cases in held.items()}
        held_masks = {name: masks_for(features, store.catalog()) for name, features in held_inputs.items()}
        state = torch.load(reader_file, map_location='cpu', weights_only=True)
        reader = VersionReplyActions(model.config.hidden_size, state['scaffold_tokens'], slots=2)
        reader.load_state_dict(state)
        reader.eval().to(model.device)
        for mode in ('personal', 'shuffled', 'supervised'):
            pointer, aux, steps = fit(inputs, supervision, public['train'], targets['train'], mode=mode)
            trained[mode] = (pointer, aux)
            cutoff = dense.calibrate(calibration, dense.proposals(pointer, cal, cal_masks, local=True),
                                     dense.proposals(pointer, cal, cal_missing, local=True), cal[2])
            public_result = (public_measure(pointer, aux, public['val'], public['test'], targets)
                             if mode != 'personal' else None)
            private_result = {}
            for name, cases in held.items():
                features, masks = held_inputs[name], held_masks[name]
                visible = dense.proposals(pointer, features, masks, local=True)
                removed = dense.subtract_targets(masks, cases, features[2])
                missing = dense.proposals(pointer, features, removed, local=True)
                rows, metrics = dense.evaluate(store, model, tokenizer, reader, cases, features[2],
                                               {mode: visible}, {mode: missing}, {mode: cutoff})
                with torch.inference_mode():
                    ranked = pointer(features[0], features[1], masks)[:, :-1].argmax(-1).tolist()
                metrics[mode]['given_read_rank'] = sum(address is not None and dense.index_of(address, names) == guess
                    for (_, address), names, guess in zip(cases, features[2], ranked, strict=True))
                private_result[name] = {'rows': rows, 'metrics': metrics[mode]}
            output[mode] = {'steps': steps, 'cutoff': cutoff, 'public': public_result, 'personal': private_result}
            print(mode, 'public', public_result, 'personal', {k: v['metrics'] for k, v in private_result.items()}, flush=True)
    if any(digest(path) != expected for path, expected in bound.items()):
        raise RuntimeError('A bound source or the dense archive changed')
    for name, (pointer, aux) in trained.items():
        dest = args.output / f'{name}.pt'
        atomic_save({'pointer': pointer.state_dict(), 'public': aux.state_dict()}, dest)
        output[name]['sha256'] = digest(dest)
    store_json(args.output / 'result.json', dict(arms=output, labels=labels, training_views=len(supervision[0]),
               corpus={split: [[dialogue, turn, fingerprint(text)] for dialogue, turn, text, _ in rows]
                       for split, rows in selected.items()},
               seconds=perf_counter() - started, sources={str(path): value for path, value in bound.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('ordinary', type=Path)
    parser.add_argument('profile', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reader', type=Path, default=Path('runs/spora-dual-version-reply-v1'))
    parser.add_argument('--normalizer', type=Path, default=Path('runs/spora-version-normalizer-v1'))
    parser.add_argument('--recurrent', type=Path, nargs=3, required=True)
    main(parser.parse_args())
