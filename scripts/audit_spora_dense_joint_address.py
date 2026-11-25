"""Learn a joint memory address and reject requests whose target is absent."""

import argparse
from collections import Counter
from datetime import date
from itertools import product
import json
from pathlib import Path
import re
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from scripts import audit_spora_spoken_query_plan as plan
from scripts.audit_spora_natural_lookup_cli import digest, replay
from scripts.audit_spora_spoken_query_data import corpus_sample
from scripts.spora_version_chat import (FrozenTurnFeatures, VersionReplyActions, VersionStore, artifacts, base,
                                        compress_dimension, local_model, reply_to_route, spoken_lookup_planner)
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_multi_version_route import resolve_address
from spora.research_version_store import MemoryAddress
from spora.research_version_timeline import VersionQuery


WIDTH = len(plan.FIELDS_ORDER)
NONE = 2 * WIDTH
SEED = 81247
NAMES = ('夏宁瑶', '夏宁遥', '郭回霜', '郭回双')
RECORDS = tuple((owner, field, value) for owner, values in zip(NAMES, (
    ('青梅汁', '杉木廊', '灰布包', '星灯记'), ('米浆', '月桂台', '竹书签', '溪畔信'),
    ('薄荷茶', '白石亭', '软尺', '河滩集'), ('石榴汤', '风桥阁', '蓝绳', '雾岸篇'),
), strict=True) for field, value in zip(plan.FIELDS_ORDER, values, strict=True))
TRAIN = (
    ('原本想问{a}{x}，这个搁置。{b}{y}那件能不能报一下？', 1),
    ('{a}{x}我需要知道。{b}{y}我会另约，别管这一项。', 0),
    ('弄反了，我不用{b}{y}。还是照{a}{x}来，先报一下。', 0),
    ('{a}{x}暂时放下。{b}{y}呢？我只核实这项。', 1),
    ('想过问{a}{x}，也想过问{b}{y}。我改主意了，两边我自己再打听。', None),
    ('{a}{x}是个例句，{b}{y}也是例句。帮我比较两句语气就行。', None),
    ('{a}{x}和{b}{y}先不提了，帮我拟条晚安留言。', None),
    ('本来要核对{a}{x}，也提到{b}{y}。想了想这些都留到下回处理。', None),
)
CAL = (
    ('关于{a}{x}先别下结论。{b}{y}这才是我想查询的，帮我说一下。', 1),
    ('要问的是{a}{x}，不是{b}{y}，请只报一项。', 0),
    ('{a}{x}也好，{b}{y}也罢，我都有办法问到，先聊些别的。', None),
)
ALIASES = (
    ('定的那杯是什么', '过去挑的饮料是哪一种'), ('约的碰头处在哪里', '原定的集合点在哪儿'),
    ('说要捎的那件叫什么', '让我准备的是哪件东西'), ('借的是哪一册', '挑出来的是哪本读物'),
)
CAL_ALIASES = (('选的喝的是什么',), ('说好在哪儿碰头',), ('需要用的是哪件物品',), ('改选的是哪本书',))
HELD = (
    ('夏宁瑶带什么东西先略过。我需要知道夏宁遥那杯说好了要买什么。', 1, 'drinks'),
    ('夏宁遥不是今天碰头的人。我就想核实跟夏宁瑶原先敲定在哪里见。', 0, 'places'),
    ('夏宁瑶挑的书由她再定，我得先替夏宁遥拿她说的那样东西，叫什么？', 1, 'items'),
    ('问夏宁遥喝什么是我刚才说错了。我要找夏宁瑶定下的那本，书叫什么？', 0, 'books'),
    ('先把郭回霜借阅的那本留着。郭回双要喝的说好了是哪样？', 3, 'drinks'),
    ('郭回霜说要我到她选的地儿集合，我该去哪？郭回双准备什么我另外问。', 2, 'places'),
    ('我和郭回霜的见面地点先不用问。我要的是郭回双要拿的那一件。', 3, 'items'),
    ('不是问郭回双缺什么用品，帮我回想郭回霜最后选中的是哪本。', 2, 'books'),
    ('夏宁瑶的会合地点和夏宁遥喝的先搁着。我直接找她们核实。', None, None),
    ('郭回双的那本和郭回霜那杯都别提了，我现在想听个笑话。', None, None),
    ('夏宁瑶的饮品我自己买，夏宁遥要带的我也会另外准备。', None, None),
    ('郭回霜和郭回双之前的安排我有数，约个新去处就行。', None, None),
)


def practice(frames, aliases, names):
    examples = []
    for (frame, expected), (left, right) in product(frames, product(range(WIDTH), repeat=2)):
        for offset in range(len(aliases[left])):
            text = frame.format(a=names[0], b=names[1], x=aliases[left][offset], y=aliases[right][offset])
            address = None if expected is None else MemoryAddress(names[expected], plan.FIELDS_ORDER[(left, right)[expected]])
            examples.append((text, address))
    return tuple(dict.fromkeys(examples))


def align(extract, questions, enrolled):
    inputs, owners = zip(*(plan.anonymize(question, enrolled) for question in questions), strict=True)
    states, spans = [], []
    for begin in range(0, len(inputs), 24):
        batch = inputs[begin:begin + 24]
        hidden = extract(batch).float()
        for index, prompt in enumerate(batch):
            text = prompt.rsplit('\n', 1)[0]
            values = hidden[index, :len(text)]
            states.append(values.mean(dim=0))
            markers = list(re.finditer('[甲乙]', text))
            owner_vectors = [torch.zeros_like(states[-1]), torch.zeros_like(states[-1])]
            counts = [0, 0]
            for pos, match in enumerate(markers):
                other = markers[pos + 1].start() if pos + 1 < len(markers) else len(text)
                slot = '甲乙'.index(match[0])
                owner_vectors[slot] += values[match.start():min(other, match.start() + 24)].mean(dim=0)
                counts[slot] += 1
            spans.append(torch.stack([state / max(count, 1) for state, count in zip(owner_vectors, counts, strict=True)]))
    return torch.stack(states), torch.stack(spans), owners


class JointPointer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.shared = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 64), nn.GELU())
        self.local = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 64), nn.GELU())
        self.owner_position = nn.Parameter(torch.zeros(2, 64))
        self.output = nn.Linear(64, WIDTH)
        self.none = nn.Linear(64, 1)

    def forward(self, global_state, evidence, allowed):
        global_state = self.shared(global_state)
        located = self.local(evidence)
        candidates = self.output(F.gelu(located + global_state[:, None] + self.owner_position)).flatten(1)
        return torch.cat((candidates.masked_fill(~allowed, -torch.inf), self.none(global_state)), dim=-1)


def index_of(address, names):
    return NONE if address is None else names.index(address.name) * WIDTH + plan.FIELDS_ORDER.index(address.field)


def mask_for(names, addresses):
    return torch.tensor([owner < len(names) and MemoryAddress(names[owner], field) in addresses
                         for owner, field in product(range(2), plan.FIELDS_ORDER)], dtype=torch.bool)


def supervised(data, owners, synthetic_count):
    indices, masks, targets = [], [], []
    for index, ((_, address), names) in enumerate(zip(data, owners, strict=True)):
        target = index_of(address, names)
        full = torch.ones(NONE, dtype=torch.bool) if index >= synthetic_count else mask_for(
            names, {MemoryAddress(owner, field) for owner, field in product(names, plan.FIELDS_ORDER)})
        indices.append(index)
        masks.append(full)
        targets.append(target)
        if address is not None:
            withheld = full.clone()
            withheld[target] = False
            indices.append(index)
            masks.append(withheld)
            targets.append(NONE)
    return torch.tensor(indices), torch.stack(masks), torch.tensor(targets)


def fit(states, spans, supervision, *, local):
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)
    head = JointPointer(states.shape[-1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    indices, masks, labels = supervision
    for _ in range(16):
        for rows in torch.randperm(len(labels), generator=rng).split(32):
            chosen = indices[rows]
            features = spans[chosen]
            logits = head(states[chosen], features if local else torch.zeros_like(features), masks[rows])
            loss = F.cross_entropy(logits, labels[rows])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return head.eval()


def proposals(head, inputs, masks, *, local):
    state, evidence, _ = inputs
    with torch.inference_mode():
        distribution = head(state, evidence if local else torch.zeros_like(evidence), masks).softmax(-1)
    weight, slot = distribution.max(-1)
    return list(zip(slot.tolist(), weight.tolist(), strict=True))


def calibrate(cal, with_target, removed, owners):
    def quality(threshold):
        correct = mistakes = 0
        for ((_, needed), names), (answer, confidence), (substitute, confidence_missing) in zip(
                zip(cal, owners, strict=True), with_target, removed, strict=True):
            expected = index_of(needed, names)
            if confidence >= threshold and answer != NONE:
                correct += answer == expected
                mistakes += answer != expected
            if needed is not None and confidence_missing >= threshold and substitute != NONE:
                mistakes += 1
        return correct - 2 * mistakes, -mistakes, -threshold
    return max((0., .4, .5, .6, .7, .8, .9, .95, 1.000001), key=quality)


def subtract_targets(masks, cases, names):
    changed = masks.clone()
    for i, ((_, address), owners) in enumerate(zip(cases, names, strict=True)):
        if address is not None:
            changed[i, index_of(address, owners)] = False
    return changed


def addressed(index, names):
    return (MemoryAddress(names[index // WIDTH], plan.FIELDS_ORDER[index % WIDTH])
            if 0 <= index < len(names) * WIDTH else None)


@torch.inference_mode()
def evaluate(store, model, tokenizer, reader, cases, owners, predictions, missing, limits):
    catalog = store.catalog()
    embedding = model.get_input_embeddings().weight
    cols = compress_dimension(embedding.shape[1]).to(embedding.device)
    codebook = F.normalize(embedding.index_select(1, cols).float(), dim=-1)
    values = {MemoryAddress(owner, field): value for owner, field, value in RECORDS}
    counters = {mode: Counter() for mode in predictions}
    rows = []
    for i, ((question, target), names) in enumerate(zip(cases, owners, strict=True)):
        if any(value in question for value in values.values()):
            raise ValueError('Test question may not reveal the stored value')
        outcomes, answers = {}, {}
        for mode in predictions:
            choice, confidence = predictions[mode][i]
            drop, confidence_missing = missing[mode][i]
            address = addressed(choice, names) if confidence >= limits[mode] else None
            substitute = addressed(drop, names) if confidence_missing >= limits[mode] else None
            selected, reply = None, None
            if address in catalog:
                selected, _ = resolve_address(catalog, address, VersionQuery('current'), date.today())
                if selected is not None:
                    if selected not in answers:
                        try:
                            _, result = reply_to_route(model, tokenizer, store, codebook, reader, question, [selected],
                                                       labels=[plan.FIELDS[address.field]])
                            answers[selected] = result['answer']
                        except ValueError as error:
                            answers[selected] = str(error)
                    reply = answers[selected]
            shown = [value for value in values.values() if reply is not None and value in reply]
            correct = target is not None and address == target and shown == [values[target]]
            counters[mode].update(exact=correct, wrong=selected is not None and address != target,
                                  missing_rejected=target is not None and substitute is None,
                                  illegal_substitute=target is not None and substitute is not None)
            outcomes[mode] = dict(raw_slot=choice, raw_probability=confidence, missing_slot=drop,
                                  missing_probability=confidence_missing, read=selected, answer=reply, correct=correct)
        rows.append(dict(question=question, expected=index_of(target, names), owners=names, arms=outcomes))
    return rows, counters


def main(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    train_names = ('程理新', '贺梦川')
    cal_names = ('周靖岭', '颜曦音')
    training = practice(TRAIN, ALIASES, train_names)
    calibration = practice(CAL, CAL_ALIASES, cal_names)
    held = [(text, MemoryAddress(NAMES[which], field) if which is not None else None) for text, which, field in HELD]
    splits = [set(plan.anonymize(text, names)[0] for text, _ in rows)
              for rows, names in ((training, train_names), (calibration, cal_names), (held, NAMES))]
    if splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2]:
        raise ValueError('Train, calibration and held questions overlap')
    external = corpus_sample(args.corpus, 'train', set())
    all_training = (*training, *((text, None) for _, _, _, text in external))
    profile = (args.output / 'profile').resolve()
    replies, _ = replay(profile, [f'/记 {owner} | {plan.FIELDS[field]} | {value}' for owner, field, value in RECORDS], args.recurrent)
    if any((row['kind'], row.get('record')) != ('saved', i) for i, row in enumerate(replies, 1)):
        raise ValueError(f'Unable to write independent dense profile: {replies}')
    database = profile / 'versions.sqlite3'
    reader_file, _, _, binding = artifacts(args.reader, args.normalizer)
    public_path = args.previous / 'public.pt'
    sources = {path: digest(path) for path in (Path(__file__), database, reader_file, public_path, args.recurrent[0],
                                              *(folder / 'attention.pt' for folder in args.recurrent[1:]))}
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    with local_model(args.recurrent) as model, VersionStore(database, binding) as store:
        spoken_lookup_planner(args.previous, args.recurrent, model, tokenizer)
        extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=plan.DEPTH)
        train_state, train_spans, train_owners = align(extract, [text for text, _ in all_training], train_names)
        target = supervised(all_training, train_owners, len(training))
        cal_inputs = align(extract, [text for text, _ in calibration], cal_names)
        held_inputs = align(extract, [text for text, _ in held], NAMES)
        cal_catalog = {MemoryAddress(owner, field) for owner, field in product(cal_names, plan.FIELDS_ORDER)}
        cal_masks = torch.stack([mask_for(names, cal_catalog) for names in cal_inputs[2]])
        test_masks = torch.stack([mask_for(names, store.catalog()) for names in held_inputs[2]])
        cal_removed = subtract_targets(cal_masks, calibration, cal_inputs[2])
        test_removed = subtract_targets(test_masks, held, held_inputs[2])
        heads, cutoffs, results, missing = {}, {}, {}, {}
        for mode, local in (('global', False), ('local', True)):
            head = heads[mode] = fit(train_state, train_spans, target, local=local)
            cutoffs[mode] = calibrate(calibration, proposals(head, cal_inputs, cal_masks, local=local),
                                      proposals(head, cal_inputs, cal_removed, local=local), cal_inputs[2])
            results[mode] = proposals(head, held_inputs, test_masks, local=local)
            missing[mode] = proposals(head, held_inputs, test_removed, local=local)
        old_features, _ = plan.vectors(extract, [text for text, _ in held], NAMES, 'user-span')
        old = plan.QueryPlanProbe(model.config.hidden_size).eval()
        old.load_state_dict(torch.load(public_path, map_location='cpu', weights_only=True))
        current = [plan.proposed_address(row, names, .9) for row, names in zip(plan.predict(old, old_features), held_inputs[2], strict=True)]
        results['old'] = [(index_of(address, names), 1.) for address, names in zip(current, held_inputs[2], strict=True)]
        missing['old'] = [(slot if slot < NONE and test_removed[i, slot] else NONE, 1.)
                          for i, (slot, _) in enumerate(results['old'])]
        cutoffs['old'] = 0.
        reader_state = torch.load(reader_file, map_location='cpu', weights_only=True)
        reader = VersionReplyActions(model.config.hidden_size, reader_state['scaffold_tokens'], slots=2)
        reader.load_state_dict(reader_state)
        rows, stats = evaluate(store, model, tokenizer, reader.eval().to(model.device), held, held_inputs[2], results, missing, cutoffs)
    if any(digest(path) != expected for path, expected in sources.items()):
        raise RuntimeError('Experiment changed a source or the dense archive')
    checksums = {}
    for mode, head in heads.items():
        path = args.output / f'{mode}.pt'
        atomic_save(head.state_dict(), path)
        checksums[mode] = digest(path)
    store_json(args.output / 'result.json', dict(rows=rows, metrics=stats, thresholds=cutoffs, sha256=checksums,
               training=len(all_training), training_views=len(target[0]), calibration=len(calibration),
               seconds=perf_counter() - started, sources={str(path): sha for path, sha in sources.items()}))
    print(stats, cutoffs, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous', type=Path)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reader', type=Path, default=Path('runs/spora-dual-version-reply-v1'))
    parser.add_argument('--normalizer', type=Path, default=Path('runs/spora-version-normalizer-v1'))
    parser.add_argument('--recurrent', type=Path, nargs=3, required=True)
    main(parser.parse_args())
