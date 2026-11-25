"""Compare whole-request authority with a decision grounded in one named address."""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from scripts import audit_spora_spoken_query_plan as plan
from scripts.audit_spora_natural_lookup_cli import digest
from scripts.audit_spora_spoken_lookup_dialogue import FRESH
from scripts.audit_spora_spoken_query_data import corpus_sample
from scripts.audit_spora_spoken_temporal_head import TIMELINES
from scripts.spora_version_chat import FIELD_IDS, FrozenTurnFeatures, base, local_model
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_version_dialogue_scope import eligible_spoken_lookup
from spora.research_version_store import MemoryAddress


SEED = 51861
FIELDS = tuple(FIELD_IDS.values())
ALIASES = (
    ('点过哪杯喝的', '手头登记了什么饮料', '挑中的是哪一种喝的'),
    ('约好在哪里碰头', '说要在哪儿会合', '最终在哪儿见'),
    ('让我捎的是哪件东西', '需要拿过去的物件叫什么', '托我带上哪一样'),
    ('选的是哪一册', '原先说要借哪本书', '后来决定读哪一本'),
)
# Each pair preserves the people and subject and changes the scope of the refusal.
TRAIN = (
    ('{a}先留着不问。我还想知道{b}{what}。', '{a}先留着不问。{b}{what}也不用告诉我。'),
    ('{b}{what}这件事帮我确认一下。至于{a}那项先不管。',
     '{b}{what}这件事也先不用确认。至于{a}那项先不管。'),
    ('本来想听{a}{what}，现在只说{b}那一份。', '本来想听{a}{what}，现在{b}那一份也不用了。'),
    ('{a}和{b}的安排我都有疑问。先略过{a}，给我说说{b}{what}？',
     '{a}和{b}的安排我都有疑问。这次先都略过，我自己再打听。'),
    ('{a}那项不用查。{b}我还没弄清{what}，给我报一下。',
     '{a}那项不用查。{b}我还没弄清{what}，这次也不用报。'),
    ('先不说{b}的事。想了想，{a}继续搁置，{b}{what}得告诉我。',
     '先不说{b}的事。想了想，{a}继续搁置，{b}{what}我自己去问。'),
)
CAL = (
    ('{a}那边下回再说，{b}{what}我得核实。', '{a}那边下回再说，{b}{what}也改天再核实。'),
    ('别报{a}的。我只问{b}{what}。', '别报{a}的。{b}{what}也先不必说。'),
    ('需要{b}{what}，另外{a}这次先跳过。', '{b}{what}暂且不用，另外{a}这次先跳过。'),
)


@dataclass(frozen=True)
class Request:
    text: str
    owner: str | None
    field: str | None
    names: tuple[str, ...]

    @property
    def address(self):
        return MemoryAddress(self.owner, self.field) if self.owner is not None else None


FAMILIES = (
    ('罗映荷', 'drinks', '叶盼山',
     '叶盼山先搁一边。罗映荷那杯究竟说好买哪样？我出门前想弄明白。',
     '叶盼山先搁一边。罗映荷那杯究竟说好买哪样？想想也算了。'),
    ('罗映禾', 'places', '罗映荷',
     '见罗映禾这件事还照商定的地方，我该过去哪里？罗映荷先没空见。',
     '见罗映禾的地点我会自己问清楚。罗映荷先没空见。'),
    ('穆云川', 'items', '罗映禾',
     '把罗映禾的要求先撤掉。穆云川提过有件东西要我捎过去，那件是什么？',
     '把罗映禾的要求先撤掉。穆云川提过有件东西要我捎过去，我也不准备管这件了。'),
    ('叶盼山', 'books', '罗映荷',
     '罗映荷没要书，先不用理会。那叶盼山手上选中的那册叫什么？',
     '罗映荷没要书，先不用理会。叶盼山手上选中的那册我也会另问。'),
    ('罗映荷', 'drinks', '罗映禾',
     '本想问罗映禾，口误。我代罗映荷买喝的，告诉我留在册子里的那款吧。',
     '本想问罗映禾，口误。我代罗映荷买喝的，但我会现挑一款新的。'),
    ('罗映禾', 'places', '叶盼山',
     '要和罗映禾见面的去处帮我想起来。叶盼山去哪里这事先不用。',
     '要和罗映禾见面的去处先别从记录找。叶盼山去哪里这事也先不用。'),
    ('穆云川', 'items', '叶盼山',
     '我不需知道叶盼山取什么。穆云川那件之前就说了要带，问的就是它。',
     '我不需知道叶盼山取什么。穆云川那件之前就说了要带，我打算直接去问她。'),
    ('叶盼山', 'books', '穆云川',
     '穆云川这次书另选。叶盼山挑的那本还是按她说过的，你能说出来吗？',
     '穆云川这次书另选。叶盼山挑的那本我也打算另外找。'),
)


def synthetic(frames, owners):
    result = []
    for positive, negative in frames:
        for field, aliases in zip(FIELDS, ALIASES, strict=True):
            for order in range(2):
                a, b = owners if order == 0 else owners[::-1]
                for what in aliases:
                    terms = dict(a=a, b=b, what=what)
                    result.append(Request(positive.format(**terms), b, field, owners))
                    result.append(Request(negative.format(**terms), None, field, owners))
    return list(dict.fromkeys(result))


def held_data():
    return [row for owner, field, other, positive, negative in FAMILIES
            for row in (Request(positive, owner, field, (owner, other)), Request(negative, None, None, (owner, other)))]


def raw_owners(row):
    normalized, owners = plan.anonymize(row.text, row.names)
    return normalized.rsplit('\n', 1)[0], owners


def address_input(row, address):
    question, owners = raw_owners(row)
    position = '甲乙'[owners.index(address.name)] if address.name in owners else '某人'
    label = plan.FIELDS[address.field]
    return f'本次候选是{position}的{label}。用户原话：{question}'


def batches(extract, prompts):
    prompts = list(prompts)
    return plan.vectors(extract, prompts, (), 'user-span')[0]


def fit(features, targets):
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)
    head = nn.Sequential(nn.LayerNorm(features.shape[-1]), nn.Linear(features.shape[-1], 64),
                         nn.GELU(), nn.Linear(64, 1))
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    for _ in range(12):
        for batch in torch.randperm(len(targets), generator=rng).split(32):
            scores = head(features[batch]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(scores, targets[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return head.eval()


def probabilities(head, inputs):
    with torch.inference_mode():
        return head(inputs).squeeze(-1).sigmoid().tolist()


def proposed(scores, owners):
    index, field = scores[1][0], scores[2][0]
    return MemoryAddress(owners[index], plan.FIELDS_ORDER[field]) if index < len(owners) else None


def score_rows(rows, predictions, threshold):
    summary = Counter()
    for row, (address, likelihood) in zip(rows, predictions, strict=True):
        accepted = address is not None and likelihood >= threshold and eligible_spoken_lookup(row.text)
        summary.update(correct=accepted and address == row.address, wrong=accepted and row.address is not None and address != row.address,
                       unnecessary=accepted and row.address is None, missed=not accepted and row.address is not None)
    return summary


def threshold(calibration, predictions):
    def quality(cut):
        row = score_rows(calibration, predictions, cut)
        mistakes = row['wrong'] + row['unnecessary']
        return row['correct'] - 2 * mistakes, -mistakes, -cut
    return max((.5, .6, .7, .8, .85, .9, .95, .99, 1.000001), key=quality)


def predict(extract, model, heads, requests, catalog=None):
    bare = [raw_owners(row) for row in requests]
    features = batches(extract, (text for text, _ in bare))
    frozen = plan.predict(model, features)
    candidates = [proposed(scored, owners) for scored, (_, owners) in zip(frozen, bare, strict=True)]
    if catalog is not None:
        candidates = [candidate if candidate in catalog else None for candidate in candidates]
    conditioned = [address_input(row, address or MemoryAddress('', FIELDS[0]))
                   for row, address in zip(requests, candidates, strict=True)]
    target_vectors = batches(extract, conditioned)
    old = [(plan.proposed_address(scored, owners, .9), 1.) for scored, (_, owners) in zip(frozen, bare, strict=True)]
    if catalog is not None:
        old = [(address if address in catalog else None, score) for address, score in old]
    return {'old': old, 'whole': list(zip(candidates, probabilities(heads['whole'], features), strict=True)),
            'target': list(zip(candidates, probabilities(heads['target'], target_vectors), strict=True))}


def training_rows(train, public):
    world = [Request(text, None, None, ()) for _, _, _, text in public]
    whole = [(raw_owners(row)[0], row.address is not None) for row in (*train, *world)]
    target = []
    # A positive request still excludes the other speaker for the same field.
    for index, row in enumerate(train):
        _, names = raw_owners(row)
        field = row.field or FIELDS[index % len(FIELDS)]
        candidates = [MemoryAddress(owner, field) for owner in names]
        target.extend((address_input(row, address), row.address == address) for address in candidates)
    target.extend((address_input(row, MemoryAddress('', FIELDS[index % len(FIELDS)])), False)
                  for index, row in enumerate(world))
    return {'whole': whole, 'target': target}


def logs(rows, output, limits):
    serial, stats = [], {}
    for mode, scores in output.items():
        stats[mode] = score_rows(rows, scores, limits[mode])
    for index, row in enumerate(rows):
        views = {}
        for mode, scores in output.items():
            address, score = scores[index]
            candidate = ((address.name, address.field) if address is not None else None)
            views[mode] = dict(candidate=candidate, score=score,
                               ask=candidate is not None and score >= limits[mode] and eligible_spoken_lookup(row.text))
        expected = (row.owner, row.field) if row.address is not None else None
        serial.append(dict(question=row.text, expected=expected, arms=views))
    return stats, serial


def main(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    source = args.previous / 'result.json'
    probe_path = args.previous / 'public.pt'
    base_report = json.loads(source.read_text(encoding='utf-8'))
    if digest(probe_path) != base_report['results']['public']['probe_sha256']:
        raise ValueError('Expected the previously audited address probe')
    inputs = (source, probe_path, Path(__file__), args.recurrent[0], *(p / 'attention.pt' for p in args.recurrent[1:]))
    originals = {p: digest(p) for p in inputs}
    binding = {Path(path).resolve(): value for path, value in base_report['sources'].items()}
    if any(binding.get(path.resolve()) != originals[path] for path in inputs[3:]):
        raise ValueError('Address probe does not match the selected backbone')
    train = synthetic(TRAIN, ('庄安澜', '项云知'))
    cal = synthetic(CAL, ('廖达远', '乔卉南'))
    held = held_data()
    fresh = [Request(text, label.split('的')[0], FIELD_IDS[label.split('的')[1]],
                     tuple(owner for owner, _, _ in TIMELINES)) for text, label in FRESH]
    catalog = {MemoryAddress(owner, FIELD_IDS[field]) for owner, field, _ in TIMELINES}
    data = [set(raw_owners(row)[0] for row in set_) for set_ in (train, cal, (*held, *fresh))]
    if data[0] & data[1] or data[0] & data[2] or data[1] & data[2]:
        raise ValueError('Train, calibration and held-out sentences overlap')
    excluded = set()
    public_train = corpus_sample(args.corpus, 'train', excluded)
    public_test = corpus_sample(args.corpus, 'test', excluded)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    with local_model(args.recurrent) as model:
        extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=plan.DEPTH)
        old = plan.QueryPlanProbe(model.config.hidden_size)
        old.load_state_dict(torch.load(probe_path, map_location='cpu', weights_only=True))
        old.eval()
        heads = {}
        samples = training_rows(train, public_train)
        for mode, examples in samples.items():
            heads[mode] = fit(batches(extract, (text for text, _ in examples)),
                             torch.tensor([label for _, label in examples], dtype=torch.float32))
        cal_predictions = predict(extract, old, heads, cal)
        limits = dict(old=.9, **{mode: threshold(cal, cal_predictions[mode]) for mode in heads})
        cal_stats, cal_rows = logs(cal, cal_predictions, limits)
        held_stats, held_rows = logs(held, predict(extract, old, heads, held, catalog), limits)
        fresh_stats, fresh_rows = logs(fresh, predict(extract, old, heads, fresh, catalog), limits)
        # Test text transformations with exact enrolled names and potentially relevant content.
        negatives = [Request(f'把这句话改得礼貌些：{row.text}', None, None, row.names) for row in held[::2]]
        negative_stats, negative_rows = logs(negatives, predict(extract, old, heads, negatives, catalog), limits)
        plain_rows = [Request(text, None, None, ()) for _, _, _, text in public_test]
        plain_features = batches(extract, (row.text for row in plain_rows))
        plain_target = batches(extract, (address_input(row, MemoryAddress('', FIELDS[index % len(FIELDS)]))
                                         for index, row in enumerate(plain_rows)))
        public_rejected = {
            'old': sum(scored[0][0] == plan.CHAT for scored in plan.predict(old, plain_features)),
            'whole': sum(value < limits['whole'] for value in probabilities(heads['whole'], plain_features)),
            'target': sum(value < limits['target'] for value in probabilities(heads['target'], plain_target)),
        }
    if any(digest(path) != stable for path, stable in originals.items()):
        raise ValueError('A bound input changed')
    checksums = {}
    for mode, head in heads.items():
        dest = args.output / f'{mode}.pt'
        atomic_save(head.state_dict(), dest)
        checksums[mode] = digest(dest)
    store_json(args.output / 'result.json', dict(threshold=limits, cal=cal_stats, held=held_stats, fresh=fresh_stats,
               negatives=negative_stats, public_rejected=public_rejected, rows=dict(cal=cal_rows, held=held_rows,
               fresh=fresh_rows, negatives=negative_rows), training={mode: len(rows) for mode, rows in samples.items()},
               sources={str(path): value for path, value in originals.items()}, sha256=checksums,
               seconds=perf_counter() - started))
    print(limits, 'calibration', cal_stats, 'held', held_stats, 'fresh', fresh_stats,
          'transformation', negative_stats, 'public', public_rejected, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous', type=Path)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--recurrent', type=Path, nargs=3, required=True)
    main(parser.parse_args())
