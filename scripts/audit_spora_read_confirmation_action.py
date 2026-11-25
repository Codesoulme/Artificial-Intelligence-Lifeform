"""Train the decision to ask, while preserving the learned address scores."""

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import re
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from scripts import audit_spora_crosswoz_field_transfer as transfer
from scripts import audit_spora_dense_joint_address as dense
from scripts import audit_spora_spoken_query_plan as plan
from scripts.audit_spora_natural_lookup_cli import digest
from scripts.audit_spora_spoken_query_data import corpus_sample
from scripts.spora_version_chat import (FrozenTurnFeatures, VersionChat, VersionReplyActions, VersionStore,
                                        artifacts, base, local_model)
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_multi_version_route import resolve_address
from spora.research_version_dialogue_scope import eligible_spoken_lookup
from spora.research_version_store import MemoryAddress
from spora.research_version_timeline import VersionQuery


SEED = 280621
TRAIN = (
    ('{a}{x}。我得用早就定的那种，给我说说。', '{a}{x}。先算了，这一项我自己另问。'),
    ('{b}那边暂时放一放。{a}{x}？有记录就告诉我。', '{b}那边暂时放一放。{a}{x}？这一条我也另找答案。'),
    ('{a}{x}？我先问着，这条需要确认。', '{a}{x}？这句是台词，我只想看看合不合适。'),
    ('我原来是想问{a}{x}，现在想了想，还是请你报一下。',
     '我原来是想问{a}{x}，现在想了想，留到以后再说。'),
    ('{a}{x}和{b}那项有点像。{b}这次不用说，只说{a}的。',
     '{a}{x}和{b}那项有点像。我会去向她们确认，你先等等。'),
    ('{b}{x}先撤了，换问{a}。她这个早就说定了，你还记得是什么吗？',
     '{b}{x}先撤了，换问{a}。算了，她这个问题也先寄着。'),
    ('{a}{x}？不是要重新选，告诉我先前那项就可以。', '{a}{x}？我问完想换个新的，不必沿用以前那项。'),
    ('{a}{x}？{b}的安排另外再打听。', '{a}{x}？这几句给我理一理说法，意思别变。'),
)
CAL = (
    ('{a}{x}我得弄清楚。{b}这里不用问。', '{a}{x}我会亲自弄清楚。{b}这里也不用问。'),
    ('{b}的事先搁置，不过{a}{x}仍需要你告诉我。', '{b}的事先搁置，{a}{x}这一题也留到改天。'),
    ('我现在要核实{a}{x}，你给我报原值吧。', '我现在想把句子「{a}{x}」写得简短些。'),
)
TRAIN_NAMES = ('沈昕晗', '俞青照')
CAL_NAMES = ('唐霁川', '乔宁鹤')
HELD = (
    ('夏宁遥那碗先不管，夏宁瑶之前说想要什么喝的？', 0, 'drinks'),
    ('夏宁瑶一会儿另有事。我要和夏宁遥碰面，原来约的地方在哪边？', 1, 'places'),
    ('不是问夏宁遥订的是哪册。夏宁瑶让我捎去的那件叫什么？', 0, 'items'),
    ('夏宁瑶拿什么不用说。夏宁遥借的读物是哪一本来着？', 1, 'books'),
    ('郭回双的先略过。郭回霜选好喝哪种了吗？想知道她说的那杯。', 2, 'drinks'),
    ('我说的是跟郭回双碰面。最后选的去处是什么呀？郭回霜我直接问。', 3, 'places'),
    ('郭回双读什么先放着。郭回霜说好了要我带的物件叫什么？', 2, 'items'),
    ('郭回霜借过的那本别管。我想找郭回双点过名的那一册。', 3, 'books'),
    ('夏宁瑶和夏宁遥喝什么我会自己买，你帮我想一下路上说什么。', None, None),
    ('郭回双读过什么下回聊。郭回霜那件事我现在也不打算查。', None, None),
    ('把这句话改得礼貌些：夏宁遥借过的是哪册？', None, None),
    ('郭回霜什么时候到我自行打电话。郭回双在等我，替我写句道歉。', None, None),
)


def pairs(frames, aliases, people):
    rows = []
    for positive, negative in frames:
        for family in aliases:
            for subject in family:
                values = dict(a=people[0], b=people[1], x=subject)
                rows.append((positive.format(**values), negative.format(**values)))
    return tuple(dict.fromkeys(rows))


def action_features(extract, questions, enrolled):
    normalized, _ = zip(*(plan.anonymize(text, enrolled) for text in questions), strict=True)
    states = []
    for begin in range(0, len(normalized), 24):
        batch = normalized[begin:begin + 24]
        hidden = extract(batch).float()
        for index, prompt in enumerate(batch):
            text = prompt.rsplit('\n', 1)[0].rstrip('。！？?!，, \r\n\t')
            boundaries = list(re.finditer('[。！？?!；;\n]', text))
            left = boundaries[-1].end() if boundaries else 0
            state = hidden[index, :max(len(text), 1)]
            states.append(torch.cat((state.mean(0), state[left:].mean(0))))
    return torch.stack(states)


def fit(features, paired_count, total_public):
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)
    layer = nn.Sequential(nn.LayerNorm(features.shape[-1]), nn.Linear(features.shape[-1], 64),
                          nn.GELU(), nn.Linear(64, 1))
    opt = torch.optim.AdamW(layer.parameters(), lr=.002)
    for _ in range(16):
        for batch in torch.randperm(paired_count, generator=rng).split(24):
            pos = layer(features[2 * batch]).squeeze(-1)
            neg = layer(features[2 * batch + 1]).squeeze(-1)
            samples = torch.randint(total_public, (len(batch),), generator=rng) + paired_count * 2
            ordinary = layer(features[samples]).squeeze(-1)
            loss = (F.softplus(-pos).mean() + F.softplus(neg).mean() + F.softplus(ordinary).mean()) / 3
            loss = loss + .25 * F.softplus(1 - (pos - neg)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return layer.eval()


@torch.inference_mode()
def predict(head, states):
    return head(states).squeeze(-1).sigmoid().tolist()


def choose_threshold(cal_scores, accepted, ordinary):
    def quality(level):
        correct = sum(score >= level and ok for score, ok in zip(cal_scores[::2], accepted[::2], strict=True))
        errors = sum(score >= level and ok for score, ok in zip(cal_scores[1::2], accepted[1::2], strict=True))
        errors += sum(score >= level for score in ordinary)
        return correct - 2 * errors, -errors, -level
    return max((.4, .5, .6, .7, .8, .9, .95, .99, 1.000001), key=quality)


@torch.inference_mode()
def rank(pointer, features, catalog):
    states, spans, names = features
    allowed = torch.stack([dense.mask_for(row, catalog) for row in names])
    logits = pointer(states, spans, allowed)
    best = logits[:, :-1].argmax(-1).tolist()
    return ([dense.addressed(slot, owners) for slot, owners in zip(best, names, strict=True)],
            [int(index) != dense.NONE for index in logits.argmax(-1)])


def run_dialogue(store, model, tokenizer, reader, settings, cases, candidates, active):
    catalog = store.catalog()
    lookup = {question: target for (question, _), target, accepted in zip(cases, candidates, active, strict=True)
              if accepted and target is not None}
    chat = VersionChat(store, model, tokenizer, reader, lambda rows: [None] * len(rows), settings,
                       ordinary_tokens=12, typed_lookup=True, spoken=lambda question, _: lookup.get(question))
    values = {MemoryAddress(owner, field): value for owner, field, value in dense.RECORDS}
    outcomes, counts = [], Counter()
    for (question, expected), candidate, authorized in zip(cases, candidates, active, strict=True):
        chat.reset_dialogue()
        first = chat.turn(question, date.today())
        proposed = 'unconfirmed_address' in first.get('reasons', [])
        correct_offer = proposed and expected is not None and candidate == expected
        confirmation = None
        exact = False
        if proposed:
            if first.get('selected'):
                raise RuntimeError('Proposal unexpectedly read a version before confirmation')
            confirmation = chat.turn('是，当前版。' if correct_offer else '不对。', date.today())
            if correct_offer:
                wanted, reason = resolve_address(catalog, expected, VersionQuery('current'), date.today())
                shown = [value for value in values.values() if value in confirmation['answer']]
                exact = (reason == 'located' and confirmation.get('selected') == [wanted]
                         and shown == [values[expected]])
            elif confirmation.get('selected'):
                raise RuntimeError('Declined proposal read a version')
        counts.update(action_accepted=authorized and expected is not None, correct_offer=correct_offer, exact=exact,
                      wrong_offer=proposed and candidate != expected, before_confirmation_reads=bool(first.get('selected')))
        outcomes.append(dict(question=question, expected=None if expected is None else [expected.name, expected.field],
                             authorized=authorized, first=first, confirmation=confirmation, passed=exact))
    return {'summary': counts, 'rows': outcomes}


def main(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    report = json.loads((args.transfer / 'result.json').read_text(encoding='utf-8'))
    known = {Path(path).resolve(): value for path, value in report['sources'].items()}
    archive = args.profile / 'profile' / 'versions.sqlite3'
    checkpoint = args.transfer / 'personal.pt'
    reader_file, _, writer_file, binding = artifacts(args.reader, args.normalizer)
    backbone = (args.recurrent[0], *(folder / 'attention.pt' for folder in args.recurrent[1:]))
    previous_sources = (archive, reader_file, *backbone)
    paths = (*previous_sources, checkpoint, writer_file, Path(__file__))
    sources = {path: digest(path) for path in paths}
    if (sources[checkpoint] != report['arms']['personal']['sha256']
            or any(sources[path] != known[path.resolve()] for path in previous_sources)):
        raise ValueError('Expected the audited dense profile and address head')
    train = pairs(TRAIN, dense.ALIASES, TRAIN_NAMES)
    cal = pairs(CAL, dense.CAL_ALIASES, CAL_NAMES)
    train_text = [text for pair in train for text in pair]
    cal_text = [text for pair in cal for text in pair]
    held = {'previous': transfer.make_cases(transfer.FRESH), 'new': transfer.make_cases(HELD)}
    train_keys = {plan.anonymize(text, TRAIN_NAMES)[0] for text in train_text}
    other_keys = {plan.anonymize(text, CAL_NAMES)[0] for text in cal_text}
    held_keys = {plan.anonymize(text, dense.NAMES)[0] for cases in held.values() for text, _ in cases}
    if train_keys & other_keys or train_keys & held_keys or other_keys & held_keys:
        raise ValueError('The templates leak between training and held samples')
    excluded = set()
    ordinary = [text for *_, text in corpus_sample(args.corpus, 'train', excluded)]
    public_test = [text for *_, text in corpus_sample(args.corpus, 'test', excluded)]
    # Hold back every fourth public training row for threshold selection.
    cal_ordinary = ordinary[::4]
    train_ordinary = [text for index, text in enumerate(ordinary) if index % 4]
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    settings = torch.load(writer_file, map_location='cpu', weights_only=True)
    results, heads = {}, {}
    with local_model(args.recurrent) as model, VersionStore(archive, binding) as store:
        state = torch.load(reader_file, map_location='cpu', weights_only=True)
        reader = VersionReplyActions(model.config.hidden_size, state['scaffold_tokens'], slots=2)
        reader.load_state_dict(state)
        reader.eval().to(model.device)
        pointer = dense.JointPointer(model.config.hidden_size).eval()
        pointer.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['pointer'])
        base_extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=plan.DEPTH)
        fixed = {name: rank(pointer, dense.align(base_extract, [text for text, _ in cases], dense.NAMES), store.catalog())
                 for name, cases in held.items()}
        for depth in (12, 19):
            extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=depth)
            x = action_features(extract, train_text + train_ordinary, TRAIN_NAMES)
            head = heads[depth] = fit(x, len(train), len(train_ordinary))
            cal_scores = predict(head, action_features(extract, cal_text, CAL_NAMES))
            ordinary_scores = predict(head, action_features(extract, cal_ordinary, ()))
            cutoff = choose_threshold(cal_scores, [eligible_spoken_lookup(q) for q in cal_text], ordinary_scores)
            evaluation = {name: predict(head, action_features(extract, [text for text, _ in cases], dense.NAMES))
                          for name, cases in held.items()}
            public = predict(head, action_features(extract, public_test, ()))
            outputs = {name: run_dialogue(store, model, tokenizer, reader, settings, cases, fixed[name][0],
                                          [score >= cutoff for score in evaluation[name]]) for name, cases in held.items()}
            results[str(depth)] = {'cutoff': cutoff, 'public_rejected': sum(score < cutoff for score in public),
                                   'action_scores': evaluation, 'held': outputs,
                                   'cal_positive': sum(value >= cutoff for value in cal_scores[::2])}
            print(depth, cutoff, 'public rejected', results[str(depth)]['public_rejected'],
                  {name: output['summary'] for name, output in outputs.items()}, flush=True)
    if any(digest(path) != expected for path, expected in sources.items()):
        raise RuntimeError('A bound checkpoint or profile changed')
    for depth, head in heads.items():
        path = args.output / f'action-{depth}.pt'
        atomic_save(head.state_dict(), path)
        results[str(depth)]['sha256'] = digest(path)
    store_json(args.output / 'result.json', dict(results=results, train_pairs=len(train), calibration_pairs=len(cal),
               address_baseline={name: {'none': len(rows[1]) - sum(rows[1]), 'answers':
                         [None if a is None else [a.name, a.field] for a in rows[0]]} for name, rows in fixed.items()},
               seconds=perf_counter() - started, sources={str(path): value for path, value in sources.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('transfer', type=Path)
    parser.add_argument('profile', type=Path)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reader', type=Path, default=Path('runs/spora-dual-version-reply-v1'))
    parser.add_argument('--normalizer', type=Path, default=Path('runs/spora-version-normalizer-v1'))
    parser.add_argument('--recurrent', type=Path, nargs=3, required=True)
    main(parser.parse_args())
