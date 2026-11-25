"""Learn whether to ask for a missing memory coordinate before accessing a record."""

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn

from scripts import audit_spora_dense_joint_address as dense
from scripts import audit_spora_read_confirmation_action as action
from scripts import audit_spora_spoken_query_plan as plan
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_version_dialogue_scope import PartialLookup


CHAT, ADDRESS, ASK_FIELD, ASK_OWNER = range(4)
CLASSES = ASK_OWNER + len(plan.FIELDS_ORDER)
TRAIN = (
    ('我想核实{p}{x}。', '记录里{x}？说的是谁我等下再给你。',
     '我想调出{p}存过的那条，具体哪一类等我想想。', '关于{p}{x}，我要把这句改得自然些。'),
    ('{p}{x}，还记不记得？', '{x}来着？我这里还没说名字。',
     '{p}以前给过一个答案，你能帮忙找吗？是哪个方面的我先理一下。', '{p}{x}等下我亲自联系确认。'),
    ('麻烦说下{p}{x}，按当时定的。', '已经说定了，想问{x}，我先只给你这个线索。',
     '有{p}的旧记录吧？我想找一项，还没想好给哪一项。', '假设有人问{p}{x}，这句话怎么说得更客气？'),
    ('我替{p}准备之前要确认一下{x}。', '想回忆前面{x}，人名一会儿告诉你。',
     '是{p}，我只记得写过，想从留着的资料里核对一下。', '帮我想个新安排，{p}{x}先别翻旧的。'),
    ('你还留着{p}{x}的记录吗？我要那项。', '想用档上那一项，应该是{x}，名字我还没报。',
     '{p}的我得再看一眼，细目我还没说。', '{p}{x}我知道了，你接着聊别的就行。'),
    ('给我报一下{p}{x}。', '那项已经定了，{x}？你先等等我说是谁。',
     '{p}的资料里我想翻出一项，我来补是关于什么的。', '{p}{x}只是例子，帮我写两句问候。'),
)
CAL = (
    ('我需要确认{p}{x}，按前面登记的。', '你还记着{x}吧，我得核对；稍后告诉你名字。',
     '我说的是{p}，还想核实旧资料里的一项，范围我等会补。', '{p}{x}我自己会问，先别管。'),
    ('{p}{x}？我想听之前留的那份。', '那项的旧值我得确认，{x}，人选还没告诉你。',
     '我这次想看看{p}那边记下来的内容，具体指哪件先保留。', '{p}{x}我拿来当个示例，你评价一下表述。'),
)
HELD = (
    (ASK_OWNER, 0, 'drinks', '说定的要喝哪一种？这事儿我还得核实，名字稍后给你。'),
    (ASK_OWNER, 1, 'places', '上次约在什么地方来着？先把这一项记着，待会我说找谁。'),
    (ASK_OWNER, 2, 'items', '我得知道前头托我捎去的东西是啥。是谁托的我还没告诉你。'),
    (ASK_OWNER, 3, 'books', '之前挑的是哪本读物？我把姓名给你，你再告诉我当时那册。'),
    (ASK_FIELD, 0, 'places', '夏宁瑶的旧安排我想核对一条，确切是哪方面我再补。'),
    (ASK_FIELD, 1, 'books', '有个事我想回头看，先给你个人名，是夏宁遥。哪一件我马上说。'),
    (ASK_FIELD, 2, 'drinks', '郭回霜留过一项我想确认。只知道是她，暂时还没说类别。'),
    (ASK_FIELD, 3, 'items', '我准备找郭回双原先那条留底。等我想想是说什么的。'),
    (ADDRESS, 0, 'drinks', '夏宁瑶选的那杯喝的是啥？我想沿用她说过的那种。'),
    (ADDRESS, 1, 'items', '夏宁遥让我带过去那件东西，你还记得叫啥吗？'),
    (ADDRESS, 2, 'books', '郭回霜当时选的是哪本读物呀？我打算找原来那册。'),
    (ADDRESS, 3, 'places', '和郭回双碰头选在哪儿了？我得照之前讲好的去。'),
    (ADDRESS, 0, 'places', '那当前版呢？'),
    (CHAT, None, None, '夏宁瑶选了什么我一会儿问她，先帮我组织个开场白。'),
    (CHAT, None, None, '之前选哪杯已经有人告诉我了。我现在想听点新的建议。'),
    (CHAT, None, None, '照这个说法润色：郭回双以前说想读哪一本？'),
    (CHAT, None, None, '碰面的路怎么走更合适？先别找以前的约定。'),
    (CHAT, None, None, '所谓回顾旧安排是什么意思？我只是问这几个字。'),
    (CHAT, None, None, '我托夏宁遥买笔袋，她不用翻原来的记录。'),
    (CHAT, None, None, '帮我拟一句给郭回霜的话，问她之后想选什么喝的。'),
    (CHAT, None, None, '我先把人名留空，等问完了再回来。'),
)
CONTEXT_REQUEST = '请查夏宁瑶的见面地点当前版。'


def supervise(frames, names, aliases):
    rows = {}
    for group in frames:
        for index, family in enumerate(aliases):
            for j, alias in enumerate(family):
                for label, template in zip((ADDRESS, ASK_OWNER, ASK_FIELD, CHAT), group, strict=True):
                    target = ASK_OWNER + index if label == ASK_OWNER else label
                    question = template.format(p=names[j % len(names)], x=alias)
                    if rows.setdefault(question, target) != target:
                        raise ValueError('Conflicting clarification labels')
    return list(rows.items())


def new_head(width):
    return nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Linear(64, CLASSES))


def fit(states, labels):
    torch.manual_seed(action.SEED)
    generator = torch.Generator().manual_seed(action.SEED)
    head = new_head(states.shape[1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    targets = torch.tensor(labels)
    for _ in range(16):
        for batch in torch.randperm(len(labels), generator=generator).split(24):
            loss = nn.functional.cross_entropy(head(states[batch]), targets[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return head.eval()


@torch.inference_mode()
def scores(head, features):
    confidence, labels = head(features).softmax(dim=-1).max(dim=-1)
    return list(zip(labels.tolist(), confidence.tolist(), strict=True))


def select_threshold(calibration, public):
    def utility(limit):
        right = sum(prediction == label and label != CHAT and score >= limit
                    for (prediction, score), (_, label) in calibration)
        wrong = sum(prediction not in (label, CHAT) and score >= limit
                    for (prediction, score), (_, label) in calibration)
        wrong += sum(prediction != CHAT and score >= limit for prediction, score in public)
        return right - 2 * wrong, -wrong, -limit
    return max((.4, .5, .6, .7, .8, .9, .95, .99, 1.000001), key=utility)


def proposed(label, names, address):
    if label == ADDRESS:
        return address
    if label == ASK_FIELD and len(names) == 1:
        return PartialLookup(owner=names[0])
    if label >= ASK_OWNER:
        return PartialLookup(field=plan.FIELDS_ORDER[label - ASK_OWNER])
    return None


def expected_proposal(kind, address):
    if kind == ASK_OWNER:
        return PartialLookup(field=address.field)
    if kind == ASK_FIELD:
        return PartialLookup(owner=address.name)
    return address


def replay(store, model, tokenizer, reader, settings, suggestions):
    chat = action.VersionChat(store, model, tokenizer, reader, lambda evidence: [None] * len(evidence), settings,
                              ordinary_tokens=12, typed_lookup=True, spoken=lambda question, _: suggestions.get(question))
    today = date.today()
    catalog = store.catalog()
    values = {action.MemoryAddress(name, field): value for name, field, value in dense.RECORDS}
    results, counts = [], Counter()
    for kind, person, field, question in HELD:
        chat.reset_dialogue()
        target = None if person is None else action.MemoryAddress(dense.NAMES[person], field)
        if question == '那当前版呢？':
            if not chat.turn(CONTEXT_REQUEST, today)['selected']:
                raise RuntimeError('The context control failed to establish an explicit address')
        first = chat.turn(question, today)
        continuation = None
        offered = chat.partial_lookup or chat.spoken_pending
        correct_offer = offered is not None and offered == expected_proposal(kind, target)
        if offered is not None and first['selected']:
            raise RuntimeError('Clarification accessed storage before a reply')
        if correct_offer:
            if kind == ASK_OWNER:
                answer = f'{target.name}当前版'
            elif kind == ASK_FIELD:
                answer = f'{plan.FIELDS[field]}当前版'
            else:
                answer = '是，当前版。'
            continuation = chat.turn(answer, today)
        if offered is not None and not correct_offer:
            continuation = chat.turn('先不查了。', today)
            if continuation['selected']:
                raise RuntimeError('Cancellation initiated a read')
        final = continuation or first
        expected_id = None if target is None else action.resolve_address(
            catalog, target, action.VersionQuery('current'), today, retain_same_day=True)[0]
        actual = final['selected']
        known_values = [value for value in values.values() if value in final['answer']]
        exact = target is not None and actual == [expected_id] and known_values == [values[target]]
        counts.update(owner_exact=kind == ASK_OWNER and exact, field_exact=kind == ASK_FIELD and exact,
                      addressed_exact=kind == ADDRESS and exact, unnecessary_clarifications=kind == CHAT and offered is not None,
                      wrong_clarifications=kind != CHAT and offered is not None and not correct_offer,
                      wrong_reads=bool(actual) and (expected_id is None or actual != [expected_id]))
        results.append({'question': question, 'action': kind, 'first': first, 'followup': continuation, 'exact': exact})
    return dict(summary=counts, rows=results)


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    prior = json.loads((args.previous / 'result.json').read_text(encoding='utf-8'))
    archive = args.profile / 'profile' / 'versions.sqlite3'
    reader_file, _, writer_file, binding = action.artifacts(args.reader, args.normalizer)
    pointer_file = args.transfer / 'personal.pt'
    checkpoint, *layers = args.recurrent
    original_head = args.previous / 'action-19.pt'
    protected = {Path(path).resolve(): checksum for path, checksum in prior['sources'].items()}
    protected[original_head.resolve()] = prior['results']['19']['sha256']
    files = (archive, reader_file, writer_file, pointer_file, checkpoint, original_head,
             *(folder / 'attention.pt' for folder in layers))
    for path in files:
        if protected.get(path.resolve()) != action.digest(path):
            raise ValueError(f'Unexpected source digest for {path.name}')
    args.output.mkdir(parents=True, exist_ok=False)
    train = supervise(TRAIN, action.TRAIN_NAMES, dense.ALIASES)
    cal = supervise(CAL, action.CAL_NAMES, dense.CAL_ALIASES)
    questions = [question for _, _, _, question in HELD]
    keys = lambda rows, names: {plan.anonymize(question, names)[0] for question, _ in rows}
    train_keys, cal_keys = keys(train, action.TRAIN_NAMES), keys(cal, action.CAL_NAMES)
    held_keys = keys([(question, CHAT) for question in questions], dense.NAMES)
    if train_keys & cal_keys or held_keys & (train_keys | cal_keys):
        raise ValueError('Repeated prompts cross the evaluation split')

    excluded = set()
    public_train = [text for *_, text in action.corpus_sample(args.corpus, 'train', excluded)]
    public_test = [text for *_, text in action.corpus_sample(args.corpus, 'test', excluded)]
    known_train = [(text, CHAT) for index, text in enumerate(public_train) if index % 4]
    tokenize = action.base.AutoTokenizer.from_pretrained(action.base.MODEL, local_files_only=True, trust_remote_code=False)
    settings = torch.load(writer_file, map_location='cpu', weights_only=True)
    with action.local_model(args.recurrent) as model, action.VersionStore(archive, binding) as store:
        state = torch.load(reader_file, map_location='cpu', weights_only=True)
        reader = action.VersionReplyActions(model.config.hidden_size, state['scaffold_tokens'], slots=2)
        reader.load_state_dict(state)
        reader.eval().to(model.device)
        pointer = dense.JointPointer(model.config.hidden_size).eval()
        pointer.load_state_dict(torch.load(pointer_file, map_location='cpu', weights_only=True)['pointer'])
        early = action.FrozenTurnFeatures(SimpleNamespace(base=model), tokenize, depth=plan.DEPTH)
        deep = action.FrozenTurnFeatures(SimpleNamespace(base=model), tokenize, depth=19)
        address_inputs = dense.align(early, questions, dense.NAMES)
        addresses, _ = action.rank(pointer, address_inputs, store.catalog())
        states = action.action_features(deep, questions, dense.NAMES)
        training = train + known_train
        head = fit(action.action_features(deep, [text for text, _ in training], action.TRAIN_NAMES),
                   [label for _, label in training])
        cal_scores = scores(head, action.action_features(deep, [text for text, _ in cal], action.CAL_NAMES))
        public_scores = scores(head, action.action_features(deep, public_train[::4], ()))
        cutoff = select_threshold(list(zip(cal_scores, cal, strict=True)), public_scores)
        baseline = nn.Sequential(nn.LayerNorm(states.shape[1]), nn.Linear(states.shape[1], 64),
                                 nn.GELU(), nn.Linear(64, 1))
        baseline.load_state_dict(torch.load(original_head, map_location='cpu', weights_only=True))
        baseline.eval()
        baseline_scores = action.predict(baseline, states)
        baseline_limit = prior['results']['19']['cutoff']
        predictions = scores(head, states)
        arms = {
            'explicit_only': {},
            'whole_address': {question: address for question, address, score in zip(
                questions, addresses, baseline_scores, strict=True) if score >= baseline_limit},
            'coordinate': {question: proposed(label, names, address)
                           for question, names, address, (label, score) in zip(
                               questions, address_inputs[2], addresses, predictions, strict=True) if score >= cutoff},
            'oracle': {q: expected_proposal(kind, action.MemoryAddress(dense.NAMES[person], field))
                       for kind, person, field, q in HELD if person is not None},
        }
        results = {}
        for name, suggestions in arms.items():
            data = replay(store, model, tokenize, reader, settings, suggestions)
            results[name] = data
            print(name, data['summary'], flush=True)
        test_scores = scores(head, action.action_features(deep, public_test, ()))
    if any(action.digest(path) != protected[path.resolve()] for path in files):
        raise RuntimeError('A protected model or archive changed')
    path = args.output / 'action.pt'
    atomic_save(head.state_dict(), path)
    store_json(args.output / 'result.json', {
        'results': results, 'predictions': predictions, 'baseline_scores': baseline_scores, 'threshold': cutoff,
        'train': len(training), 'calibration': len(cal), 'public_test': len(test_scores),
        'public_rejected': sum(label == CHAT or score < cutoff for label, score in test_scores),
        'seconds': perf_counter() - started, 'action_sha256': action.digest(path),
        'sources': {str(path): protected[path.resolve()] for path in files},
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous', type=Path)
    parser.add_argument('transfer', type=Path)
    parser.add_argument('profile', type=Path)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reader', type=Path, default=Path('runs/spora-dual-version-reply-v1'))
    parser.add_argument('--normalizer', type=Path, default=Path('runs/spora-version-normalizer-v1'))
    parser.add_argument('--recurrent', type=Path, nargs=3, required=True)
    main(parser.parse_args())
