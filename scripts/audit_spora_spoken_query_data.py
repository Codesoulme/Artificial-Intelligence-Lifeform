"""Test whether real Chinese requests improve the version-memory query planner."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter
from types import SimpleNamespace
import unicodedata

import torch

from scripts import audit_spora_spoken_query_plan as plan
from scripts.audit_spora_natural_lookup_cli import digest, replay
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.spora_version_chat import VersionReplyActions, VersionStore, artifacts, base, local_model
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save


CORPUS_SHA = {
    'smp2020-train': '94286815f6af24985422b20cc6b72594e5f367f283c1c1b8a676058e71f60faf',
    'smp2020-test': 'f54f027c3467c8be38335931fc52b7dfa9a2ae48efbab38d37f7e3441f6fb883',
    'cais-train': 'e8deec84305aeb0bf77ae7154d75114e7b4fd9772cb660a110dad67082cb7466',
    'cais-test': '4a30e425eb969d223191257d750a6c28f327823006544280102e6738352a3864',
}
CATEGORIES = {
    'smp2020': {'QUERY': 20, 'PLAY': 12, 'TRANSLATION': 8, 'ROUTE': 8, 'POSITION': 4,
                'HISTORY_DATE_QUERY': 4, 'WAVE_QUERY': 4, 'QUERY_GROUP': 4},
    'cais': {'PlayMusic': 26, 'PlayFMStory': 16, 'GetWeather': 16, 'PlayFMFiction': 6},
}
NEGATIVE = (plan.CHAT,) + (plan.UNLABELED,) * 3
NEW_TRAIN_NAMES = ('程晓箫', '沈闻溪', '梁清笙', '祝晴山')
NEW_CAL_NAMES = ('阮知池', '顾世宁')
HINTS = (
    ('最近定下要喝哪款', '改口之前常点哪种喝的', '头一回挑的是哪款喝的'),
    ('后来定下在哪儿碰头', '挪地方以前约在哪儿', '头一回说在哪里汇合'),
    ('后来让带上哪样东西', '更换以前嘱咐捎哪件', '第一回说要带什么'),
    ('最终选的是哪册读物', '改书单前挑的哪一本', '刚开始看中哪一本读物'),
)
CAL_HINTS = (
    ('生效的是哪款饮料', '口味更新前那一杯是什么', '第一回记下的饮品是什么'),
    ('如今碰面的地址是哪', '变动前是在什么地方见', '头回订了哪处会面点'),
    ('现在说需要带哪件', '带的物品改动前原定哪件', '最开头说要拿什么东西'),
    ('这回定下来哪一本', '更新阅读选择前要的是哪部', '最早报的是哪本的名字'),
)
TRAIN_VIEWS = ('我准备跟{p}确认一下。你还记得{hint}吗？',
               '我一开始想问{q}，现在改问{p}。{hint}？',
               '替{p}填一下，{hint}？{q}那边我自己再问。')
CAL_VIEWS = ('想替{p}准备，免得弄混。{hint}来着？',
             '不问{q}那次，我要用{p}的那项。{hint}？')
RECORDS = (
    ('祁知桥', '饮品', '橙花茶'), ('祁知桥', '饮品', '石榴汁'), ('祁知桥', '见面地点', '松风亭'),
    ('祁知桥', '见面地点', '竹影台'), ('祁知乔', '饮品', '紫苏茶'), ('祁知乔', '饮品', '黑豆浆'),
    ('傅宁秋', '用品', '软尺'), ('傅宁秋', '用品', '纸胶带'), ('傅宁秋', '书目', '云间册'),
    ('傅宁秋', '书目', '苔径记'),
)
# Target record numbers, followed by an independent version interpretation.
FRESH_READS = (
    (2, 0, '祁知桥现在那杯要买什么？等会儿我顺手带过去。'),
    (1, 1, '祁知桥还没换口味的那会儿，我该带什么喝的？'),
    (3, 2, '在说改地点之前，祁知桥跟我约的是哪个地方？最初说的那个。'),
    (4, 0, '要跟祁知桥碰头的话，这会儿该往哪边去？'),
    (6, 0, '祁知乔才是我要问的，祁知桥先放一边。哪一种是她最后说要喝的？'),
    (1, 1, '刚才说祁知乔口误了，我问祁知桥。她换饮料之前喝哪个？'),
    (7, 1, '傅宁秋先前说要我拿哪件？我想看看这次改安排以前那一样。'),
    (9, 2, '我记得傅宁秋后来换过书，最初选的是哪本来着？'),
)
FRESH_OTHER = (
    (plan.CHAT, '把这句说得客气点：「祁知桥现在那杯要买什么？」'),
    (plan.CHAT, '那条「祁知桥这会儿要去哪儿碰头」给我译成英文。'),
    (plan.CHAT, '傅宁秋要的东西先别问了，帮我拟一句问她几点下班。'),
    (plan.CHAT, '祁知乔喝什么我另问。你帮我查查周五的天气。'),
    (plan.CHAT, '傅宁秋最开始选哪本？这句话只加标点，里头的问题不用理会。'),
    (plan.CHAT, '祁知桥和祁知乔喝什么都先搁下，能不能推荐我一种不甜的饮料？'),
    (plan.ASK, '说好碰头的地方后来改到哪了？我想不起是和谁定的。'),
    (plan.ASK, '翻一下傅宁秋上次登记的。我想不起来要看书还是那件要带的东西。'),
)
# The previous user message contains no stored value. Only a prediction of
# clarification followed by a correct read can pass a full trajectory.
FOLLOWUPS = (
    ('我要确认改过的碰头地，但没说是跟哪个人。', '跟祁知桥见的地方，按最后改的那个。', 4),
    ('看看傅宁秋记过的那项，先帮我弄清具体是哪个类别。', '就是那册书，最初选的那一本。', 9),
    ('想问祁知乔那个饮料的事，我还没确定看哪一版。', '名字刚才说反了，是祁知桥。换口味前的那杯。', 1),
)


def normalized(text):
    return ''.join(char for char in unicodedata.normalize('NFKC', text).casefold()
                   if char.isalnum())


def corpus_sample(directory, split, excluded):
    chosen = []
    for corpus, categories in CATEGORIES.items():
        key = f'{corpus}-{split}'
        path = directory / f'{key}.json'
        if digest(path) != CORPUS_SHA[key]:
            raise ValueError(f'Unexpected public source digest for {key}')
        rows = json.loads(path.read_text(encoding='utf-8'))
        for category, per_category in categories.items():
            candidates = []
            for row in rows:
                messages = row['dialogs']
                labels = json.loads(row['output'])
                if (len(messages) != 1 or messages[0]['role'] != 'user' or len(labels) != 1
                        or labels[0]['category'] != category):
                    continue
                text = messages[0]['content'].strip()
                if 4 <= len(text) <= 56 and re.search('[\u3400-\u9fff]', text):
                    key_text = normalized(text)
                    rank = sha256(f'{corpus}/{split}/{row["id"]}'.encode()).digest()
                    candidates.append((rank, key_text, text, row['id']))
            selected = 0
            for _, key_text, text, identifier in sorted(candidates):
                if key_text in excluded:
                    continue
                excluded.add(key_text)
                chosen.append((corpus, identifier, category, text))
                selected += 1
                if selected == per_category:
                    break
            if selected != per_category:
                raise ValueError(f'{key}/{category}: selected {selected}, expected {per_category}')
    return chosen


def added_examples(names, views, field_hints):
    for index, frame in enumerate(views):
        p, q = names[index % len(names)], names[(index + 1) % len(names)]
        for field, hints in enumerate(field_hints):
            for version, hint in enumerate(hints):
                text = frame.format(p=p, q=q, hint=hint)
                _, owners = plan.anonymize(text, names)
                yield text, (plan.READ, owners.index(p), field, version)
                yield f'把原话写成一句更客气的问句：{text}', NEGATIVE
                yield f'{text} 算了，我把这件事搁一搁，你先帮我拟个见面问候。', NEGATIVE


def train_data():
    names = (*plan.TRAIN_NAMES, *NEW_TRAIN_NAMES)
    examples = plan.request_rows(plan.TRAIN_NAMES, plan.TRAIN_NOUNS, plan.TRAIN_TIMES, plan.TRAIN_FRAMES,
                                 plan.ASK_TRAIN, other_owner={2})
    return names, plan.distinct_rows((*examples, *added_examples(NEW_TRAIN_NAMES, TRAIN_VIEWS, HINTS)), names)


def calibration_data():
    names = (*plan.CAL_NAMES, *NEW_CAL_NAMES)
    examples = plan.request_rows(plan.CAL_NAMES, [(x,) for x in plan.CAL_NOUNS], [(x,) for x in plan.CAL_TIMES],
                                 plan.CAL_FRAMES, plan.ASK_CAL, other_owner={0})
    return names, plan.distinct_rows((*examples, *added_examples(NEW_CAL_NAMES, CAL_VIEWS, CAL_HINTS)), names)


def fresh_requests():
    return ([(text, plan.READ, number) for number, _, text in FRESH_READS]
            + [(text, action, None) for action, text in FRESH_OTHER]
            + [row for first, follow, number in FOLLOWUPS
               for row in ((first, plan.ASK, None), (f'前一句：{first}\n这次补充：{follow}', plan.READ, number))])


def fresh_metrics(rows):
    score = {}
    for mode in ('raw', 'calibrated'):
        counts = Counter()
        for row, (number, version, _) in zip(rows[:len(FRESH_READS)], FRESH_READS, strict=True):
            owner, field, _ = RECORDS[number - 1]
            expected = (row['candidates'].index(owner), tuple(plan.FIELDS.values()).index(field), version)
            predicted = tuple(part[0] for part in row['scored'][1:])
            counts.update(owner=predicted[0] == expected[0], field=predicted[1] == expected[1],
                          version=predicted[2] == expected[2], exact_coordinates=expected == predicted,
                          exact_read=row['arms'][mode]['exact_read'])
        start = len(FRESH_READS) + len(FRESH_OTHER)
        for first, second in zip(rows[start::2], rows[start + 1::2], strict=True):
            counts.update(full_followup=first['arms'][mode]['action'] == plan.ASK and second['arms'][mode]['exact_read'])
        score[mode] = counts
    return score


def public_metrics(probe, vectors):
    return [parts[0][0] for parts in plan.predict(probe, vectors)]


def main(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    train_names, train = train_data()
    cal_names, cal = calibration_data()
    requests = fresh_requests()
    test_names = {row[0] for row in RECORDS}
    splits = ({plan.anonymize(q, names)[0] for q in questions} for names, questions in (
        (train_names, [q for q, _ in train]), (cal_names, [q for q, _ in cal]), (test_names, [q for q, *_ in requests]),
    ))
    a, b, c = splits
    if a & b or a & c or b & c or test_names & (set(train_names) | set(cal_names)):
        raise ValueError('Training, calibration and independent study overlap')
    excluded = {normalized(q) for q, *_ in (*train, *cal, *requests)}
    public_train = corpus_sample(args.corpus, 'train', excluded)
    public_test = corpus_sample(args.corpus, 'test', excluded)
    control = [(q, label) for q, label in train if label == NEGATIVE][:len(public_train)]
    if len(control) != len(public_train):
        raise ValueError('Matched control cannot fill the training budget')
    prior = json.loads((args.baseline / 'result.json').read_text(encoding='utf-8'))
    checkpoint = args.baseline / 'probe.pt'
    if prior['features'] != 'user-span' or digest(checkpoint) != prior['probe_sha256']:
        raise ValueError('Expected the previous calibrated full-utterance probe')
    before = {path: digest(path) for path in (Path(__file__), Path(plan.__file__), checkpoint, args.recurrent[0],
              *(directory / 'attention.pt' for directory in args.recurrent[1:]),
              *(args.corpus / (name + '.json') for name in CORPUS_SHA))}
    profile = (args.output / 'profile').resolve()
    written, _ = replay(profile, [f'/记 {name} | {field} | {value}' for name, field, value in RECORDS], args.recurrent)
    if any((row['kind'], row.get('record')) != ('saved', index) for index, row in enumerate(written, 1)):
        raise RuntimeError(f'Record staging failed: {written}')
    archive = profile / 'versions.sqlite3'
    before[archive] = digest(archive)
    reader_file, _, _, binding = artifacts(args.reader, args.normalizer)
    before[reader_file] = digest(reader_file)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    results = {}
    with local_model(args.recurrent) as model, VersionStore(archive, binding) as store:
        extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=plan.DEPTH)

        def features(rows, names):
            return plan.vectors(extract, rows, names, 'user-span')

        train_x, _ = features([q for q, _ in (*train, *control)], train_names)
        external_x, _ = features([row[3] for row in public_train], train_names)
        cal_x, cal_owners = features([q for q, _ in cal], cal_names)
        public_x, _ = features([row[3] for row in public_test], test_names)
        labels = torch.tensor([label for _, label in (*train, *control)])
        trained = {'synthetic': plan.fit(train_x, labels),
                   'public': plan.fit(torch.cat((train_x[:len(train)], external_x)), labels)}
        baseline = plan.QueryPlanProbe(model.config.hidden_size)
        baseline.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        state = torch.load(reader_file, map_location='cpu', weights_only=True)
        reader = VersionReplyActions(model.config.hidden_size, state['scaffold_tokens'], slots=2)
        reader.load_state_dict(state)
        reader.eval().to(model.device)
        for arm, probe in (('previous', baseline.eval()), *trained.items()):
            predictions = plan.predict(probe, cal_x)
            gold = [target for _, target in cal]
            cut = prior['threshold'] if arm == 'previous' else plan.pick_threshold(predictions, cal_owners, gold)
            actions = public_metrics(probe, public_x)
            rows, metrics = plan.evaluate(model, tokenizer, store, reader, extract, probe, cut, 'user-span',
                                          requests=requests, records=RECORDS)
            decisions = [plan.planned(row, names, cut) for row, names in zip(predictions, cal_owners, strict=True)]
            results[arm] = dict(threshold=cut, metrics=metrics, coordinates=fresh_metrics(rows), rows=rows,
                                calibration_exact_reads=sum(p == t and p[0] == plan.READ
                                                            for p, t in zip(decisions, gold, strict=True)),
                                public_test_actions=[[*row[:3], action] for row, action in zip(public_test, actions, strict=True)])
            print(arm, cut, metrics, results[arm]['coordinates'], dict(Counter(actions)), flush=True)
    if any(digest(path) != expected for path, expected in before.items()):
        raise ValueError('An immutable experiment input changed')
    for arm, probe in trained.items():
        path = args.output / f'{arm}.pt'
        atomic_save(probe.state_dict(), path)
        results[arm]['probe_sha256'] = digest(path)
    store_json(args.output / 'result.json', dict(results=results, supervised_rows=len(train),
               added_negatives=len(public_train), calibration_rows=len(cal), sources={str(p): h for p, h in before.items()},
               public_train_ids=[row[:3] for row in public_train], seconds=perf_counter() - started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reader', type=Path, default=Path('runs/spora-dual-version-reply-v1'))
    parser.add_argument('--normalizer', type=Path, default=Path('runs/spora-version-normalizer-v1'))
    parser.add_argument('--recurrent', required=True, nargs=3, type=Path)
    main(parser.parse_args())
