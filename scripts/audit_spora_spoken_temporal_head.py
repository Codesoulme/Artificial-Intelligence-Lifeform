"""Train a temporal decision head and test it on distinct persistent revisions."""

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from scripts import audit_spora_spoken_query_plan as plan
from scripts.audit_spora_natural_lookup_cli import digest, replay
from scripts.audit_spora_spoken_query_data import corpus_sample
from scripts.spora_version_chat import (FIELD_IDS, FrozenTurnFeatures, TemporalProposals, VersionReplyActions,
                                        VersionStore, artifacts, base, compress_dimension, local_model, reply_to_route)
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_multi_version_route import parse_lookup_scope, resolve_address, spoken_scope_evidence
from spora.research_version_store import MemoryAddress


CURRENT, PREVIOUS, FIRST, UNCLEAR = range(4)
SEED = 60433
TEMPORAL_TRAIN = (
    ('刚换上去的那份是什么', '刚换下来那份是什么', '第一次写进去的是什么'),
    ('最近一次改成什么了', '最近一次改动前是什么', '最早一次说定的是什么'),
    ('上次更新填进的是哪个', '上次更新替掉的是哪个', '最先填的那个是哪个'),
    ('这阵子实际用的怎么样', '这阵子换掉的那样是什么', '头回挑的哪一个'),
    ('新调整到哪一项了', '新调整前是哪一项', '开头报的是哪一项'),
    ('这回刚定下的是什么', '这回刚删掉的是什么', '有记录时第一笔是什么'),
    ('手头要照办的是什么', '换成现在这样前用的是什么', '刚开始照办的是什么'),
    ('现在有效的那条是什么', '现在这一条接替了哪一条', '当初还没改过时是哪一条'),
)
TEMPORAL_CAL = (
    ('新换来的是哪一个', '新换下来的是哪一个', '最初用的是哪一个'),
    ('刚改定的到底是哪一份', '刚被改掉的是哪一份', '启用时拿的哪一份'),
    ('眼下到底要按哪个', '现用这一份换掉了哪个', '第一个决定下来的是哪个'),
)
UNCLEAR_TRAIN = ('我们先对一对是要旧的还是新的', '有好几次改动，随便说哪一回',
                 '我没想好按最早那一项，还是刚改的这项', '之前那一个是什么，哪回也说不准',
                 '帮我报这条记录，我没指定用哪回的', '新版或旧版都先给我看看能不能选')
UNCLEAR_CAL = ('原来那个还是新换来的，我得先想想', '我想查一项，还不知道问的是哪回',
               '那早些年的某一次是哪一条，具体什么时候我忘记了')
TRAIN_VIEWS = ('我问甲的{field}，{cue}？', '乙的不用问。只说甲的{field}，{cue}？',
               '甲的{field}，本来我想问{foil}，改口了。说说{cue}？',
               '关于甲的{field}，{cue}？知道这个就行。')
CAL_VIEWS = ('甲那条{field}我想确认一遍。{cue}？', '有关甲的{field}先别按{foil}回答。现在问{cue}。')

TIMELINES = (
    ('罗映荷', '饮品', ('山楂茶', '糯米汤', '桂花乳')),
    ('罗映禾', '见面地点', ('海棠廊', '望湖台', '枫桥亭')),
    ('穆云川', '用品', ('木夹子', '红绳', '折叠尺')),
    ('叶盼山', '书目', ('松间泉', '月下笺', '野渡舟')),
)
RECORDS = tuple((owner, field, value) for owner, field, values in TIMELINES for value in values)
HELD = (
    (0, CURRENT, '罗映荷的饮品，刚改到口头那杯是哪种？'),
    (0, PREVIOUS, '罗映荷的饮品，刚从单上划掉换走的那杯是哪种？'),
    (0, FIRST, '刚建罗映荷那份饮料单的时候，开篇那杯叫什么？'),
    (1, CURRENT, '罗映禾的约见地，改来改去以后最后落在哪儿？'),
    (1, PREVIOUS, '罗映禾的约见地，最后一次挪地方之前在哪儿？'),
    (1, FIRST, '罗映禾的碰头地在后来挪过几回。第一次见面时是约在哪儿的？'),
    (2, CURRENT, '穆云川要我拿的物件，末了换上的是哪一件？'),
    (2, PREVIOUS, '穆云川要我拿的物件，末了被换下的是哪一件？'),
    (2, FIRST, '穆云川要我拿的物件，第一天谈定的又是哪一件？'),
    (3, CURRENT, '叶盼山书目，开始那本作罢，我想知道最后保留的是哪本。'),
    (3, PREVIOUS, '叶盼山书目，开始那本作罢，我想知道最后被撤下的是哪本。'),
    (3, FIRST, '叶盼山书目，最后那本先别说，从还没改过的最初那份里报书名。'),
    (0, UNCLEAR, '罗映荷的饮品，我想起来她换过几回，之前是哪杯我说不清具体哪次。'),
    (1, UNCLEAR, '罗映禾约见的地方我得核对一下，头回约的还是最后约的我还没定。'),
    (2, UNCLEAR, '穆云川要带的东西是哪件？'),
    (3, UNCLEAR, '找叶盼山选的某一次书目就好，我忘了更改的先后。'),
)


def examples(phrases, unresolved, views):
    rows = []
    fields = tuple(FIELD_IDS)
    for index, cues in enumerate(phrases):
        for view in views:
            for label, cue in enumerate(cues):
                foil = cues[(label + 1) % len(cues)]
                text = view.format(field=fields[index % len(fields)], cue=cue, foil=foil)
                begin = text.index(cue)
                rows.append((text, label, (begin, begin + len(cue))))
    for index, cue in enumerate(unresolved):
        for view in views:
            text = view.format(field=fields[index % len(fields)], cue=cue, foil=phrases[0][index % 3])
            begin = text.index(cue)
            rows.append((text, UNCLEAR, (begin, begin + len(cue))))
    if len(rows) != len({text for text, *_ in rows}):
        raise ValueError('Temporal data must contain unique requests')
    return rows


def extracted(extract, rows):
    states = []
    for start in range(0, len(rows), 16):
        texts = [text for text, *_ in rows[start:start + 16]]
        hidden = extract(texts)
        states.extend(hidden[index, :len(text)].clone() for index, text in enumerate(texts))
    return pad_sequence(states, batch_first=True), torch.tensor([len(x[0]) for x in rows])


class TemporalHead(nn.Module):
    def __init__(self, width, *, focus):
        super().__init__()
        self.compression = nn.Linear(width, 64)
        self.focus = nn.Linear(64, 1) if focus else None
        self.classifier = nn.Linear(64, 4)

    def forward(self, vectors, limits):
        hidden = F.gelu(self.compression(F.layer_norm(vectors.float(), (vectors.shape[-1],))))
        mask = torch.arange(vectors.shape[1]) < limits[:, None]
        weights = mask / limits[:, None] if self.focus is None else self.focus(hidden).squeeze(-1).masked_fill(~mask, -torch.inf).softmax(-1)
        return self.classifier((hidden * weights.unsqueeze(-1)).sum(1)), weights


def fit(vectors, lengths, rows, *, focus):
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)
    head = TemporalHead(vectors.shape[-1], focus=focus)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    labels = torch.tensor([label for _, label, _ in rows])
    starts, ends = torch.tensor([span for _, _, span in rows]).T
    positions = torch.arange(vectors.shape[1])
    evidence = (positions >= starts[:, None]) & (positions < ends[:, None])
    for _ in range(20):
        for batch in torch.randperm(len(rows), generator=rng).split(24):
            scores, weights = head(vectors[batch], lengths[batch])
            loss = F.cross_entropy(scores, labels[batch])
            if focus:
                mass = (weights * evidence[batch]).sum(1).clamp_min(1e-8)
                loss = loss - .25 * mass.log().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return head.eval()


@torch.inference_mode()
def prediction(head, features):
    logits, _ = head(*features)
    confidence, choice = logits.softmax(-1).max(-1)
    return list(zip(choice.tolist(), confidence.tolist(), strict=True))


def calibrate(results, rows):
    def quality(cut):
        counts = Counter()
        for (label, confidence), (_, expected, _) in zip(results, rows, strict=True):
            if label != UNCLEAR and confidence >= cut:
                counts['correct' if label == expected else 'wrong'] += 1
        return -counts['wrong'], counts['correct'], -cut
    return max((0., .5, .6, .7, .8, .9, .95, 1.), key=quality)


def record_at(group, label):
    return group * 3 + (3, 2, 1)[label] if label != UNCLEAR else None


@torch.inference_mode()
def evaluate(model, tokenizer, store, reader, old, old_cut, inquiry, names, temporal, readings, normalizer):
    catalog = store.catalog()
    embedding = model.get_input_embeddings().weight
    columns = compress_dimension(embedding.shape[1]).to(embedding.device)
    codebook = F.normalize(embedding.index_select(1, columns).float(), dim=-1)
    groups = ('oracle', 'self_selected')
    modes = ('existing', 'joint', 'mean_raw', 'mean_calibrated', 'focused_raw', 'focused_calibrated')
    totals = {kind: {mode: Counter() for mode in modes} for kind in groups}
    values = [row[-1] for row in RECORDS]
    rows = []
    for index, ((track, label, question), scores, candidates, time_text) in enumerate(zip(HELD, old, names, inquiry, strict=True)):
        if any(value in question for value in values):
            raise ValueError('Held question contains a record value')
        expected = record_at(track, label)
        owner, field, _ = TIMELINES[track]
        gold_address = MemoryAddress(owner, FIELD_IDS[field])
        joint = scores[-1][0]
        rule = parse_lookup_scope(question)
        evidence = spoken_scope_evidence(time_text)
        existing = rule if rule is not None else normalizer([evidence])[0] if evidence else None
        queries = {'existing': existing, 'joint': plan.VERSIONS[joint]}
        for mode, predictions in temporal.items():
            guess, weight = predictions[index]
            raw = plan.VERSIONS[guess] if guess != UNCLEAR else None
            queries[f'{mode}_raw'] = raw
            queries[f'{mode}_calibrated'] = raw if weight >= readings[mode] else None
        (action, action_conf), (person, person_conf), (field_index, field_conf), _ = scores
        permitted = action == plan.READ and person < len(candidates) and min(action_conf, person_conf, field_conf) >= old_cut
        address = MemoryAddress(candidates[person], plan.FIELDS_ORDER[field_index]) if permitted else None
        cached = {}
        details = {}
        for kind, chosen in zip(groups, (gold_address, address), strict=True):
            details[kind] = result = {}
            for mode, query in queries.items():
                valid = chosen is not None and query is not None
                if kind == 'self_selected' and mode == 'joint':
                    valid &= scores[-1][1] >= old_cut
                record, _ = resolve_address(catalog, chosen, query, date.today(), retain_same_day=True) if valid else (None, '')
                answer = None
                if record is not None:
                    if record not in cached:
                        try:
                            _, response = reply_to_route(model, tokenizer, store, codebook, reader, question, [record],
                                                         labels=[plan.FIELDS[chosen.field]])
                            cached[record] = response['answer']
                        except ValueError as error:
                            cached[record] = str(error)
                    answer = cached[record]
                correct = record == expected and expected is not None and [x for x in values if x in (answer or '')] == [values[record - 1]]
                scope = query == (None if label == UNCLEAR else plan.VERSIONS[label])
                totals[kind][mode].update(scope=scope, exact_read=correct, wrong_read=record is not None and record != expected,
                                          no_read=record is None, valid_scope_read=correct and scope)
                result[mode] = dict(version=None if query is None else [query.kind, query.coordinate],
                                    number=record, answer=answer, correct=correct)
        rows.append(dict(question=question, expected=expected, scope=label, candidates=candidates,
                         route_allowed=permitted, guessed_temporal=[temporal[mode][index] for mode in temporal], arms=details))
    return rows, totals


def main(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    train = examples(TEMPORAL_TRAIN, UNCLEAR_TRAIN, TRAIN_VIEWS)
    cal = examples(TEMPORAL_CAL, UNCLEAR_CAL, CAL_VIEWS)
    all_names = {owner for owner, _, _ in TIMELINES}
    inputs = [plan.anonymize(question, all_names)[0].rsplit('\n', 1)[0] for _, _, question in HELD]
    if {row[0] for row in train} & ({row[0] for row in cal} | set(inputs)):
        raise ValueError('Temporal held-out examples overlap with training')
    old_report = json.loads((args.previous / 'result.json').read_text(encoding='utf-8'))
    old_head = args.previous / 'public.pt'
    if digest(old_head) != old_report['results']['public']['probe_sha256']:
        raise ValueError('Expected the frozen public-negative query planner')
    excluded = set()
    corpus_sample(args.corpus, 'train', excluded)
    public = corpus_sample(args.corpus, 'test', excluded)
    written, _ = replay((args.output / 'profile').resolve(), [f'/记 {p} | {f} | {v}' for p, f, v in RECORDS], args.recurrent)
    if any((row['kind'], row.get('record')) != ('saved', n) for n, row in enumerate(written, 1)):
        raise ValueError(f'Record staging failed: {written}')
    database = args.output / 'profile' / 'versions.sqlite3'
    reader_file, normalizer_file, _, binding = artifacts(args.reader, args.normalizer)
    bound = {p: digest(p) for p in (database, old_head, reader_file, normalizer_file, Path(__file__), args.recurrent[0],
                                   *(folder / 'attention.pt' for folder in args.recurrent[1:]))}
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    with local_model(args.recurrent) as model, VersionStore(database, binding) as store:
        extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=plan.DEPTH)
        train_features = extracted(extract, train)
        cal_features = extracted(extract, cal)
        test_features = extracted(extract, [(text, None, None) for text in inputs])
        trained, temporal, cutoffs = {}, {}, {}
        for mode, focused in (('mean', False), ('focused', True)):
            head = trained[mode] = fit(*train_features, train, focus=focused)
            cutoffs[mode] = calibrate(prediction(head, cal_features), cal)
            temporal[mode] = prediction(head, test_features)
        old = plan.QueryPlanProbe(model.config.hidden_size)
        old.load_state_dict(torch.load(old_head, map_location='cpu', weights_only=True))
        original, aliases = plan.vectors(extract, [row[2] for row in HELD], all_names, 'user-span')
        public_vectors, _ = plan.vectors(extract, [row[3] for row in public], all_names, 'user-span')
        public_predictions = [parts[0][0] for parts in plan.predict(old.eval(), public_vectors)]
        saved_rows = old_report['results']['public']['public_test_actions']
        saved = {(source, identifier): action for source, identifier, _, action in saved_rows}
        shared = [action == saved[(source, identifier)] for (source, identifier, *_), action in zip(public, public_predictions, strict=True)
                  if (source, identifier) in saved]
        state = torch.load(reader_file, map_location='cpu', weights_only=True)
        reader = VersionReplyActions(model.config.hidden_size, state['scaffold_tokens'], slots=2)
        reader.load_state_dict(state)
        final = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=len(model.model.layers))
        normalizer = TemporalProposals(final, model.config.hidden_size,
                                       torch.load(normalizer_file, map_location='cpu', weights_only=True)['final_layer'])
        rows, totals = evaluate(model, tokenizer, store, reader.eval().to(model.device), plan.predict(old, original),
                                old_report['results']['public']['threshold'], inputs, aliases, temporal, cutoffs, normalizer)
    if any(digest(path) != stable for path, stable in bound.items()):
        raise ValueError('A bound experiment file changed')
    weights = {}
    for name, head in trained.items():
        path = args.output / f'{name}.pt'
        atomic_save(head.state_dict(), path)
        weights[name] = digest(path)
    store_json(args.output / 'result.json', dict(rows=rows, metrics=totals, threshold=cutoffs, sha256=weights,
               training=len(train), calibration=len(cal), seconds=perf_counter() - started,
               public_normal=public_predictions.count(plan.CHAT), public_overlap=sum(shared), public_overlap_total=len(shared),
               sources={str(p): value for p, value in bound.items()}))
    print(totals, cutoffs, public_predictions.count(plan.CHAT), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous', type=Path)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reader', type=Path, default=Path('runs/spora-dual-version-reply-v1'))
    parser.add_argument('--normalizer', type=Path, default=Path('runs/spora-version-normalizer-v1'))
    parser.add_argument('--recurrent', required=True, nargs=3, type=Path)
    main(parser.parse_args())
