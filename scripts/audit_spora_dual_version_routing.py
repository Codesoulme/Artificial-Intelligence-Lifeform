"""Compare clause and whole-question routing into a persisted dual-field reader."""

import argparse
from collections import Counter
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_dual_version_reply import EVENTS, FIELDS, reply, score
from scripts.audit_spora_fast_weight_lexical_write import SOURCE, base, compress_dimension
from scripts.audit_spora_version_coverage import write_fresh
from scripts.audit_spora_version_reply_actions import FACTS
from scripts.audit_spora_versioned_memory_generation import read_versions
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_multi_version_route import Lookup, LookupHead, explicit_queries, mentions, resolve, resolve_query
from spora.research_ttt_ntp import LowRankFastWeight
from spora.research_version_store import MemoryAddress, VersionStore
from spora.research_versioned_memory_pointer import VersionReplyActions


SEED = 32615
TODAY = date(2026, 10, 20)
S, C, FST, P = tuple(Lookup)
EXTRA = {'闻韶宁': {'drinks': ('薄荷红茶',)}, '闻绍宁': {'drinks': ('杏仁奶茶',), 'places': ('北岸书店',)}}
TRAIN = (
    ('请核对{a}当前是什么，{b}首版是什么。', (C, FST)),
    ('查{a}最先记的内容；查{b}上一版。', (FST, P)),
    ('看一下{a}的前一份记录，还有{b}当前有效的内容。', (P, C)),
    ('查查{a}如今记的是什么，也查{b}起初登记的值。', (C, FST)),
    ('请查{a}的最初一版，{b}的现用内容也报一下。', (FST, C)),
    ('给出{a}在改动前用的上一版；不要查询{b}。', (P, S)),
    ('先别查{a}；请告诉我{b}现在的值。', (S, C)),
    ('只查{a}刚被换下的那个版本，{b}仅作话题背景。', (P, S)),
    ('我在聊{a}这个栏目，查{b}头一份登记吧。', (S, FST)),
    ('这只是措辞练习，请改写「查{a}当前值、{b}第一版」这句话。', (S, S)),
    ('把「{a}初版是什么，{b}现行是什么」翻译成英语。', (S, S)),
    ('{a}与{b}的查询先推迟，今天只是讨论。', (S, S)),
)
CAL = (
    ('请检查{a}最早的一笔，{b}眼下还在用的是哪条？', (FST, C)),
    ('请查看{a}上一个有效稿。{b}先不查看。', (P, S)),
    ('{a}暂时不必翻；我只想查{b}第一回登记的。', (S, FST)),
    ('「{a}的当前版；{b}的上一版」这句话好懂吗？只点评。', (S, S)),
)
# Evaluation IDs refer to address and effective revision, never to answer text.
HELD = (
    ('mixed', '核对闻韶宁的饮品现在定的内容，还有见面地点最初那次记的。', ('闻韶宁/drinks/2', '闻韶宁/places/0')),
    ('mixed', '想看看方吟晚的用品第一笔登了什么，方吟晚的见面地点按现在这份查。', ('方吟晚/items/0', '方吟晚/places/1')),
    ('mixed', '查许照岚的饮品上个版本；请给许照岚的用品此刻生效的那份。', ('许照岚/drinks/0', '许照岚/items/1')),
    ('mixed', '查一下丁蔓庭的书目从前刚登记时写的，饮品则以目前的为准。', ('丁蔓庭/books/0', '丁蔓庭/drinks/1')),
    ('mixed', '请报叶昕卓的书目现行记录，和叶昕卓的见面地点最初一版。', ('叶昕卓/books/1', '叶昕卓/places/0')),
    ('mixed', '我想核对贾宁杉的用品仍有效的那款；查贾宁杉的见面地点此前紧挨现行的版本。', ('贾宁杉/items/1', '贾宁杉/places/0')),
    ('near_name', '请查闻绍宁的饮品当前那份，再查闻韶宁的饮品最早落笔那份。', ('闻绍宁/drinks/0', '闻韶宁/drinks/0')),
    ('near_name', '告诉我闻韶宁的饮品现行前一份，以及闻绍宁的饮品当前登记。', ('闻韶宁/drinks/1', '闻绍宁/drinks/0')),
    ('near_name', '请核对闻韶宁的饮品眼下有效的记录、闻绍宁的见面地点现用记录。', ('闻韶宁/drinks/2', '闻绍宁/places/0')),
    ('cross_person', '看一眼许照岚的饮品现在选的哪款；再把丁蔓庭的饮品最初的一款也告诉我。', ('许照岚/drinks/1', '丁蔓庭/drinks/0')),
    ('cross_person', '请查方吟晚的见面地点现行版，以及贾宁杉的见面地点当初写下的。', ('方吟晚/places/1', '贾宁杉/places/0')),
    ('single', '贾宁杉的用品先搁下，我只想核实见面地点目前登记在哪里。', ('贾宁杉/places/1',)),
    ('single', '请查许照岚的饮品最早写的是哪款，许照岚的用品暂不必查。', ('许照岚/drinks/0',)),
    ('single', '请回忆丁蔓庭的书目第一本登记的是哪一本？', ('丁蔓庭/books/0',)),
    ('single', '查到闻韶宁的饮品上一版就够了，另一个话题稍后再说。', ('闻韶宁/drinks/1',)),
    ('partial', '请查方吟晚的用品现行版、书目现行版也需要。', ('方吟晚/items/1', None)),
    ('partial', '请查闻韶凝的饮品当前值；闻绍宁的饮品现在登记什么？', (None, '闻绍宁/drinks/0')),
    ('partial', '核对闻绍宁的饮品上一版，并查闻韶宁的饮品目前这份。', (None, '闻韶宁/drinks/2')),
    ('unknown', '贾凝杉的见面地点第一版是什么？请查一下。', (None,)),
    ('nonread', '「查叶昕卓的书目现行、查贾宁杉的用品首版」听着自然吗？只评价说法。', ()),
    ('nonread', '今天先搁置许照岚的饮品和用品，两个都不用翻。', ()),
    ('nonread', '如果我写下「核对闻韶宁的饮品当前版，见面地点查首版」，可以帮我润色这句话吗？', ()),
    ('nonread', '怎么让刚沏好的茶凉得快一点？', ()),
    ('nonread', '我想起了方吟晚的用品这件事，先别查档，聊点别的。', ()),
)
SCOPED_HELD = (
    ('dual', '核对闻韶宁的饮品第一版；查闻绍宁的饮品当前版。', ('闻韶宁/drinks/0', '闻绍宁/drinks/0')),
    ('dual', '请查闻韶宁的饮品上一版、见面地点最新登记各是什么。', ('闻韶宁/drinks/1', '闻韶宁/places/1')),
    ('dual', '查丁蔓庭的书目首版；再报方吟晚的见面地点现在用哪份。', ('丁蔓庭/books/0', '方吟晚/places/1')),
    ('dual', '请核对许照岚的用品之前用的初版，贾宁杉的用品现在是哪件。', ('许照岚/items/0', '贾宁杉/items/1')),
    ('dual', '告诉我叶昕卓的见面地点现行版、书目上一版。', ('叶昕卓/places/1', '叶昕卓/books/0')),
    ('dual', '请查看贾宁杉的用品第二版；回忆许照岚的饮品最早的一版。', ('贾宁杉/items/1', '许照岚/drinks/0')),
    ('dated', '查闻韶宁的饮品在2026-10-07有效的记录，还有见面地点第2版。', ('闻韶宁/drinks/1', '闻韶宁/places/1')),
    ('single', '方吟晚的用品暂不必查；查方吟晚的见面地点首版。', ('方吟晚/places/0',)),
    ('single', '查到丁蔓庭的书目当前版就够了。', ('丁蔓庭/books/1',)),
    ('partial', '帮我查闻绍宁的见面地点上一版，同时查闻韶宁的饮品首版。', (None, '闻韶宁/drinks/0')),
    ('partial', '查许照岚的书目目前用的那份；说说许照岚的用品首版。', (None, '许照岚/items/0')),
    ('partial', '请查翁闻韶宁的饮品当前版，再查闻绍宁的饮品当前版。', (None, '闻绍宁/drinks/0')),
    ('unresolved', '核对闻韶宁的饮品开头落笔那一回，另外查见面地点当前版。', (None, '闻韶宁/places/1')),
    ('unresolved', '查叶昕卓的书目首版还是当前版，地点先不用查。', (None,)),
    ('dual', '告诉我许照岚的用品现在的前一份、贾宁杉的用品当前版。', ('许照岚/items/0', '贾宁杉/items/1')),
    ('unresolved', '核对闻韶宁的饮品现行记录以前用的那份；查闻绍宁的饮品当前版。', (None, '闻绍宁/drinks/0')),
    ('nonread', '把「核对闻韶宁的饮品初版和现行版」改写得客气一点。', ()),
    ('nonread', '请只点评「查贾宁杉的用品首版；查贾宁杉的见面地点现行版」。', ()),
    ('nonread', '丁蔓庭的书目和饮品都不用查，讲讲怎么整理阅读笔记。', ()),
    ('nonread', '怎样收纳书签才不会把它压弯？', ()),
)


def samples(templates, pairs):
    rows = []
    for name, other, fields in pairs:
        for form, labels in templates:
            question = form.format(a=f'{name}的{FIELDS[fields[0]]}', b=f'{other}的{FIELDS[fields[1]]}')
            entries = mentions(question, (name, other), FIELDS)
            if len(entries) != 2:
                raise ValueError('Training mentions do not cover both annotated requests')
            rows.extend((entry.views(question), int(label)) for entry, label in zip(entries, labels, strict=True))
    return rows


def feature_rows(extract, views):
    texts = [view.strip() + '\n核对该处版次：' for view in views]
    chunks = []
    for offset in range(0, len(texts), 16):
        group = texts[offset:offset + 16]
        vectors = extract(group)
        chunks.extend(vector[len(text) - 7:len(text)].float().mean(0) for text, vector in zip(group, vectors, strict=True))
    return torch.stack(chunks).clone()


def choose(logits, threshold):
    confidence, labels = logits.softmax(-1).max(-1)
    return [Lookup(int(label)) if float(weight) >= threshold else Lookup.SKIP
            for weight, label in zip(confidence, labels, strict=True)]


def fit(extract, head, train, cal, index, steps):
    train_x = feature_rows(extract, [views[index] for views, _ in train])
    cal_x = feature_rows(extract, [views[index] for views, _ in cal])
    targets = torch.tensor([label for _, label in train])
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    rng = torch.Generator().manual_seed(SEED)
    for _ in range(steps):
        batch = torch.randperm(len(targets), generator=rng)[:48]
        loss = F.cross_entropy(head(train_x[batch]), targets[batch])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        logits = head.eval()(cal_x)
    def quality(cut):
        pair = list(zip(choose(logits, cut), (label for _, label in cal), strict=True))
        wrong = sum(gold == S and pred != S for pred, gold in pair)
        right = sum(gold != S and pred == gold for pred, gold in pair)
        return right - 2 * wrong, -wrong, -cut
    return max((.35, .5, .65, .8, .9, .97), key=quality)


def build_bank(store, model, tokenizer, columns, source):
    extra = {}
    settings = torch.load(SOURCE / 'adapters.pt', map_location='cpu', weights_only=True)
    for field in ('drinks', 'places'):
        records = [(name, *events[field]) for name, events in EXTRA.items() if field in events]
        extra[field] = write_fresh(model, tokenizer, columns, layer=settings['layers'][0], eta=settings['eta'],
                                   ridge=settings['ridge'], events={field: records}, facts=FACTS)
    metadata = {}
    for name, fields in (EVENTS | {'闻绍宁': EXTRA['闻绍宁']}).items():
        for field, raw in fields.items():
            if name == '闻绍宁':
                bank = extra[field][name]
            elif name == '闻韶宁' and field == 'drinks':
                first, current = source[field][name]
                bank = [first, *extra[field][name], current]
                raw = (raw[0], *EXTRA[name][field], raw[1])
            else:
                bank = source[field][name]
            for index, ((keys, values), text) in enumerate(zip(bank, raw, strict=True)):
                event = f'{name}/{field}/{index}'
                day = date(2026, 10, 2 + 4 * index)
                number = store.append(event, MemoryAddress(name, field), day, day, LowRankFastWeight(keys, values))
                metadata[event] = {'number': number, 'text': text}
    return metadata


def reply_to_route(model, tokenizer, store, codebook, reader, question, route, *, history=(), labels=None):
    if not route:
        return [], reply(model, tokenizer, reader, question, (None, None), history=history)
    if None in route or len(route) > 2 or len(set(route)) != len(route):
        return [], {'answer': '我没能确定全部记录，请核对姓名、事项或版本。', 'token_ids': [], 'copied': []}
    bank = {}
    for number in route:
        state = store.read(number)
        bank[number] = [(state.keys, state.values)]
    decoded = read_versions(bank, codebook)
    values = tuple(decoded[number][0] for number in route)
    output = reply(model, tokenizer, reader, question, values + (None,) * (2 - len(values)),
                   require_coverage=labels is not None or len(values) == 2,
                   cover_present_slots=labels is not None, history=history)
    if labels is not None:
        copied = {item['slot']: item['at'] for item in output['copied']}
        if len(labels) != len(values) or set(copied) != set(range(len(values))):
            raise ValueError('这次没有完整读回选中的记忆，请重试。')
        fields = []
        for index, (label, value) in enumerate(zip(labels, values, strict=True)):
            start = copied[index]
            actual = output['token_ids'][start:start + len(value)]
            if actual != value.tolist():
                raise ValueError('读出的名称与记录不一致，请重试。')
            fields.append(f'{label}：{tokenizer.decode(actual)}')
        output['answer'] = '；'.join(fields) + '。'
    return route, output


@torch.inference_mode()
def evaluate(model, tokenizer, extract, store, codebook, reader, saved, metadata, *, cases=HELD):
    catalog = store.catalog()
    enrolled = {address.name for address in catalog}
    rows = []
    totals = {mode: Counter() for mode in saved}
    for group, question, expected in cases:
        if any(item['text'] in question for item in metadata.values()):
            raise ValueError('Evaluation question contains a stored value')
        candidates = mentions(question, enrolled, FIELDS)
        views = [candidate.views(question) for candidate in candidates]
        expected_numbers = [metadata[x]['number'] if x else None for x in expected]
        outputs = {}
        fallback = None
        for index, (mode, settings) in enumerate(saved.items()):
            decisions = []
            if mode == 'scoped':
                queries = explicit_queries(question, candidates)
                labels = [item.name if item == S else (item.kind if item is not None else 'UNRESOLVED') for item in queries]
                resolved = [(None, 'not_requested') if query == S else resolve_query(catalog, item, query, TODAY)
                            for item, query in zip(candidates, queries, strict=True)]
            else:
                head, threshold = settings
                selected = choose(head(feature_rows(extract, [view[index] for view in views])), threshold) if views else []
                labels = [item.name for item in selected]
                resolved = [resolve(catalog, item, label, TODAY) for item, label in zip(candidates, selected, strict=True)]
            for item, label, (number, reason) in zip(candidates, labels, resolved, strict=True):
                decisions.append({'owner': item.owner, 'field': item.field, 'request': label,
                                  'number': number, 'reason': reason})
            route = [item['number'] for item in decisions if item['request'] != Lookup.SKIP.name]
            if not route and fallback is not None:
                selected, output = [], fallback
            else:
                selected, output = reply_to_route(model, tokenizer, store, codebook, reader, question, route)
                if not route:
                    fallback = output
            wanted = [metadata[event]['text'] for event in expected if event]
            foils = [value['text'] for event, value in metadata.items() if event not in expected]
            complete = bool(expected and None not in expected)
            valid = (route == expected_numbers and score(output, wanted, foils, tokenizer.eos_token_id)['all']
                     if complete else not selected and not any(value['text'] in output['answer']
                                                               for value in metadata.values()))
            counts = totals[mode]
            counts.update({'cases': 1, 'exact_plan': route == expected_numbers,
                           'known_selected': len(selected), 'correct_complete': complete and valid,
                           'complete_cases': complete, 'nonread_leak': not complete and bool(selected),
                           'partial_clarified': None in expected and route == expected_numbers and not selected,
                           'prior_or_wrong_value': any(value in output['answer'] for value in foils)})
            outputs[mode] = {'decisions': decisions, 'route': route, 'selected': selected, 'valid': valid, **output}
        rows.append({'group': group, 'question': question, 'expected': expected_numbers, 'outputs': outputs})
    return rows, {mode: dict(counts) for mode, counts in totals.items()}


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    db, weight_path, result_path = (args.output / name for name in ('versions.sqlite3', 'lookup.pt', 'result.json'))
    versions, reader_path, prior_path = (args.reader / name for name in ('versions.pt', 'head.pt', 'result.json'))
    paths = (versions, reader_path, prior_path, SOURCE / 'adapters.pt', base.MODEL / 'model.safetensors',
             base.MODEL / 'tokenizer.json', base.MODEL / 'tokenizer_config.json')
    sources = {str(path): base.sha(path) for path in paths}
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    if (sources[str(versions)] != prior['versions_sha256'] or sources[str(reader_path)] != prior['head_sha256']
            or any(prior['sources'][str(path)] != sources[str(path)] for path in paths[3:])):
        raise ValueError('The frozen dual-reader artifacts changed')
    earlier = json.loads(result_path.read_text(encoding='utf-8')) if args.verify else None
    if earlier and (earlier['sources'] != sources or earlier['database_sha256'] != base.sha(db)
                    or earlier['lookup_sha256'] != base.sha(weight_path)):
        raise ValueError('A persisted input changed before restart')
    binding = sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    model = load_full_streamed()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=12)
    embedding = model.get_input_embeddings().weight.detach()
    columns = compress_dimension(embedding.shape[1]).to(embedding.device)
    codebook = F.normalize(embedding.index_select(1, columns).float(), dim=-1)
    reader_state = torch.load(reader_path, map_location='cpu', weights_only=True)
    reader = VersionReplyActions(model.config.hidden_size, reader_state['scaffold_tokens'], slots=2)
    reader.load_state_dict(reader_state)
    reader.eval().to(model.device)
    torch.manual_seed(SEED)
    heads = {mode: LookupHead(model.config.hidden_size) for mode in ('whole', 'clause')}
    if earlier:
        saved = torch.load(weight_path, map_location='cpu', weights_only=True)
        metadata = earlier['metadata']
    else:
        with VersionStore(db, binding, create=True) as store:
            metadata = build_bank(store, model, tokenizer, columns, torch.load(versions, map_location='cpu', weights_only=True))
        train = samples(TRAIN, [('童晓今', '辛明舟', ('drinks', 'places')), ('应乔安', '刁初晴', ('books', 'items')),
                                ('苗芳泽', '傅丹音', ('places', 'drinks')), ('谢庭欢', '霍未然', ('items', 'books')),
                                ('俞安露', '尤远亭', ('books', 'drinks')), ('关霁雯', '袁一棠', ('items', 'places'))])
        cal = samples(CAL, [('米晓珊', '唐默', ('drinks', 'books')), ('包云乔', '陆青野', ('places', 'items'))])
        saved = {}
        for index, (mode, head) in enumerate(heads.items()):
            threshold = fit(extract, head, train, cal, index, args.steps)
            saved[mode] = {'weights': head.state_dict(), 'threshold': threshold}
            print(mode, 'fitted', len(train), 'threshold', threshold, flush=True)
        atomic_save(saved, weight_path)
    candidates = {}
    for mode, head in heads.items():
        head.load_state_dict(saved[mode]['weights'])
        candidates[mode] = head.eval(), saved[mode]['threshold']
    digest = base.sha(db)
    if not args.scoped_only:
        with VersionStore(db, binding) as store:
            rows, totals = evaluate(model, tokenizer, extract, store, codebook, reader, candidates, metadata)
        if digest != base.sha(db) or any(base.sha(path) != sources[str(path)] for path in paths):
            raise ValueError('The read-only replay changed a protected input')
        result = {'metadata': metadata, 'sources': sources, 'database_sha256': digest, 'lookup_sha256': base.sha(weight_path),
                  'rows': rows, 'totals': totals, 'thresholds': {mode: value['threshold'] for mode, value in saved.items()}}
        if earlier and {key: earlier[key] for key in result} != result:
            raise ValueError('Restart changed routing decisions or generation')
        store_json(args.output / ('reload.json' if args.verify else 'result.json'),
                   {**result, 'seconds': perf_counter() - started, 'peak_bytes': torch.cuda.max_memory_allocated()})
        print(totals, flush=True)
    if args.scoped_probe or args.scoped_only:
        probe_start = perf_counter()
        with VersionStore(db, binding) as store:
            old_rows, old_totals = evaluate(model, tokenizer, extract, store, codebook, reader, {'scoped': None}, metadata)
            new_rows, new_totals = evaluate(model, tokenizer, extract, store, codebook, reader, {'scoped': None}, metadata,
                                           cases=SCOPED_HELD)
        if digest != base.sha(db) or any(base.sha(path) != sources[str(path)] for path in paths):
            raise ValueError('Scoped probe changed a protected input')
        path = args.output / ('scoped-only.json' if args.scoped_only else 'scoped-probe.json')
        store_json(path, {'original': old_rows, 'fresh': new_rows, 'original_totals': old_totals,
                          'fresh_totals': new_totals, 'database_sha256': digest, 'seconds': perf_counter() - probe_start})
        print('scoped original', old_totals, 'fresh', new_totals, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reader', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--steps', type=int, default=160)
    parser.add_argument('--verify', action='store_true')
    probes = parser.add_mutually_exclusive_group()
    probes.add_argument('--scoped-probe', action='store_true')
    probes.add_argument('--scoped-only', action='store_true')
    options = parser.parse_args()
    if options.steps < 1:
        parser.error('At least one update is required')
    if (options.scoped_probe or options.scoped_only) and not options.verify:
        parser.error('The scoped probe requires the frozen, saved trial')
    main(options)
