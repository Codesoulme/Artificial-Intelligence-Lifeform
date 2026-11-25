"""Recover spoken temporal scope without replacing exact timeline decisions."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_dual_version_reply import FIELDS, score
from scripts.audit_spora_dual_version_routing import HELD, SCOPED_HELD, TODAY, feature_rows, reply_to_route
from scripts.audit_spora_fast_weight_lexical_write import base, compress_dimension
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_multi_version_route import (Lookup, LookupHead, corroborated_version, explicit_queries,
                                               fallback_evidence, mentions, normalize_version, resolve_query)
from spora.research_version_store import VersionStore
from spora.research_versioned_memory_pointer import VersionReplyActions


SEED = 32623
SKIP, CURRENT, FIRST, PREVIOUS = Lookup
TRAIN = {
    SKIP: ('之前的是哪种', '旧的还有哪些', '说过的那回填的是什么', '某一次用过的是哪条',
           '随便挑一条答', '先不翻那条', '上回提过的那件事', '一共改了几次',
           '曾经用的那份还记得吗', '有些老记录写了几遍', '首稿或目前内容挑一份', '哪一稿合适'),
    CURRENT: ('手头执行的是哪条', '正在用的是哪份', '眼下作准的是哪一个', '现在仍然有效的内容',
              '最后一次更新留下的值', '仍旧在生效的那份', '这阵子实际用的哪款', '以这会儿为准',
              '后续继续沿用的是哪份', '当下还管用的内容', '手边刚确认的有效记录', '以执行中的那份为准'),
    FIRST: ('第一次写进去的是哪一条', '追溯到最开头那次的记录', '头一回交上来写的内容', '从没有改动起头那份',
            '刚建立档案那刻填的内容', '一开始落下的是哪一个', '第一次设定的是哪项', '打头的登记里有啥',
            '最早收录下来的内容', '开篇的那条原始记载', '起步建档用的哪份', '头一次做记录选的哪个'),
    PREVIOUS: ('最后一回变动之前用的那份', '刚刚被替下来的版本', '紧接在现值之前的那条', '最近更新前的内容',
               '上一轮刚退下去的是哪条', '紧挨着新值的上一条', '最新调整生效前用的是哪份', '前一轮用的那一份',
               '现用记录刚接替的旧站位', '挨着现在这条往前倒一格', '在最近一次变更前仍有效的哪项', '刚换走的那条'),
}
CAL = {
    SKIP: ('以前常写的是哪个', '旧稿里想挑一个', '你看用哪次的内容合适', '过去那版是啥',
           '翻翻不工作的那一条', '曾经有一笔是怎么样'),
    CURRENT: ('眼前仍执行的是哪版', '此间仍该据哪份', '当前还没失效的内容', '按今天继续走的那项',
              '离眼下最近仍管用的记录', '最新留下并仍使用的是哪份'),
    FIRST: ('有记录的头一笔是什么', '刚开始入档时写下的值', '翻到最前面第一次存下的那项',
            '最早一回初建时定的是哪个', '从零开始登记的那次', '先后顺序排第一的记录'),
    PREVIOUS: ('刚改值以前执行的那个', '贴着现用款往前一笔', '最近一次替换掉的旧值',
               '前次紧邻这次更新的值', '最近撤换前最后有效的一项', '与正在执行的值相接续的那条旧记录'),
}
FRESH = (
    ('oral', '请查闻韶宁的饮品存档起头落下的是啥；查闻绍宁的饮品现行版。', ('闻韶宁/drinks/0', '闻绍宁/drinks/0')),
    ('oral', '请核对闻韶宁的饮品最近一次改动刚替掉的哪份，见面地点当前版。', ('闻韶宁/drinks/1', '闻韶宁/places/1')),
    ('oral', '请查方吟晚的用品刚立档那阵填了啥；说说方吟晚的见面地点手头还认的那条。', ('方吟晚/items/0', '方吟晚/places/1')),
    ('oral', '查丁蔓庭的书目开头头一遭写的是什么；查贾宁杉的用品现在还算有效的那款。', ('丁蔓庭/books/0', '贾宁杉/items/1')),
    ('oral', '核对许照岚的用品挨着最后一版向前数一条；查闻韶宁的饮品眼目前采用的哪款。', ('许照岚/items/0', '闻韶宁/drinks/2')),
    ('oral', '请回忆叶昕卓的书目一上来登记了哪一本，见面地点还是按当前版。', ('叶昕卓/books/0', '叶昕卓/places/1')),
    ('oral', '请查贾宁杉的用品最开始收进档案的那件、见面地点当前版。', ('贾宁杉/items/0', '贾宁杉/places/1')),
    ('oral', '告诉我方吟晚的见面地点最后还在照办的是哪处。', ('方吟晚/places/1',)),
    ('oral', '查许照岚的饮品刚换新以前仍算有效的那款。', ('许照岚/drinks/0',)),
    ('oral', '想看看丁蔓庭的书目从建册那天最先录入了什么。', ('丁蔓庭/books/0',)),
    ('mixed', '请查闻韶宁的饮品在2026-10-09有效的版本，见面地点最开头整进清单的是哪处。', ('闻韶宁/drinks/1', '闻韶宁/places/0')),
    ('mixed', '请报闻绍宁的饮品第1版，闻韶宁的饮品往前挪一档是啥。', ('闻绍宁/drinks/0', '闻韶宁/drinks/1')),
    ('exclude', '请查方吟晚的用品头一回记进去的哪件，见面地点暂不必查。', ('方吟晚/items/0',)),
    ('ambiguous', '请查闻韶宁的饮品旧的记过哪款，见面地点当前版。', (None, '闻韶宁/places/1')),
    ('ambiguous', '说说方吟晚的用品以前那回登记了什么。', (None,)),
    ('ambiguous', '请查闻韶宁的饮品我记不清是哪一回了。', (None,)),
    ('ambiguous', '查叶昕卓的书目哪一回被记下来过。', (None,)),
    ('ambiguous', '查贾宁杉的用品有一回挑了啥，见面地点第2版。', (None, '贾宁杉/places/1')),
    ('ambiguous', '请核对闻韶宁的饮品初版还是当前版。', (None,)),
    ('missing', '查闻韶宁的书目头一遭收的是什么；见面地点当前版。', (None, '闻韶宁/places/1')),
    ('missing', '请查闻绍宁的饮品旧款刚调换下来的那个；核对闻韶宁的饮品第1版。', (None, '闻韶宁/drinks/0')),
    ('missing', '请查闻韶蔚的饮品档案最前头写的哪款，查闻绍宁的饮品当前版。', (None, '闻绍宁/drinks/0')),
    ('nonread', '把「请查闻韶宁的饮品刚换下的是哪款」改写一下。', ()),
    ('nonread', '麻烦点评「查叶昕卓的书目一上来写的啥」好不好懂。', ()),
    ('nonread', '先别查档，方吟晚的用品稍后再说。', ()),
    ('nonread', '怎么给草稿安排留档顺序？', ()),
)
THREE_VERSIONS = (
    ('first', '查闻韶宁的饮品最开始入档时留下哪款；查闻绍宁的饮品当前版。', ('闻韶宁/drinks/0', '闻绍宁/drinks/0')),
    ('previous', '查闻韶宁的饮品挨着最后定稿向前数一条；闻绍宁的饮品当前版。', ('闻韶宁/drinks/1', '闻绍宁/drinks/0')),
    ('current', '查闻韶宁的饮品手头还在照办的是哪条；查闻绍宁的饮品当前版。', ('闻韶宁/drinks/2', '闻绍宁/drinks/0')),
    ('first', '告诉我闻韶宁的饮品头一回存档填的啥。', ('闻韶宁/drinks/0',)),
    ('previous', '核对闻韶宁的饮品最后一次更换之前记的那款。', ('闻韶宁/drinks/1',)),
    ('ambiguous', '查闻韶宁的饮品目前这一版出现以前用的是哪款。', (None,)),
    ('ambiguous', '请查闻韶宁的饮品从前刚登记时写的是哪款。', (None,)),
    ('ambiguous', '闻韶宁的饮品开头提到的那次是什么？', (None,)),
    ('ambiguous', '查闻韶宁的饮品前年的某一回用的是哪款。', (None,)),
    ('nonread', '「闻韶宁的饮品挨着末版找往前一格」这句话帮我润色。', ()),
)


def examples(families, frames):
    rows = [(frame.format(phrase), int(label)) for label, phrases in families.items()
            for phrase in phrases for frame in frames]
    return [row[0] for row in rows], torch.tensor([row[1] for row in rows])


def choices(logits, confidence, margin):
    scores, labels = logits.softmax(-1).topk(2, dim=-1)
    accepted = (scores[:, 0] >= confidence) & (scores[:, 0] - scores[:, 1] >= margin)
    return labels[:, 0].masked_fill(~accepted, int(SKIP))


def fit(extract, width, steps):
    torch.manual_seed(SEED)
    texts, targets = examples(TRAIN, ('查该项{}。', '请核对该项{}？', '该项{}', '请回忆该项{}。'))
    cal_text, gold = examples(CAL, ('告诉我该项{}。', '说说该项{}吧。'))
    vectors, calibration = feature_rows(extract, texts), feature_rows(extract, cal_text)
    head = LookupHead(width)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    rng = torch.Generator().manual_seed(SEED)
    for _ in range(steps):
        batch = torch.randperm(len(targets), generator=rng)[:48]
        loss = F.cross_entropy(head(vectors[batch]), targets[batch])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        logits = head.eval()(calibration)
    def objective(parameters):
        predicted = choices(logits, *parameters)
        live = predicted != SKIP
        right = int((live & (predicted == gold)).sum())
        wrong = int((live & (predicted != gold)).sum())
        return right - 4 * wrong, -wrong, right, *parameters
    grid = [(confidence, margin) for confidence in (.55, .7, .8, .9, .95, .99)
            for margin in (.15, .3, .5)]
    cutoffs = max(grid, key=objective)
    return {'weights': head.state_dict(), 'cutoffs': list(cutoffs), 'calibration': list(objective(cutoffs)),
            'train_rows': len(texts), 'calibration_rows': len(cal_text)}


def resolved_route(catalog, candidates, queries):
    return [resolve_query(catalog, item, query, TODAY)[0]
            for item, query in zip(candidates, queries, strict=True) if query != SKIP]


@torch.inference_mode()
def evaluate(model, tokenizer, store, codebook, reader, routers, metadata, cases, *, guarded=False):
    catalog = store.catalog()
    enrolled = {address.name for address in catalog}
    rows, totals = [], {mode: Counter() for mode in ('explicit', *routers)}
    if guarded:
        totals['guarded_final'] = Counter()
    for group, question, expected in cases:
        if any(item['text'] in question for item in metadata.values()):
            raise ValueError('A stored value appears in the user question')
        candidates = mentions(question, enrolled, FIELDS)
        explicit = explicit_queries(question, candidates)
        evidence = {index: text for index, (item, query) in enumerate(zip(candidates, explicit, strict=True))
                    if query is None and (text := fallback_evidence(question, item)) is not None}
        routes = {'explicit': (resolved_route(catalog, candidates, explicit), [])}
        for mode, (extract, head, cutoffs) in routers.items():
            plan = explicit.copy()
            decisions = []
            accepted, checks = explicit.copy(), []
            if evidence:
                values = choices(head(feature_rows(extract, list(evidence.values()))), *cutoffs).tolist()
                for (index, text), label in zip(evidence.items(), values, strict=True):
                    proposal = Lookup(label)
                    plan[index] = normalize_version(proposal)
                    record = {'field': candidates[index].field, 'evidence': text, 'decision': proposal.name}
                    decisions.append(record)
                    if guarded and mode == 'final_layer':
                        verified = corroborated_version(proposal, text)
                        accepted[index] = verified
                        checks.append({**record, 'proposal': record['decision'],
                                       'decision': proposal.name if verified else SKIP.name})
            routes[mode] = (resolved_route(catalog, candidates, plan), decisions)
            if guarded and mode == 'final_layer':
                routes['guarded_final'] = (resolved_route(catalog, candidates, accepted), checks)
        known = [metadata[event]['number'] if event else None for event in expected]
        complete = bool(expected and None not in expected)
        wanted = [metadata[event]['text'] for event in expected if event]
        foils = [item['text'] for key, item in metadata.items() if key not in expected]
        generated, outputs = {}, {}
        for mode, (route, decisions) in routes.items():
            cache_key = tuple(route)
            if cache_key not in generated:
                generated[cache_key] = reply_to_route(model, tokenizer, store, codebook, reader, question, route)
            selected, answer = generated[cache_key]
            grade = score(answer, wanted, foils, tokenizer.eos_token_id)['all'] if complete else not selected
            counts = totals[mode]
            counts.update({'cases': 1, 'exact_route': route == known, 'complete': complete,
                           'correct_complete': complete and route == known and grade,
                           'wrong_read': bool(selected) and route != known,
                           'clarified': bool(route) and not selected,
                           'false_read_on_unresolved': not complete and bool(selected),
                           'learned_reads': sum(item['decision'] != SKIP.name for item in decisions)})
            outputs[mode] = {'route': route, 'selected': selected, 'answer_correct': grade, 'decisions': decisions, **answer}
        rows.append({'group': group, 'question': question, 'expected': known, 'outputs': outputs})
    return {'totals': {mode: dict(count) for mode, count in totals.items()}, 'rows': rows}


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    version_db = args.routing / 'versions.sqlite3'
    reader_path, reference_path = args.reader / 'head.pt', args.routing / 'result.json'
    scoped = args.routing / 'scoped-only.json'
    sources = {str(path): base.sha(path) for path in (version_db, reader_path, reference_path, scoped)}
    old = json.loads(reference_path.read_text(encoding='utf-8'))
    if (old['database_sha256'] != sources[str(version_db)]
            or old['sources'].get(str(reader_path)) != sources[str(reader_path)]
            or any(base.sha(Path(path)) != digest for path, digest in old['sources'].items())):
        raise ValueError('Frozen study and reader do not match the saved runs')
    binding = sha256(json.dumps(old['sources'], sort_keys=True).encode()).hexdigest()
    path = args.output / 'normalizers.pt'
    prior = json.loads((args.output / 'result.json').read_text(encoding='utf-8')) if args.verify else None
    if prior and (prior['sources'] != sources or prior['normalizer_sha256'] != base.sha(path)):
        raise ValueError('The replay artifacts differ from the fitted study')
    model = load_full_streamed()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    embedding = model.get_input_embeddings().weight.detach()
    columns = compress_dimension(embedding.shape[1]).to(embedding.device)
    codebook = F.normalize(embedding.index_select(1, columns).float(), dim=-1)
    state = torch.load(reader_path, map_location='cpu', weights_only=True)
    reader = VersionReplyActions(model.config.hidden_size, state['scaffold_tokens'], slots=2)
    reader.load_state_dict(state)
    reader.eval().to(model.device)
    depths = {'layer12': 12, 'final_layer': len(model.model.layers)}
    extractors = {mode: FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=depth)
                  for mode, depth in depths.items()}
    if prior:
        weights = torch.load(path, map_location='cpu', weights_only=True)
    else:
        weights = {}
        for mode, extractor in extractors.items():
            weights[mode] = fit(extractor, model.config.hidden_size, args.steps)
            print(mode, 'calibrated', weights[mode]['cutoffs'], weights[mode]['calibration'], flush=True)
        atomic_save(weights, path)
    routers = {}
    for mode, settings in weights.items():
        head = LookupHead(model.config.hidden_size)
        head.load_state_dict(settings['weights'])
        routers[mode] = (extractors[mode], head.eval(), settings['cutoffs'])
    with VersionStore(version_db, binding) as store:
        splits = {}
        for split, cases in (('original', HELD), ('explicit_regression', SCOPED_HELD), ('fresh_oral', FRESH)):
            splits[split] = evaluate(model, tokenizer, store, codebook, reader, routers, old['metadata'], cases,
                                     guarded=args.guarded)
            print(split, splits[split]['totals'], flush=True)
        contrast = None
        if args.guarded:
            contrast = evaluate(model, tokenizer, store, codebook, reader, routers, old['metadata'], THREE_VERSIONS,
                                guarded=True)
            print('three_versions', contrast['totals'], flush=True)
    baseline = json.loads(scoped.read_text(encoding='utf-8'))
    for split, previous in (('original', 'original'), ('explicit_regression', 'fresh')):
        for now, before in zip(splits[split]['rows'], baseline[previous], strict=True):
            active, saved = now['outputs']['explicit'], before['outputs']['scoped']
            if now['question'] != before['question'] or any(active[key] != saved[key] for key in ('route', 'token_ids', 'answer')):
                raise ValueError('The unchanged explicit reader did not reproduce the reference')
    if any(base.sha(Path(path)) != digest for path, digest in sources.items()):
        raise ValueError('A protected source changed during normalization')
    result = {'splits': splits, 'sources': sources, 'normalizer_sha256': base.sha(path), 'depths': depths,
              'fitting': {mode: {key: item for key, item in state.items() if key != 'weights'}
                          for mode, state in weights.items()}}
    if args.guarded:
        def unguarded(group):
            totals = {mode: count for mode, count in group['totals'].items() if mode != 'guarded_final'}
            rows = [{**row, 'outputs': {mode: info for mode, info in row['outputs'].items() if mode != 'guarded_final'}}
                    for row in group['rows']]
            return {'totals': totals, 'rows': rows}
        reproduced = {**result, 'splits': {split: unguarded(group) for split, group in splits.items()}}
    else:
        reproduced = result
    if prior and {key: prior[key] for key in reproduced} != reproduced:
        raise ValueError('Reload changed temporal decisions or reply output')
    run_stats = {'seconds': perf_counter() - started, 'peak_bytes': torch.cuda.max_memory_allocated()}
    store_json(args.output / ('reload.json' if args.verify else 'result.json'), {**reproduced, **run_stats})
    if args.guarded:
        store_json(args.output / 'guarded.json', {**result, 'three_versions': contrast, **run_stats})


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('reader', 'routing', 'output'):
        cli.add_argument(name, type=Path)
    cli.add_argument('--steps', type=int, default=140)
    cli.add_argument('--verify', action='store_true')
    cli.add_argument('--guarded', action='store_true')
    options = cli.parse_args()
    if options.steps < 1:
        cli.error('At least one fitting step is required')
    if options.guarded and not options.verify:
        cli.error('Verify the frozen original trial before comparing the additional evidence check')
    main(options)
