"""Route natural version requests to independently written neural matrices."""

import argparse
from collections import Counter
from datetime import date
import gc
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_qwen_compatible_natural_memory_20261121 import load_head
from scripts.audit_spora_fast_weight_lexical_write import base, compress_dimension, SOURCE
from scripts.audit_spora_version_coverage import write_fresh
from scripts.audit_spora_version_reply_actions import atomic_reply, FACTS
from scripts.audit_spora_versioned_memory_generation import read_versions
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.integrated_checkpoint import load_integrated_runtime
from spora.language_training import atomic_save, frozen_digest
from spora.research_temporal_intent import temporal_logits, time_selection
from spora.research_version_timeline import effective_order, parse_version_query, select_version
from spora.research_versioned_memory_pointer import VersionReplyActions


ANCHOR = Path('runs/spora-anchored-native-candidate-v1/bootstrap.pt')
TEMPORAL_SOURCE = Path('runs/qwen-compatible-gelu-layer17-v1/integrated.pt')
TEMPORAL = Path('runs/qwen-compatible-gelu-temporal-factorized-v1/factorized')
SELECTOR = Path('runs/spora-anchored-temporal-selector-v1/selector.pt')


def day(number):
    return date(2026, 10, number)


TODAY = day(26)
# Each version has an effective day and a first observed day. The assertions
# include a backfill, a same-date correction and an announced future change.
SUBJECTS = (
    ('drinks', '霍盈', '饮品', ('橘皮红茶', '芋泥豆奶', '桂花乌龙', '紫米米浆'), ((2, 2), (7, 7), (13, 13), (19, 19))),
    ('places', '范远清', '碰面地点', ('松风书屋', '白石码头', '江畔剧场', '鹿鸣园'), ((2, 2), (7, 7), (19, 19), (13, 23))),
    ('items', '瞿知然', '物品', ('藤纹茶盘', '帆布笔袋', '铜扣书夹', '方格便签'), ((2, 2), (7, 7), (13, 13), (13, 18))),
    ('books', '游敏佳', '书目', ('星海拾光', '山城旧信', '石阶回声', '晚潮来信'), ((3, 3), (9, 9), (16, 16), (28, 24))),
)


def cases():
    results = []
    for index, (_, name, field, _, revision_days) in enumerate(SUBJECTS):
        switch = day(revision_days[1][0]).isoformat()
        tail = f'{name}的{field}记的是什么？只说名字。'
        planned = (
            ('current', f'核对一下目前有效的记录，{tail}', (3, 2, 3, 2)[index]),
            ('ordinal', f'查首版的登记内容，{tail}', 0),
            ('ordinal', f'只翻第三份有效记录，{tail}', (2, 3, 3, 2)[index]),
            ('previous', f'上一版是怎么记的？{tail}', (2, 3, 1, 1)[index]),
            ('date', f'2026年10月14日当天有效的那份，{tail}', (2, 3, 3, 1)[index]),
            ('known_by', f'按截至2026-10-17已经收到的记录，查2026-10-14有效的那份，{tail}', (2, 1, 2, 1)[index]),
            ('change', f'{switch}那回更正生效前，{tail}', 0),
            ('change', f'{switch}那回更正生效时，{tail}', 1),
            ('no_match', f'2026-10-01有效的记录里，{tail}', None),
            ('no_match', f'2026-10-28那天的记录里，{tail}', None),
            ('no_match', f'第五版里，{tail}', None),
            ('language', f'就眼下那张单据来核对，{tail}', (3, 2, 3, 2)[index]),
            ('language', f'别用那份过时的，拿手头还算数的那张核对。{tail}', (3, 2, 3, 2)[index]),
        )
        results.extend({'category': kind, 'name': name, 'question': question, 'expected': slot}
                       for kind, question, slot in planned)
    name, field = SUBJECTS[0][1:3]
    prompt = f'2026-10-10有效的那份记录中，{name}的{field}是什么？只说名字。'
    results.extend(({'category': 'swap_metadata', 'name': name, 'question': prompt, 'expected': 1},
                    {'category': 'swap_metadata', 'name': name, 'question': prompt, 'expected': 0, 'alternate': True}))
    return results


@torch.inference_mode()
def learned_routes(questions, tokenizer, hashes):
    temporal, temporal_sha = load_head(TEMPORAL, 'temporal.pt', 'temporal_sha256', hashes[str(TEMPORAL_SOURCE)])
    manifest = json.loads(SELECTOR.with_name('result.json').read_text(encoding='utf-8'))
    selector = torch.load(SELECTOR, map_location='cpu', weights_only=True)
    if (manifest['selector_sha256'] != hashes[str(SELECTOR)] or selector['temporal_sha256'] != temporal_sha
            or selector['checkpoint_sha256'] != hashes[str(ANCHOR)]):
        raise ValueError('Version selector does not match the original detector')
    temporal['version_selector'] = {**selector, 'char_width': selector.get('char_width', 0)}
    runtime = load_integrated_runtime(ANCHOR, load_full_streamed(), base.MODEL, disposable_base=True)
    before = frozen_digest(runtime.organism, runtime.state)
    features, masks = extract(runtime.decoder.base.eval(), tokenizer, questions, batch_size=2,
                              hidden_indices=tuple(temporal['hidden_indices']), last_logit_only=True)
    probabilities = temporal_logits(temporal, features, masks, texts=questions).softmax(-1)
    decisions = time_selection(probabilities, temporal['threshold'], factorized=True).tolist()
    if frozen_digest(runtime.organism, runtime.state) != before:
        raise ValueError('Classification modified the source individual')
    return dict(zip(questions, decisions, strict=True))


def choose_versions(case, revisions, label):
    if case.get('alternate'):
        revisions = (revisions[0], (day(12), revisions[1][1]), *revisions[2:])
    current = effective_order(revisions, as_of=TODAY, known_by=TODAY)
    # Legacy history labels address the first two observations, with current
    # resolved against present effective time. They never see the query dates.
    legacy = (current[-1], 0, 1, None)[label]
    calendar = select_version(revisions, parse_version_query(case['question']), TODAY)
    return {'latest': current[-1], 'classifier': legacy, 'timeline': calendar, 'provided': case['expected']}


def evaluate(model, tokenizer, head, states, schedules, labeled, all_cases):
    facts = {name: values for _, name, _, values, _ in SUBJECTS}
    empty = torch.empty(0, dtype=torch.long, device='cuda')
    for name, options in states.items():
        if tuple(tokenizer.decode(tokens.tolist()) for tokens in options) != facts[name]:
            raise ValueError('A version payload failed before routing')
    memo, rows = {}, []
    for case in all_cases:
        name, question, gold = (case[key] for key in ('name', 'question', 'expected'))
        selected = choose_versions(case, schedules[name], labeled[question])
        decisions = {}
        for mode, slot in selected.items():
            key = (question, name, slot)
            if key not in memo:
                memo[key] = atomic_reply(model, tokenizer, head, question, empty if slot is None else states[name][slot])
            reply = memo[key]
            hits = [index for index, value in enumerate(facts[name]) if value in reply['answer']]
            ended = bool(reply['token_ids'] and reply['token_ids'][-1] == tokenizer.eos_token_id)
            decisions[mode] = {'slot': slot, 'reply': reply, 'route_correct': slot == gold,
                               'value_correct': gold is not None and ended and hits == [gold],
                               'wrong_value': any(index != gold for index in hits),
                               'cooccurrence': gold in hits and len(hits) > 1,
                               'read_unknown': gold is None and slot is not None}
        rows.append({**case, 'versions': list(facts[name]), 'classifier_label': labeled[question], 'decisions': decisions})
        if len(rows) % 8 == 0:
            print('generated', len(rows), 'of', len(all_cases), flush=True)
    return rows


def summarize(rows):
    summary = {}
    for row in rows:
        for group in ('all', row['category']):
            totals = summary.setdefault(group, {})
            for mode, entry in row['decisions'].items():
                counts = totals.setdefault(mode, Counter())
                counts.update({'cases': 1, 'value_cases': row['expected'] is not None,
                               'unknown_cases': row['expected'] is None})
                counts.update({key: entry[key] for key in ('route_correct', 'value_correct', 'wrong_value',
                                                            'cooccurrence', 'read_unknown')})
    return summary


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    source_paths = (base.MODEL / 'model.safetensors', base.MODEL / 'tokenizer.json', base.MODEL / 'tokenizer_config.json',
                    ANCHOR, TEMPORAL_SOURCE, TEMPORAL / 'temporal.pt', TEMPORAL / 'result.json', SELECTOR,
                    SELECTOR.with_name('result.json'), args.reply / 'heads.pt', args.reply / 'result.json', SOURCE / 'adapters.pt')
    hashes = {str(p): base.sha(p) for p in source_paths}
    prior = json.loads((args.reply / 'result.json').read_text(encoding='utf-8'))
    if (hashes[str(args.reply / 'heads.pt')] != prior['heads_sha256']
            or hashes[str(source_paths[0])] != base.WEIGHT_SHA
            or any(hashes[str(p)] != prior['sources'][str(p)] for p in (source_paths[1], source_paths[2], source_paths[-1]))):
        raise ValueError('The frozen model or readout does not match the trained source')
    report_path, bank_path = (args.output / p for p in ('result.json', 'versions.pt'))
    prior_report = json.loads(report_path.read_text(encoding='utf-8')) if args.verify else None
    if prior_report is not None and (hashes != prior_report['sources'] or base.sha(bank_path) != prior_report['bank_sha256']):
        raise ValueError('Timeline or source artifacts changed')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    test = cases()
    questions = list(dict.fromkeys(row['question'] for row in test))
    labeled = learned_routes(questions, tokenizer, hashes)
    gc.collect()
    torch.cuda.empty_cache()
    print('Existing version classifier finished', flush=True)
    model = load_full_streamed()
    embedding = model.get_input_embeddings().weight.detach()
    columns = compress_dimension(embedding.shape[1]).to(embedding.device)
    codebook = F.normalize(embedding.index_select(1, columns).float(), dim=-1)
    if not args.verify:
        settings = torch.load(source_paths[-1], map_location='cpu', weights_only=True)
        authored = {domain: ((name, *values),) for domain, name, _, values, _ in SUBJECTS}
        timeline = {name: [(day(e).isoformat(), day(o).isoformat()) for e, o in revisions]
                    for _, name, _, _, revisions in SUBJECTS}
        atomic_save({'factors': write_fresh(model, tokenizer, columns, layer=settings['layers'][0],
                                           eta=settings['eta'], ridge=settings['ridge'], events=authored, facts=FACTS),
                     'revisions': timeline}, bank_path)
    bank = torch.load(bank_path, map_location='cpu', weights_only=True)
    values = read_versions(bank['factors'], codebook)
    del codebook
    schedules = {name: tuple((date.fromisoformat(e), date.fromisoformat(o)) for e, o in sequence)
                 for name, sequence in bank['revisions'].items()}
    weights = torch.load(args.reply / 'heads.pt', map_location='cpu', weights_only=True)['atomic']
    head = VersionReplyActions(model.config.hidden_size, weights['scaffold_tokens'])
    head.load_state_dict(weights, strict=True)
    head.eval().to('cuda')
    rows = evaluate(model, tokenizer, head, values, schedules, labeled, test)
    if any(base.sha(p) != hashes[str(p)] for p in source_paths):
        raise ValueError('A protected source changed')
    report = {'sources': hashes, 'bank_sha256': base.sha(bank_path), 'today': TODAY.isoformat(),
              'summary': summarize(rows), 'rows': rows}
    if prior_report is not None and any(prior_report[key] != value for key, value in report.items()):
        raise ValueError('Timeline replay changed routing or generation')
    print(report['summary']['all'], flush=True)
    store_json(args.output / ('reload.json' if args.verify else 'result.json'),
               {**report, 'seconds': perf_counter() - started, 'peak_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reply', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
