"""Test whether diverse Chinese supervision improves the native version selector."""

import argparse
from collections import Counter
from datetime import date
import gc
import json
from pathlib import Path
from time import perf_counter

import torch

from scripts.audit_spora_version_encoder import calibrate, classify, route, summarize
from scripts.audit_spora_version_roles import (CAL, CAL_NAMES, SEED, TRAIN, TRAIN_NAMES, base, clause_features,
                                              contrast_examples, fit, role_predictions)
from scripts.audit_spora_version_scope import verify_answers
from scripts.audit_spora_version_timeline import SUBJECTS, TODAY, learned_routes
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.identity_fingerprints import IdentityFingerprints
from spora.language_training import atomic_save
from spora.research_version_roles import ABSTAIN, VersionRoleHead, clause_spans, version_plan
from spora.research_version_scope import addressed_subject
from spora.research_version_timeline import effective_order
from spora.research_versioned_memory_pointer import VersionReplyActions


EXTRA_TRAIN = (
    ('给我看看{v}', '{v}暂且留着不核', ('眼下照办的那张', '建档之初存下的稿子', '上回交接时被撤下的单据')),
    ('{v}是这次要翻出来的', '这一问略过{v}', ('手头还作数的那页', '头回录入留下的那页', '换到手头这套之前刚撤下的一页')),
    ('我来查的是{v}', '{v}等回头再过目', ('如今还沿用的那份', '从空表起步写进去的那份', '最近一次改用新表时收走的那份')),
    ('麻烦调出{v}', '省去回头再问的{v}', ('尚在执行的那份', '刚开始记事时落的那一笔', '刚刚被新版取代的项目')),
    ('照着{v}答我', '{v}不用在这轮看', ('这会儿仍算数的登记', '档案创建当天的原始填法', '更新到如今这一项前紧挨的一项')),
    ('{v}我需要核实一下', '先把{v}收起来', ('今天依然采用的填写结果', '头一次落笔留下的字样', '由现用这份直接替掉的那份')),
)
EXTRA_CAL = (
    ('这回看{v}', '{v}且放在一旁', ('手上正运行的那一份', '从没改过之前的底本', '现在这条刚推出时替下的旧条目')),
    ('{v}得翻开看看', '本回撇开{v}', ('此刻正在沿用的方案', '最先写到纸面的方案', '换为眼前方案前刚停用的方案')),
)
HOLDOUT = (
    ('查给我{v}', '{v}过几天再说', ('眼前照着走的条目', '开卷那天留下的条目', '现用条目接班时刚退下的条目')),
    ('{v}才是我想对账的', '无需拿{v}来对账', ('到这会儿还没作废的一张', '开头空白处填成的那一张', '最近一轮交班前还顶用的一张')),
    ('我只调看{v}', '{v}且封存到改天', ('当面正在采用的字样', '表格建立时打头写成的字样', '最近接替现用项之前占着位置的字样')),
)
FREE = (
    ('开张就写进去的那行先别动，给我手上一直照办的那行。', 0, 'current-first'),
    ('手上一直照办的那行先别动，给我开张就写进去的那行。', 1, 'current-first'),
    ('把档案头一笔翻出来，接班前最后一次采用的那笔以后再谈。', 1, 'first-previous'),
    ('把接班前最后一次采用的那笔翻出来，档案头一笔以后再谈。', 2, 'first-previous'),
    ('我说的不是仍按它办的那张。提上来的是刚被它挤下去的那张。', 2, 'current-previous'),
    ('我说的不是刚被挤下去的那张。提上来的是仍按它办的那张。', 0, 'current-previous'),
    ('档案开头和这会儿执行的两份暂时都搁着，具体取哪份我再定。', ABSTAIN, None),
    ('我本想看仍在用的那张，先缓缓，我一会儿再定。', ABSTAIN, None),
    ('把刚撤掉的那张圈个记号就行，求你先别翻开。', ABSTAIN, None),
    ('用新一张之前紧跟着收起来的那张，取那一张。', 2, None),
)


def fixtures():
    families = (*TRAIN, *CAL, *EXTRA_TRAIN, *EXTRA_CAL, *HOLDOUT)
    phrases = [phrase for seek, exclude, cues in families for phrase in (seek, exclude, *cues)]
    if len(phrases) != len(set(phrases)):
        raise ValueError('A phrase appears in two authored families')
    original = contrast_examples(TRAIN_NAMES, TRAIN)
    old_questions = {row['question'] for row in original}
    expanded = original + [row for row in contrast_examples(TRAIN_NAMES, (*TRAIN, *EXTRA_TRAIN))
                           if row['question'] not in old_questions]
    calibration = contrast_examples(CAL_NAMES, (*CAL, *EXTRA_CAL))
    names = [(name, field) for _, name, field, _, _ in SUBJECTS]
    held = [{**row, 'split': 'families'} for row in contrast_examples(names, HOLDOUT)]
    free = [{'question': f'{text}{name}的{field}记的是什么？', 'target': target,
             'pair': f'{name}-{pair}' if pair is not None else None, 'split': 'free'}
            for name, field in names for text, target, pair in FREE]
    return original, expanded, calibration, held + free


def check_splits(*splits):
    seen = set()
    for rows in splits:
        questions = {row['question'] for row in rows}
        if len(questions) != len(rows) or seen.intersection(questions):
            raise ValueError('A question was duplicated across evaluation splits')
        seen.update(questions)


def check_labels(rows):
    for row in rows:
        if 'roles' not in row:
            continue
        plan = version_plan(row['roles'], row['versions'])
        target = ABSTAIN if plan is None else plan
        if len(clause_spans(row['question'])) != len(row['roles']) or target != row['target']:
            raise ValueError(f"Clause annotations and expected plan differ: {row['question']}")


def addresses():
    names = IdentityFingerprints(len(SUBJECTS))
    fields = {}
    for subject, (_, name, field, _, _) in enumerate(SUBJECTS):
        names.bind(subject, name)
        fields[subject] = (field,)
    return names, fields


def expected_subject(question):
    candidates = [i for i, (_, name, _, _, _) in enumerate(SUBJECTS) if name in question]
    if len(candidates) != 1:
        raise ValueError('An authored example needs an unambiguous expected address')
    return candidates[0]


def addressed_rows(fixtures_, schedules, legacy):
    names, fields = addresses()
    for item in fixtures_:
        question = item['question']
        subject = expected_subject(question)
        order = effective_order(schedules[subject], as_of=TODAY, known_by=TODAY)
        options = (order[-1], order[0], order[-2], None)
        slot = options[item['target']]
        yield {**item, 'address': addressed_subject(question, names, fields), 'legacy': legacy[question],
               'gold': None if slot is None else [subject, slot]}


def diagnostic(row, prediction):
    role, versions = prediction['roles'], prediction['versions']
    return {name: ABSTAIN if plan is None else plan for name, plan in (
        ('raw', prediction['plan']), ('role_given', version_plan(row['roles'], versions)),
        ('version_given', version_plan(role, row['versions'])),
        ('both_given', version_plan(row['roles'], row['versions'])),
    )}


def load_schedules(timeline):
    bank = torch.load(timeline, map_location='cpu', weights_only=True)
    schedules = [tuple((date.fromisoformat(e), date.fromisoformat(o)) for e, o in bank['revisions'][name])
                 for _, name, _, _, _ in SUBJECTS]
    return bank, schedules


def diagnose_address(args):
    path, weights = args.output / 'result.json', args.output / 'heads.pt'
    prior = json.loads(path.read_text(encoding='utf-8'))
    if base.sha(weights) != prior['weights_sha256'] or base.sha(args.timeline) != prior['sources'][str(args.timeline)]:
        raise ValueError('The selected checkpoint or timeline changed')
    state = torch.load(weights, map_location='cpu', weights_only=True)
    _, schedules = load_schedules(args.timeline)
    names, fields = addresses()
    rows = []
    suffix = '记的是什么？'
    for saved in prior['rows']:
        if saved['split'] != 'free':
            continue
        question = saved['question']
        if not question.endswith(suffix):
            raise ValueError('The controlled address substitution needs the authored suffix')
        subject = expected_subject(question)
        rewritten = question.removesuffix(suffix) + '写的是什么？'
        row = {**saved, 'address': subject, 'evaluated': {}, 'rewritten_address': addressed_subject(rewritten, names, fields)}
        row['evaluated']['baseline'] = route(row, schedules)
        for mode, checkpoint in state.items():
            proposal = row['proposals'][mode]
            row['evaluated'][mode] = route(row, schedules, classify(proposal, 0.))
            row['evaluated'][mode + '_calibrated'] = route(row, schedules, classify(proposal, checkpoint['threshold']))
        rows.append(row)
    report = {'source_sha256': base.sha(path), 'summary': summarize(rows),
              'same_subject_after_rewrite': sum(row['address'] == row['rewritten_address'] for row in rows), 'rows': rows}
    store_json(args.output / 'address-diagnostic.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'rows'}, ensure_ascii=False), flush=True)


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    prior_path = args.output / 'result.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8')) if args.verify else None
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    menu = json.loads(args.menu.read_text(encoding='utf-8'))
    earlier = json.loads(args.earlier.read_text(encoding='utf-8'))
    sources = {path: base.sha(Path(path)) for path in menu['sources']}
    if sources != menu['sources'] or earlier['sources'].get(str(args.menu)) != base.sha(args.menu):
        raise ValueError('Archived experiment inputs changed')
    for path in (args.menu, args.earlier, args.timeline, args.reader):
        if path in (args.timeline, args.reader) and sources.get(str(path)) != base.sha(path):
            raise ValueError('A versioned memory input changed')
        sources[str(path)] = base.sha(path)
    weights = args.output / 'heads.pt'
    if prior is not None and (sources != prior['sources'] or base.sha(weights) != prior['weights_sha256']):
        raise ValueError('Reload inputs changed')
    original, expanded, calibration, held = fixtures()
    archival = [{key: row[key] for key in ('split', 'question', 'pair', 'address', 'legacy', 'gold')}
                for row in earlier['rows']]
    check_splits(expanded, calibration, held, archival)
    check_labels(expanded + calibration + held)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    if prior is None:
        legacy = learned_routes([row['question'] for row in held], tokenizer, sources)
        gc.collect()
        torch.cuda.empty_cache()
    else:
        legacy = {row['question']: row['legacy'] for row in prior['rows'] if row['split'] in ('families', 'free')}
    print('Loaded legacy decisions for', len(legacy), 'new questions', flush=True)
    model = load_full_streamed()
    bank, schedules = load_schedules(args.timeline)
    rows = list(addressed_rows(held, schedules, legacy)) + archival
    for row in rows:
        row['evaluated'] = {'baseline': route(row, schedules)}
    for row, saved in zip(rows[len(held):], earlier['rows'], strict=True):
        if row['evaluated']['baseline'] != saved['evaluated']['baseline']:
            raise ValueError('The archived baseline changed')
    train_inputs = {'original': original, 'expanded': expanded}
    state = torch.load(weights, map_location='cpu', weights_only=True) if args.verify else {}

    def representations(samples):
        return clause_features(model, tokenizer, [row['question'] for row in samples])

    if not args.verify:
        train_features, train_mask = representations(expanded)
        cal_features = representations(calibration)
    test_features = representations(rows)
    oracle = {}
    for name, training in train_inputs.items():
        torch.manual_seed(SEED)
        head = VersionRoleHead(model.config.hidden_size)
        if prior is None:
            fit({'roles': head}, train_features[:len(training)], train_mask[:len(training)], training, steps=240)
            threshold = calibrate(role_predictions(head, *cal_features), calibration)
            state[name] = {'weights': head.state_dict(), 'threshold': threshold}
        else:
            head.load_state_dict(state[name]['weights'], strict=True)
            threshold = state[name]['threshold']
        head.eval()
        predictions = role_predictions(head, *test_features)
        counts = Counter()
        for row, proposal in zip(rows, predictions, strict=True):
            if 'roles' in row:
                counts.update({key: plan == row['target'] for key, plan in diagnostic(row, proposal).items()})
            row.setdefault('proposals', {})[name] = proposal
            row['evaluated'][name] = route(row, schedules, classify(proposal, 0.))
            row['evaluated'][name + '_calibrated'] = route(row, schedules, classify(proposal, threshold))
        oracle[name] = counts
    if not args.verify:
        atomic_save(state, weights)
    results = summarize(rows)
    print(json.dumps(results, ensure_ascii=False), flush=True)
    # Include the unresolved-address bucket and the first example per target for each known subject.
    generated = {}
    for row in rows[:len(held)]:
        if row['split'] == 'free':
            generated.setdefault((row['address'], row['target']), row)
    reader_state = torch.load(args.reader, map_location='cpu', weights_only=True)['atomic']
    reader = VersionReplyActions(model.config.hidden_size, reader_state['scaffold_tokens'])
    reader.load_state_dict(reader_state, strict=True)
    reader.eval().to('cuda')
    output = {}
    for mode in ('baseline', 'expanded_calibrated'):
        samples = [{**row, 'routes': row['evaluated']} for row in generated.values()]
        verify_answers(model, tokenizer, bank, reader, samples, route_mode=mode)
        output[mode] = {row['question']: row['automatic_output'] for row in samples}
    if any(base.sha(Path(path)) != digest for path, digest in sources.items()):
        raise ValueError('An original artifact changed')
    report = {'sources': sources, 'weights_sha256': base.sha(weights), 'training': {name: len(items)
              for name, items in train_inputs.items()}, 'calibration': len(calibration), 'oracle': oracle,
              'summary': results, 'rows': rows, 'generation_sample': output}
    if prior is not None and any(prior[key] != value for key, value in report.items()):
        raise ValueError('A result changed on reload')
    store_json(args.output / ('reload.json' if args.verify else 'result.json'),
               {**report, 'seconds': perf_counter() - started, 'peak_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('menu', 'earlier', 'timeline', 'reader', 'output'):
        parser.add_argument(name, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify', action='store_true')
    mode.add_argument('--diagnose-address', action='store_true')
    args = parser.parse_args()
    if args.diagnose_address:
        diagnose_address(args)
    else:
        main(args)
