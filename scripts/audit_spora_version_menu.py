"""Audit a frozen typed-choice decoder before versioned memory reads."""

import argparse
from collections import Counter
from datetime import date
import gc
import json
from pathlib import Path
from time import perf_counter

import torch

from scripts.audit_spora_fast_weight_lexical_write import base
from scripts.audit_spora_version_evidence import calibration_key, frozen_choice_logits, version_slots
from scripts.audit_spora_version_roles import CAL, CAL_NAMES, contrast_examples
from scripts.audit_spora_version_scope import verify_answers
from scripts.audit_spora_version_timeline import learned_routes, SUBJECTS, TODAY
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.identity_fingerprints import IdentityFingerprints
from spora.research_version_menu import LETTERS, ORDERS, menu_evidence, menu_prompt, relative_menu_allowed
from spora.research_version_roles import ABSTAIN, apply_version_plan
from spora.research_version_scope import addressed_subject, choose_scoped_version
from spora.research_versioned_memory_pointer import VersionReplyActions


def fresh_cases():
    rows = []
    terms = ('这阵子还照着执行的那页', '开档时头一笔写进去的那页', '换成手头这份之前顶用的那页')
    for index, (_, name, field, _, _) in enumerate(SUBJECTS):
        tail = f'{name}的{field}登记的是哪项？'
        for left, right in ((0, 1), (0, 2)):
            pair = f'{name}-{left}-{right}'
            for target, dropped in ((left, right), (right, left)):
                question = f'拿{terms[target]}核对，{terms[dropped]}先搁在一旁。{tail}'
                rows.append({'subject': index, 'question': question, 'target': target, 'pair': pair})
        for text in (f'{terms[0]}和{terms[1]}都先撤下，备选还没有挑定。{tail}',
                     f'{terms[0]}划去。{tail}', f'{name}的{field}，2026-10-30有效的是哪项？',
                     f'只帮我润色「{name}的{field}现在是哪项」，不需要查资料。'):
            rows.append({'subject': index, 'question': text, 'target': ABSTAIN, 'pair': None})
    return rows


def plan_route(row, plan, schedules):
    if not row['menu']:
        return row['baseline']
    address = row['address']
    slot, _ = apply_version_plan(row['question'], schedules[address], TODAY, row['legacy'], plan)
    return None if slot is None else [address, slot]


def gated(row, evidence, agreement, margin, schedules):
    if evidence is None:
        return row['baseline']
    choice = evidence['option'] if evidence['agreement'] >= agreement and evidence['margin'] >= margin else ABSTAIN
    return plan_route(row, choice, schedules)


def choose_policy(rows, mode, schedules):
    choices = [(votes, margin) for votes in (2, 3, 4) for margin in (0., .05, .10, .20, .35)]
    return max(choices, key=lambda pair: (
        *calibration_key([gated(row, row['options'][mode], *pair, schedules) for row in rows], rows),
        pair[0], pair[1]))


def summary(rows):
    reports = {}
    for split in ('fresh', 'position', 'regression'):
        items = [row for row in rows if row['split'] == split]
        modes = {}
        pairs = {row['pair'] for row in items if row['pair'] is not None}
        for mode in items[0]['routes']:
            counts = Counter(total=len(items), pairs=len(pairs))
            for row in items:
                pick, gold = row['routes'][mode], row['gold']
                counts.update({'correct': pick == gold, 'positive': gold is not None, 'negative': gold is None,
                               'positive_correct': pick == gold and gold is not None,
                               'negative_read': gold is None and pick is not None,
                               'wrong_read': pick is not None and pick != gold})
            counts['paired_correct'] = sum(all(row['routes'][mode] == row['gold'] for row in items if row['pair'] == pair)
                                           for pair in pairs)
            modes[mode] = counts
        order = {}
        for style in ('plain', 'examples'):
            ballots = [row['options'][style] for row in items if row['options'][style] is not None]
            order[style] = {'scored': len(ballots), 'all_agree': sum(len(set(ballot['votes'])) == 1 for ballot in ballots),
                            'physical_positions': [sum(ballot['positions'].count(idx) for ballot in ballots)
                                                   for idx in range(len(LETTERS))]}
        reports[split] = {'modes': modes, 'orders': order}
    return reports


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / 'result.json'
    prior = json.loads(report_path.read_text(encoding='utf-8')) if args.verify else None
    reference = json.loads(args.evidence.read_text(encoding='utf-8'))
    bank_path, reader_path = args.timeline / 'versions.pt', args.reader / 'heads.pt'
    sources = {name: base.sha(Path(name)) for name in reference['sources']}
    if sources != reference['sources'] or any(str(path) not in sources for path in (bank_path, reader_path)):
        raise ValueError('The prior version experiment inputs changed')
    sources[str(args.evidence)] = base.sha(args.evidence)
    if prior is not None and prior['sources'] != sources:
        raise ValueError('Replay sources changed')
    old_scope = json.loads(args.scope.read_text(encoding='utf-8'))
    if base.sha(args.scope) != sources[str(args.scope)]:
        raise ValueError('The previous scope outputs changed')
    by_question = {row['question']: row for row in old_scope['rows']}
    fresh = fresh_cases()
    calibration = contrast_examples(CAL_NAMES, CAL)
    prior_questions = {row['question'] for row in reference['rows']}
    new_questions = {row['question'] for row in fresh}
    if len(new_questions) != len(fresh) or new_questions & (prior_questions | {row['question'] for row in calibration}):
        raise ValueError('Fresh questions overlap prior or calibration examples')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    old = learned_routes([row['question'] for row in fresh], tokenizer, sources)
    gc.collect()
    torch.cuda.empty_cache()
    print('Finished frozen legacy classifier', flush=True)
    model = load_full_streamed()
    bank = torch.load(bank_path, map_location='cpu', weights_only=True)
    names = IdentityFingerprints(len(SUBJECTS))
    fields, schedules = {}, {}
    for idx, (_, name, field, _, _) in enumerate(SUBJECTS):
        names.bind(idx, name)
        fields[idx] = (field,)
        schedules[idx] = tuple((date.fromisoformat(e), date.fromisoformat(o)) for e, o in bank['revisions'][name])
    rows = []
    for split, fixtures in (('calibration', calibration), ('fresh', fresh)):
        for index, fixture in enumerate(fixtures):
            subject = index % len(SUBJECTS) if split == 'calibration' else fixture['subject']
            slot = version_slots(schedules[subject])[fixture['target']]
            rows.append({'question': fixture['question'], 'split': split, 'pair': fixture['pair'],
                         'address': subject if split == 'calibration' else addressed_subject(fixture['question'], names, fields),
                         'legacy': old.get(fixture['question'], ABSTAIN), 'gold': None if slot is None else [subject, slot]})
    rows.extend({'question': row['question'], 'split': 'position' if row['split'] == 'fresh' else 'regression',
                 'pair': row['pair'], 'address': row['address'], 'legacy': row['legacy'], 'gold': row['gold']}
                for row in reference['rows'])
    for row in rows:
        subject = row['address']
        if subject is None:
            pick, reason = None, 'unresolved_address'
        else:
            pick, reason = choose_scoped_version(row['question'], schedules[subject], TODAY, row['legacy'])
        row['baseline'] = None if pick is None else [subject, pick]
        row['menu'] = relative_menu_allowed(row['question'], subject, reason)
        row['options'] = {'plain': None, 'examples': None}
    eligible = [row for row in rows if row['menu']]
    for mode in ('plain', 'examples'):
        print('Menu prompts', mode, len(eligible) * len(ORDERS), flush=True)
        prompts = [menu_prompt(row['question'], order, examples=mode == 'examples') for row in eligible for order in ORDERS]
        logits = frozen_choice_logits(model, tokenizer, prompts, LETTERS).cpu().tolist()
        for idx, row in enumerate(eligible):
            view = logits[idx * len(ORDERS):(idx + 1) * len(ORDERS)]
            row['options'][mode] = {**menu_evidence(view), 'logits': view}
    boundary = len(calibration)
    policies = {mode: choose_policy(rows[:boundary], mode, schedules) for mode in ('plain', 'examples')}
    rows = rows[boundary:]
    generated = []
    for row in rows:
        picks = {'baseline': row['baseline']}
        for mode, evidence in row['options'].items():
            picks[f'{mode}_first'] = plan_route(row, evidence['votes'][0] if evidence else None, schedules)
            picks[f'{mode}_mean'] = plan_route(row, evidence['option'] if evidence else None, schedules)
            picks[f'{mode}_gated'] = gated(row, evidence, *policies[mode], schedules)
        row['routes'] = picks
        previous = by_question.get(row['question']) if row['split'] == 'regression' else None
        if previous is not None and row['baseline'] != previous['routes']['scoped_auto']:
            raise ValueError('Regression baseline changed')
        if previous is not None and previous['routes']['scoped_auto'] == picks['examples_gated']:
            row['automatic_output'] = previous['automatic_output']
        else:
            generated.append(row)
    counts = summary(rows)
    print('Calibrated policies', policies, 'fresh', counts['fresh'], flush=True)
    state = torch.load(reader_path, map_location='cpu', weights_only=True)['atomic']
    reader = VersionReplyActions(model.config.hidden_size, state['scaffold_tokens'])
    reader.load_state_dict(state, strict=True)
    reader.eval().to('cuda')
    verify_answers(model, tokenizer, bank, reader, generated, route_mode='examples_gated')
    if any(base.sha(Path(name)) != digest for name, digest in sources.items()):
        raise ValueError('A protected artifact changed')
    generation = {split: {'correct_values': sum(row['automatic_output']['correct_value'] for row in rows if row['split'] == split),
                          'negative_mentions': sum(row['gold'] is None and bool(row['automatic_output']['mentioned_slots'])
                                                   for row in rows if row['split'] == split)}
                  for split in ('fresh', 'position', 'regression')}
    print(generation, flush=True)
    report = {'sources': sources, 'policies': {mode: list(rule) for mode, rule in policies.items()},
              'summary': counts, 'generation': generation, 'generated': len(generated), 'rows': rows}
    if prior is not None and any(prior[key] != value for key, value in report.items()):
        raise ValueError('Menu choices or outputs changed after reload')
    store_json(args.output / ('reload.json' if args.verify else 'result.json'),
               {**report, 'seconds': perf_counter() - started, 'peak_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evidence', 'scope', 'timeline', 'reader', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
