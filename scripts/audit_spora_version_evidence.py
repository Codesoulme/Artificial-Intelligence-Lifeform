"""Compare clause-end evidence and a frozen verifier before memory reads."""

import argparse
from collections import Counter
from datetime import date
import gc
import json
from pathlib import Path
from time import perf_counter

import torch

from scripts.audit_spora_fast_weight_lexical_write import base, tokens
from scripts.audit_spora_version_roles import (CAL, CAL_NAMES, SEED, TRAIN, TRAIN_NAMES, clause_features, compatible_plan,
                                               contrast_examples, fit, make_question, role_predictions)
from scripts.audit_spora_version_scope import verify_answers
from scripts.audit_spora_version_timeline import learned_routes, SUBJECTS, TODAY
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.identity_fingerprints import IdentityFingerprints
from spora.language_training import atomic_save
from spora.research_version_roles import (ABSTAIN, CURRENT, EXCLUDE, FIRST, PREVIOUS, SEEK, UNSPECIFIED,
                                          VersionRoleHead, apply_version_plan)
from spora.research_version_scope import _DATE_HINT, _ORDINAL_HINT, addressed_subject
from spora.research_version_timeline import effective_order
from spora.research_versioned_memory_pointer import VersionReplyActions


POSITION = (
    ('查阅{v}', '划去{v}'),
    ('{v}查阅', '{v}划去'),
)
LABELS = ('目前有效的那一版', '最早生效的那一版', '目前有效版之前的紧邻一版')
TAIL_CHARS = 4


def held_cases():
    rows = []
    for name_index, (_, name, field, _, _) in enumerate(SUBJECTS):
        ask = (f'{name}的{field}登记的是哪项', SEEK, UNSPECIFIED)
        cues = ('手上正在沿用的那张', '刚起草时留下的那页', '现用版本前面紧接的那份')
        for position, (seek, exclude) in enumerate(POSITION):
            for left, right in ((CURRENT, FIRST), (CURRENT, PREVIOUS), (FIRST, PREVIOUS)):
                key = f'{name}-{position}-{left}-{right}'
                for positive, negative in ((left, right), (right, left)):
                    row = make_question([(seek.format(v=cues[positive]), SEEK, positive),
                                         (exclude.format(v=cues[negative]), EXCLUDE, negative), ask], positive, key)
                    rows.append({**row, 'subject': name_index, 'position': 'last' if position else 'first'})
            row = make_question([(exclude.format(v=cues[CURRENT]), EXCLUDE, CURRENT), ask], ABSTAIN)
            rows.append({**row, 'subject': name_index, 'position': 'last' if position else 'first'})
        for text in (f'照2026/10/29的日期查。{name}的{field}是什么？',
                     f'不要读取{name}的{field}，只给一份空表。'):
            rows.append({'question': text, 'target': ABSTAIN, 'subject': name_index, 'pair': None, 'position': 'control'})
    return rows


def version_slots(schedule):
    order = effective_order(schedule, as_of=TODAY, known_by=TODAY)
    return order[-1], order[0], order[-2], None


def proposal(row, decision, threshold, schedules, *, baseline=False):
    subject = row['address']
    if subject is None:
        return None
    plan = None if baseline else compatible_plan(decision, threshold)
    selected, _ = apply_version_plan(row['question'], schedules[subject], TODAY, row['legacy'], plan)
    return None if selected is None else [subject, selected]


def calibration_key(choices, rows):
    paired = list(zip(choices, rows, strict=True))
    return (-sum(row['gold'] is None and choice is not None for choice, row in paired),
            -sum(row['gold'] is not None and choice is not None and choice != row['gold'] for choice, row in paired),
            sum(choice == row['gold'] and choice is not None for choice, row in paired))


def choose_thresholds(proposals, rows, schedules):
    thresholds = {}
    for name, decisions in proposals.items():
        def key(threshold):
            selected = [proposal(row, decision, threshold, schedules) for row, decision in zip(rows, decisions, strict=True)]
            return (*calibration_key(selected, rows), threshold)
        thresholds[name] = max((.40, .50, .60, .70, .80, .90), key=key)
    return thresholds


@torch.inference_mode()
def frozen_choice_logits(model, tokenizer, questions, labels):
    words = [tokens(tokenizer, word) for word in labels]
    if any(len(encoded) != 1 for encoded in words) or len({encoded[0] for encoded in words}) != len(words):
        raise ValueError('Choice labels need distinct single tokens')
    label_ids = [encoded[0] for encoded in words]
    prompts = []
    for question in questions:
        chat = tokenizer.apply_chat_template([{'role': 'user', 'content': question}], tokenize=False,
                                             add_generation_prompt=True)
        if not chat.endswith('<think>\n'):
            raise ValueError('Unexpected DeepSeek template')
        prompts.append(chat + '</think>\n\n')
    scores = []
    for offset in range(0, len(prompts), 4):
        inputs = tokenizer(prompts[offset:offset + 4], return_tensors='pt', padding=True,
                           padding_side='left', add_special_tokens=False).to(model.device)
        logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, label_ids].float()
        scores.append(logits)
        if (offset + 4) % 64 == 0:
            print('Scored prompts', offset + 4, 'of', len(prompts), flush=True)
    return torch.cat(scores) if scores else torch.empty(0, len(label_ids), device=model.device)


@torch.inference_mode()
def semantic_scores(model, tokenizer, probes):
    questions = [f'{test}\n原话：{question}\n候选：{label}\n只答是或否。'
                 for question, label in probes for test in
                 ('这句话是否要求读取候选版本？', '候选版本是否属于被排除或未被请求的内容？')]
    scores = frozen_choice_logits(model, tokenizer, questions, ('是', '否')).softmax(-1).cpu().tolist()
    return {probe: min(positive[0], negative[1]) for probe, positive, negative in
            zip(probes, scores[::2], scores[1::2], strict=True)}


def verification_probe(row, chosen, schedules):
    if chosen is None or _DATE_HINT.search(row['question']) or _ORDINAL_HINT.search(row['question']):
        return None
    subject, slot = chosen
    version = version_slots(schedules[subject]).index(slot)
    return row['question'], LABELS[version]


def describe(rows):
    groups = {}
    for row in rows:
        for group in (row['split'], f"position_{row['position']}"):
            modes = groups.setdefault(group, {})
            for mode, route in row['routes'].items():
                counts = modes.setdefault(mode, Counter())
                gold = row['gold']
                counts.update({'total': 1, 'correct': route == gold, 'positive': gold is not None,
                               'negative': gold is None, 'wrong_read': route is not None and route != gold,
                               'negative_read': gold is None and route is not None})
    for mode, counts in groups['fresh'].items():
        paired = {row['pair'] for row in rows if row['split'] == 'fresh' and row['pair'] is not None}
        counts['pairs'] = len(paired)
        counts['pair_correct'] = sum(all(row['routes'][mode] == row['gold'] for row in rows if row['pair'] == pair)
                                     for pair in paired)
    return groups


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    weights_path, report_path = args.output / 'heads.pt', args.output / 'result.json'
    prior = json.loads(report_path.read_text(encoding='utf-8')) if args.verify else None
    reference = json.loads(args.scope.read_text(encoding='utf-8'))
    bank_path, reader_path = args.timeline / 'versions.pt', args.reader / 'heads.pt'
    sources = {name: base.sha(Path(name)) for name in reference['sources']}
    if sources != reference['sources'] or any(str(path) not in sources for path in (bank_path, reader_path)):
        raise ValueError('The previous scope inputs changed')
    sources[str(args.scope)] = base.sha(args.scope)
    if prior is not None and (prior['sources'] != sources or prior['heads_sha256'] != base.sha(weights_path)):
        raise ValueError('Replay inputs changed')
    regressions = {row['question']: row for row in reference['rows']}
    fresh, training, calibration = held_cases(), contrast_examples(TRAIN_NAMES, TRAIN), contrast_examples(CAL_NAMES, CAL)
    splits = [set(row['question'] for row in collection) for collection in (training, calibration, fresh)]
    if any(left & right for index, left in enumerate(splits) for right in splits[index + 1:]) or splits[-1] & regressions.keys():
        raise ValueError('Question splits overlap')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    old = learned_routes([row['question'] for row in (*calibration, *fresh)], tokenizer, sources)
    gc.collect()
    torch.cuda.empty_cache()
    print('Finished frozen baseline', flush=True)
    model = load_full_streamed()
    heads = {}
    for mode in ('mean', 'tail'):
        torch.manual_seed(SEED)
        heads[mode] = VersionRoleHead(model.config.hidden_size)
    if not args.verify:
        vectors, mask, tail = clause_features(model, tokenizer, [row['question'] for row in training], tail_chars=TAIL_CHARS)
        for mode, head in heads.items():
            print('Training', mode, flush=True)
            fit({'roles': head}, vectors, mask, training, args.steps, role_features=tail if mode == 'tail' else None)
    else:
        saved = torch.load(weights_path, map_location='cpu', weights_only=True)
        for mode, head in heads.items():
            head.load_state_dict(saved['heads'][mode], strict=True)
    bank = torch.load(bank_path, map_location='cpu', weights_only=True)
    names = IdentityFingerprints(len(SUBJECTS))
    fields, schedules = {}, {}
    for index, (_, name, field, _, _) in enumerate(SUBJECTS):
        names.bind(index, name)
        fields[index] = (field,)
        schedules[index] = tuple((date.fromisoformat(e), date.fromisoformat(o)) for e, o in bank['revisions'][name])
    held = []
    for split, collection in (('calibration', calibration), ('fresh', fresh)):
        for index, item in enumerate(collection):
            text = item['question']
            source = index % len(SUBJECTS) if split == 'calibration' else item['subject']
            slot = version_slots(schedules[source])[item['target']]
            address = source if split == 'calibration' else addressed_subject(text, names, fields)
            held.append({**item, 'split': split, 'legacy': old[text], 'address': address,
                         'gold': None if slot is None else [source, slot]})
    held.extend({'question': text, 'split': 'regression', 'position': 'regression', 'pair': None,
                 'legacy': row['legacy_label'], 'address': addressed_subject(text, names, fields), 'gold': row['gold']}
                for text, row in regressions.items())
    vectors, mask, tail = clause_features(model, tokenizer, [row['question'] for row in held], tail_chars=TAIL_CHARS)
    predicted = {mode: role_predictions(head, vectors, mask, tail if mode == 'tail' else None)
                 for mode, head in heads.items()}
    boundary = len(calibration)
    thresholds = choose_thresholds({mode: rows[:boundary] for mode, rows in predicted.items()}, held[:boundary], schedules)
    for index, row in enumerate(held):
        row['proposals'] = {mode: predictions[index] for mode, predictions in predicted.items()}
        row['routes'] = {mode: proposal(row, preds[index], thresholds[mode], schedules) for mode, preds in predicted.items()}
        row['routes']['baseline'] = proposal(row, None, 0., schedules, baseline=True)
        row['probe'] = verification_probe(row, row['routes']['tail'], schedules)
    probes = list(dict.fromkeys(row['probe'] for row in held if row['probe'] is not None))
    print('Frozen verifier probes:', len(probes), flush=True)
    evidence = semantic_scores(model, tokenizer, probes)
    for row in held:
        row['verification_score'] = evidence.get(row.pop('probe'))

    def verified(row, threshold):
        score = row['verification_score']
        return row['routes']['tail'] if score is None or score >= threshold else None

    thresholds['verified'] = max((0., .50, .65, .80, .90, .95), key=lambda t: (
        *calibration_key([verified(row, t) for row in held[:boundary]], held[:boundary]), t))
    if not args.verify:
        atomic_save({'heads': {mode: head.state_dict() for mode, head in heads.items()}}, weights_path)
    rows = held[boundary:]
    generate_rows = []
    for row in rows:
        row['routes']['verified'] = verified(row, thresholds['verified'])
        previous = regressions.get(row['question']) if row['split'] == 'regression' else None
        if previous is not None and row['routes']['baseline'] != previous['routes']['scoped_auto']:
            raise ValueError('Baseline changed during regression replay')
        if previous is not None and row['routes']['verified'] == previous['routes']['scoped_auto']:
            row['automatic_output'] = previous['automatic_output']
        else:
            generate_rows.append(row)
    summary = describe(rows)
    print('Thresholds and routing:', thresholds, summary['fresh'], summary['regression'], flush=True)
    state = torch.load(reader_path, map_location='cpu', weights_only=True)['atomic']
    reader = VersionReplyActions(model.config.hidden_size, state['scaffold_tokens'])
    reader.load_state_dict(state, strict=True)
    reader.eval().to('cuda')
    verify_answers(model, tokenizer, bank, reader, generate_rows, route_mode='verified')
    if any(base.sha(Path(name)) != digest for name, digest in sources.items()):
        raise ValueError('A protected input changed')
    generation = {split: {'correct_values': sum(row['automatic_output']['correct_value'] for row in rows if row['split'] == split),
                          'negative_mentions': sum(row['gold'] is None and bool(row['automatic_output']['mentioned_slots'])
                                                   for row in rows if row['split'] == split)}
                  for split in ('fresh', 'regression')}
    report = {'sources': sources, 'heads_sha256': base.sha(weights_path), 'thresholds': thresholds,
              'training': len(training), 'calibration': boundary, 'steps': args.steps if prior is None else prior['steps'],
              'generated': len(generate_rows), 'summary': summary, 'generation': generation, 'rows': rows}
    if prior is not None and any(prior[key] != value for key, value in report.items()):
        raise ValueError('Verifier replay changed the outcome')
    print(generation, flush=True)
    store_json(args.output / ('reload.json' if args.verify else 'result.json'),
               {**report, 'seconds': perf_counter() - started, 'peak_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scope', type=Path)
    parser.add_argument('timeline', type=Path)
    parser.add_argument('reader', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--steps', type=int, default=180)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
