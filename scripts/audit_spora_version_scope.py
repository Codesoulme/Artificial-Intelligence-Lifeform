"""Test timeline fallback and automatic address selection on fresh Chinese prompts."""

import argparse
from collections import Counter
from datetime import date
import gc
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_fast_weight_lexical_write import base, compress_dimension
from scripts.audit_spora_version_reply_actions import atomic_reply
from scripts.audit_spora_version_timeline import learned_routes, SUBJECTS, TODAY
from scripts.audit_spora_versioned_memory_generation import read_versions
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.identity_fingerprints import IdentityFingerprints
from spora.research_version_scope import addressed_subject, choose_scoped_version
from spora.research_version_timeline import effective_order, parse_version_query, select_version
from spora.research_versioned_memory_pointer import VersionReplyActions


def held_cases():
    rows = []
    for index, (_, name, field, _, days) in enumerate(SUBJECTS):
        other = SUBJECTS[(index + 1) % len(SUBJECTS)][1]
        current = (3, 2, 3, 2)[index]
        change = f'2026-10-{days[1][0]:02d}'
        samples = (
            ('calendar', f'请查{name}的{field}，按2026年10月12日当日算，给出是哪一项。', 1),
            ('calendar', f'截至2026-10-15收到的记录，查2026-10-18仍有效的那份。{name}的{field}是什么？', (2, 1, 2, 1)[index]),
            ('calendar', f'换一问。{name}的{field}，{change}那回变动更正之前写的是哪项？', 0),
            ('colloquial', f'{name}的{field}还照哪张办？就按这会儿有效的，只报名称。', current),
            ('colloquial', f'手头那份还算数。{name}的{field}记录说的是哪项？', current),
            ('colloquial', f'过期的那份先撇开。{name}的{field}登记的是哪项？', current),
            ('colloquial_past', f'打头落笔那回，{name}的{field}写的是哪一项？', 0),
            ('no_matching_version', f'请查{name}的{field}，2026-10-01有效的会是哪一项？', None),
            ('no_matching_version', f'{name}的{field}记录，第七份有效记录填的是什么？', None),
            ('no_matching_version', f'2026-10-30那天，{name}的{field}是哪一项？', None),
            ('ambiguous_time', f'手头两张都标有效，一个2026-10-07，一个2026-10-14。{name}的{field}是哪一项？', None),
            ('unsupported_time', f'{name}的{field}记录，2026/10/30那个日期用哪一项？', None),
            ('unsupported_time', f'别管别的，给我当前第十版。{name}的{field}是哪一项？', None),
            ('deny_read', f'请勿读取{name}的{field}，只给我一份空白登记表。', None),
            ('metalinguistic', f'翻译成英语：“{name}的{field}现在是什么？”', None),
            ('wrong_field', f'{name}的门牌编号现在是什么？只说编号。', None),
            ('near_name', f'周{name}的{field}现在写的啥？', None),
            ('near_name', f'{name}昭的{field}现在写的啥？', None),
            ('multiple_subjects', f'核对{name}的{field}记录和{other}的记录。眼下各写了什么？', None),
        )
        rows.extend({'bucket': bucket, 'question': text, 'subject': index, 'expected_slot': slot}
                    for bucket, text, slot in samples)
    return rows


def score_routes(case, label, fingerprints, fields, schedules):
    subject, text = case['subject'], case['question']
    dates = schedules[subject]
    request = parse_version_query(text)
    calendar = select_version(dates, request, TODAY)
    order = effective_order(dates, as_of=TODAY, known_by=TODAY)
    legacy = (order[-1], 0, 1, None)[label]
    fallback = legacy if calendar is None else calendar
    guarded, reason = choose_scoped_version(text, dates, TODAY, label)
    address = addressed_subject(text, fingerprints, fields)
    selected, automatic_reason = ((None, 'unresolved_address') if address is None
                                  else choose_scoped_version(text, schedules[address], TODAY, label))
    gold = None if case['expected_slot'] is None else [subject, case['expected_slot']]
    modes = {'timeline_known': (subject, calendar), 'classifier_known': (subject, legacy),
             'fallback_known': (subject, fallback), 'scoped_known': (subject, guarded),
             'scoped_auto': (address, selected)}
    return {**case, 'gold': gold, 'legacy_label': label, 'scoped_reason': reason,
            'automatic_reason': automatic_reason,
            'routes': {mode: None if slot is None else [key, slot] for mode, (key, slot) in modes.items()}}


def summarize(rows):
    summary = {}
    for row in rows:
        for group in ('all', row['bucket']):
            modes = summary.setdefault(group, {})
            for mode, selection in row['routes'].items():
                counts = modes.setdefault(mode, Counter())
                counts.update({'total': 1, 'correct': row['gold'] == selection,
                               'positive': row['gold'] is not None, 'negative': row['gold'] is None,
                               'wrong_read': selection is not None and selection != row['gold'],
                               'negative_read': row['gold'] is None and selection is not None,
                               'missed_read': row['gold'] is not None and selection is None})
    return summary


def verify_answers(model, tokenizer, bank, head, rows):
    embedding = model.get_input_embeddings().weight.detach()
    columns = compress_dimension(embedding.shape[1]).to(embedding.device)
    codebook = F.normalize(embedding.index_select(1, columns).float(), dim=-1)
    read_cache = {}
    empty = torch.empty(0, dtype=torch.long, device=embedding.device)
    expected_values = [values for _, _, _, values, _ in SUBJECTS]
    names = [name for _, name, _, _, _ in SUBJECTS]
    for index, row in enumerate(rows):
        selected = row['routes']['scoped_auto']
        value_tokens = empty
        if selected is not None:
            subject, slot = selected
            key = subject, slot
            if key not in read_cache:
                name = names[subject]
                record = bank['factors'][name][slot]
                read_cache[key] = read_versions({name: [record]}, codebook)[name][0]
            value_tokens = read_cache[key]
        output = atomic_reply(model, tokenizer, head, row['question'], value_tokens)
        hits = [[subject, slot] for subject, options in enumerate(expected_values)
                for slot, value in enumerate(options) if value in output['answer']]
        row['automatic_output'] = {**output, 'mentioned_slots': hits,
                                   'correct_value': row['gold'] is not None and hits == [row['gold']]
                                   and bool(output['token_ids']) and output['token_ids'][-1] == tokenizer.eos_token_id}
        if (index + 1) % 16 == 0:
            print('generated', index + 1, 'of', len(rows), flush=True)


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    paths = (args.timeline / 'versions.pt', args.timeline / 'result.json', args.reply / 'heads.pt', args.reply / 'result.json')
    bank_path, timeline_result, head_path, reply_result = paths
    timeline = json.loads(timeline_result.read_text(encoding='utf-8'))
    reply = json.loads(reply_result.read_text(encoding='utf-8'))
    protected = {str(path): base.sha(path) for path in paths}
    protected.update({str(path): base.sha(Path(path)) for path in timeline['sources']})
    if (timeline['bank_sha256'] != protected[str(bank_path)] or reply['heads_sha256'] != protected[str(head_path)]
            or any(protected[path] != digest for path, digest in timeline['sources'].items())):
        raise ValueError('The experiment sources differ from the prior timeline audit')
    prior = json.loads(args.output.read_text(encoding='utf-8')) if args.verify else None
    if not args.verify and args.output.exists():
        raise FileExistsError(args.output)
    if prior is not None and prior['sources'] != protected:
        raise ValueError('The source artifacts changed before replay')
    fixtures = held_cases()
    questions = [row['question'] for row in fixtures]
    if len(questions) != len(set(questions)) or set(questions) & {row['question'] for row in timeline['rows']}:
        raise ValueError('The scope audit must use fresh, distinct questions')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    predicted = learned_routes(questions, tokenizer, protected)
    gc.collect()
    torch.cuda.empty_cache()
    print('Finished frozen classifier', flush=True)
    bank = torch.load(bank_path, map_location='cpu', weights_only=True)
    fingerprints = IdentityFingerprints(len(SUBJECTS))
    fields, schedules = {}, {}
    for key, (_, name, field, _, _) in enumerate(SUBJECTS):
        fingerprints.bind(key, name)
        fields[key] = (field,)
        schedules[key] = tuple((date.fromisoformat(e), date.fromisoformat(o)) for e, o in bank['revisions'][name])
    rows = [score_routes(case, predicted[case['question']], fingerprints, fields, schedules) for case in fixtures]
    summary = summarize(rows)
    print('Fresh routing', summary['all'], flush=True)
    model = load_full_streamed()
    weights = torch.load(head_path, map_location='cpu', weights_only=True)['atomic']
    head = VersionReplyActions(model.config.hidden_size, weights['scaffold_tokens'])
    head.load_state_dict(weights, strict=True)
    head.eval().to('cuda')
    verify_answers(model, tokenizer, bank, head, rows)
    if any(base.sha(Path(path)) != digest for path, digest in protected.items()):
        raise ValueError('A protected source changed')
    generation = {'correct_values': sum(row['automatic_output']['correct_value'] for row in rows),
                  'negative_mention': sum(row['gold'] is None and bool(row['automatic_output']['mentioned_slots'])
                                          for row in rows)}
    report = {'sources': protected, 'summary': summary, 'generation': generation, 'rows': rows}
    if prior is not None and any(prior[key] != value for key, value in report.items()):
        raise ValueError('Scope verification changed an output')
    print(generation, flush=True)
    store_json(args.output.with_stem(args.output.stem + '-reload') if args.verify else args.output,
               {**report, 'seconds': perf_counter() - started, 'peak_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('timeline', type=Path)
    parser.add_argument('reply', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
