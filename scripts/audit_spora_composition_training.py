"""Compare matched lookup and composition supervision for tensor-backed replies."""

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import re
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_dual_version_reply import SEED, fit, reply
from scripts.audit_spora_fast_weight_lexical_write import SOURCE, base, compress_dimension, tokens
from scripts.audit_spora_fresh_dialogue import digest
from scripts.audit_spora_memory_in_ordinary_replies import ARMS, BUDGET, output_metrics
from scripts.audit_spora_version_coverage import write_fresh
from scripts.audit_spora_versioned_memory_generation import read_versions
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_memory_composition_cases import FIELDS, FORMS, HELD, TIMES, TRAIN_NAMES
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_ttt_ntp import LowRankFastWeight
from spora.research_version_store import MemoryAddress, VersionStore
from spora.research_versioned_memory_pointer import VersionReplyActions


DATES = (date(2026, 10, 7), date(2026, 10, 16))
MODES = ('lookup', 'compose')


def encode(tokenizer, question, form, values, time):
    tail, actions, text = [], [], []
    for part in form:
        if isinstance(part, int):
            word = values[part]
            ids = tokens(tokenizer, word)
            if not 1 <= len(ids) <= 4:
                raise ValueError(f'Unsupported fact width: {word}')
            actions.append(('copy', part))
            actions.extend(('skip', part) for _ in ids[1:])
        else:
            word = part.format(time=time)
            ids = tokens(tokenizer, word)
            actions.extend(('text', token) for token in ids)
        text.append(word)
        tail.extend(ids)
    if tokenizer.decode(tail) != ''.join(text):
        raise ValueError('Splitting the answer changed the decoded text')
    tail.append(tokenizer.eos_token_id)
    actions.append(('text', tokenizer.eos_token_id))
    return question, tail, actions, None


def training_rows(tokenizer):
    held_values = {value for spec in HELD for pair in spec.scenario.values for value in pair}
    held_names = {spec.scenario.name for spec in HELD}
    rows = {mode: [] for mode in MODES}
    examples = []
    if held_names.intersection(TRAIN_NAMES) or held_values.intersection(value for values in FIELDS.values() for value in values):
        raise ValueError('A held identity or value occurs in training')
    for form_index, (fields, template, answer) in enumerate(FORMS):
        for index, name in enumerate(TRAIN_NAMES):
            time = TIMES[(form_index + index) % len(TIMES)]
            values = tuple(FIELDS[field][(index + slot + form_index) % len(TRAIN_NAMES)]
                           for slot, field in enumerate(fields))
            prompt = template.format(name=name, time=time)
            if any(value in prompt for value in values) or prompt in {spec.scenario.question for spec in HELD}:
                raise ValueError('A training prompt includes a copy target or duplicates an evaluation request')
            short = ('记下的是', 0, '、', 1, '。')
            for mode, form in zip(MODES, (short, answer), strict=True):
                rows[mode].append(encode(tokenizer, prompt, form, values, time))
            examples.append({'question': prompt, 'values': values, 'time': time})
    return rows, examples


def make_records(path, binding, model, tokenizer, columns, settings):
    events = {field: [] for field in FIELDS}
    for spec in HELD:
        case = spec.scenario
        for field, values in zip(case.fields, case.values, strict=True):
            events[field].append((case.name, *values))
    with VersionStore(path, binding, create=True) as store:
        for field, records in events.items():
            bank = write_fresh(model, tokenizer, columns, layer=settings['layers'][0],
                               eta=settings['eta'], ridge=settings['ridge'], events={field: records},
                               facts={field: '\n{value}'}, minimum_tokens=1)
            for name, versions in bank.items():
                for date_value, (keys, factors) in zip(DATES, versions, strict=True):
                    store.append(f'{field}/{name}/{date_value}', MemoryAddress(name, field),
                                 date_value, date_value, LowRankFastWeight(keys, factors))


def get_versions(store, catalog, case, tokenizer, codebook):
    memories = []
    for field, expected in zip(case.fields, case.values, strict=True):
        rows = sorted(catalog[MemoryAddress(case.name, field)], key=lambda row: row.effective)
        if tuple(row.effective for row in rows) != DATES:
            raise ValueError('A held-out record is missing a version')
        states = [store.read(row.number) for row in rows]
        values = read_versions({field: [(state.keys, state.values) for state in states]}, codebook)[field]
        if tuple(tokenizer.decode(value.tolist()) for value in values) != expected:
            raise ValueError('A held-out tensor recovered a different value')
        memories.append(values)
    return memories


def grade(answer, expected, foils, eos, time_pattern, action_groups, stale_time):
    text = answer['answer']
    fields = output_metrics(answer, expected, foils, eos)
    time_ok = bool(re.search(time_pattern, text))
    has_old_time = bool(stale_time and re.search(stale_time, text))
    actions = all(re.search(pattern, text) for pattern in action_groups)
    form = len([part for part in re.split(r'[。！？!?]+', text) if part.strip()]) == 1 and '\n' not in text
    return {**fields, 'time': time_ok, 'stale_time': has_old_time, 'actions': actions, 'one_sentence': form,
            'screened': fields['both'] and time_ok and not has_old_time and actions and form}


def evaluate(model, tokenizer, heads, store, codebook, selected):
    catalog = store.catalog()
    all_cases = []
    summary = {mode: Counter() for mode in heads}
    for spec in selected:
        case = spec.scenario
        memories = get_versions(store, catalog, case, tokenizer, codebook)
        questions = [(arm, case.question, versions, spec.time_pattern, '') for arm, versions in ARMS.items()]
        if spec.changed_question:
            questions.append(('change_time', spec.changed_question, (1, 1), spec.changed_pattern, spec.time_pattern))
        results = {}
        for arm, question, versions, time_pattern, stale_time in questions:
            if any(value in question for pair in case.values for value in pair):
                raise ValueError('A hidden field value appears in a question')
            expected = [case.values[slot][version] for slot, version in enumerate(versions)]
            foils = [case.values[slot][1 - version] for slot, version in enumerate(versions)]
            values = tuple(bank[version] for bank, version in zip(memories, versions, strict=True))
            answers = {}
            for mode, head in heads.items():
                output = reply(model, tokenizer, head, question, values, require_coverage=True, max_new_tokens=BUDGET)
                checks = grade(output, expected, foils, tokenizer.eos_token_id, time_pattern, spec.action_groups, stale_time)
                answers[mode] = {**output, 'checks': checks}
                summary[mode].update(both=checks['both'], stale=checks['stale'], time=checks['time'],
                                     screened=checks['screened'], ended=checks['ended'])
            results[arm] = {'question': question, 'expected': expected, 'answers': answers}
        empty = {
            name: reply(model, tokenizer, head, case.question, (None, None), max_new_tokens=BUDGET)
            for name, head in heads.items()
        }
        if empty['lookup'] != empty['compose']:
            raise ValueError('Changing the head changed generation with no memory selected')
        all_cases.append({'name': case.name, 'arms': results, 'empty': empty['lookup']})
        print(case.name, {arm: {name: value['checks']['screened'] for name, value in row['answers'].items()}
                          for arm, row in results.items()}, flush=True)
    return all_cases, summary


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / ('replay.json' if args.verify else 'result.json')
    if result_path.exists():
        raise FileExistsError(result_path)
    writer_path = SOURCE / 'adapters.pt'
    files = (writer_path, base.MODEL / 'model.safetensors', base.MODEL / 'tokenizer.json',
             base.MODEL / 'tokenizer_config.json', Path(__file__), Path(__file__).with_name('spora_memory_composition_cases.py'))
    sources = {str(path): digest(path) for path in files}
    if sources[str(files[1])] != base.WEIGHT_SHA:
        raise ValueError('Unexpected frozen model digest')
    model = load_full_streamed()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    rows, examples = training_rows(tokenizer)
    scaffold = sorted({token for group in rows.values() for _, _, actions, _ in group for kind, token in actions if kind == 'text'})
    heads = {}
    artifact_digests = {}
    for mode in MODES:
        path = args.output / f'{mode}.pt'
        torch.manual_seed(SEED)
        head = VersionReplyActions(model.config.hidden_size, scaffold, slots=2)
        if args.verify:
            head.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        else:
            fit(model, tokenizer, head, rows[mode], args.steps)
            atomic_save(head.state_dict(), path)
        artifact_digests[mode] = digest(path)
        heads[mode] = head.eval().to(model.device)
    embeddings = model.get_input_embeddings().weight.detach()
    columns = compress_dimension(embeddings.shape[1]).to(embeddings.device)
    codebook = F.normalize(embeddings.index_select(1, columns).float(), dim=-1)
    db = args.output / 'versions.sqlite3'
    binding = 'composition-audit-v1:' + sources[str(writer_path)] + ':' + sources[str(files[1])]
    if not args.verify:
        settings = torch.load(writer_path, map_location='cpu', weights_only=True)
        make_records(db, binding, model, tokenizer, columns, settings)
    before = digest(db)
    picked = HELD if not args.verify else tuple(HELD[index] for index in (0, 2, 5))
    with VersionStore(db, binding) as store:
        observed, totals = evaluate(model, tokenizer, heads, store, codebook, picked)
    if before != digest(db) or any(digest(Path(path)) != expected for path, expected in sources.items()):
        raise ValueError('A protected source or the version store changed')
    result = {'sources': sources, 'heads': artifact_digests, 'database_sha256': before,
              'training_examples': examples, 'training_steps_per_arm': args.steps, 'rows': observed,
              'summary': totals, 'budget': BUDGET, 'seconds': perf_counter() - started,
              'peak_bytes': torch.cuda.max_memory_allocated()}
    if args.verify:
        previous = json.loads((args.output / 'result.json').read_text(encoding='utf-8'))
        names = {row['name'] for row in observed}
        same = ('sources', 'heads', 'database_sha256', 'budget')
        if any(result[key] != previous[key] for key in same) or json.loads(json.dumps(observed)) != [
            row for row in previous['rows'] if row['name'] in names
        ]:
            raise ValueError('Independent replay changed the received evidence')
        result['training_steps_per_arm'] = previous['training_steps_per_arm']
    store_json(result_path, result)
    print(totals, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--steps', type=int, default=220)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('At least one optimization step is required')
    main(args)
