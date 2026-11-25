"""Test semantic field slots and direct copying of confirmed user time spans."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_composition_training import encode, get_versions, grade
from scripts.audit_spora_dual_version_reply import SEED, fit, reply
from scripts.audit_spora_fast_weight_lexical_write import base, compress_dimension, tokens
from scripts.audit_spora_fresh_dialogue import digest
from scripts.audit_spora_memory_in_ordinary_replies import ARMS, BUDGET
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_memory_composition_cases import FIELDS, FORMS, HELD, TIMES, TRAIN_NAMES
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_version_store import VersionStore
from spora.research_versioned_memory_pointer import VersionReplyActions


SLOTS = (*FIELDS, 'time')
MODES = ('typed', 'typed_time')
BINDING_PREFIX = 'composition-audit-v1:'


def arrangement(pairs):
    pairs = tuple(pairs)
    bank = dict(pairs)
    if len(bank) != len(pairs) or bank.keys() - set(SLOTS):
        raise ValueError('Expected one value per recognized field')
    return tuple(bank.get(field) for field in SLOTS)


def remap_answer(form, fields, *, copy_time):
    answer = []
    for part in form:
        if isinstance(part, int):
            answer.append(SLOTS.index(fields[part]))
        elif copy_time and '{time}' in part:
            before, after = part.split('{time}')
            answer.extend(piece for piece in (before, SLOTS.index('time'), after) if piece != '')
        else:
            answer.append(part)
    return tuple(answer)


def training_rows(tokenizer):
    rows = {mode: [] for mode in MODES}
    for form_index, (fields, template, answer) in enumerate(FORMS):
        for index, name in enumerate(TRAIN_NAMES):
            when = TIMES[(form_index + index) % len(TIMES)]
            facts = [FIELDS[field][(index + slot + form_index) % len(TRAIN_NAMES)]
                     for slot, field in enumerate(fields)]
            question = template.format(name=name, time=when)
            values = arrangement([*zip(fields, facts, strict=True), ('time', when)])
            for mode in MODES:
                form = remap_answer(answer, fields, copy_time=mode == 'typed_time')
                rows[mode].append(encode(tokenizer, question, form, values, when))
    return rows


def visible_time(tokenizer, question, pattern, device):
    matches = tuple(re.finditer(pattern, question))
    if len(matches) != 1:
        raise ValueError('The annotated time must identify exactly one span in the visible question')
    span = matches[0].group()
    ids = tokens(tokenizer, span)
    if not 1 <= len(ids) <= 4 or tokenizer.decode(ids) != span:
        raise ValueError('Annotated time span did not survive tokenization')
    return span, torch.tensor(ids, device=device)


def run_case(model, tokenizer, heads, store, catalog, codebook, spec, baseline, summary):
    case = spec.scenario
    if baseline['name'] != case.name:
        raise ValueError('Baseline case order differs')
    memories = get_versions(store, catalog, case, tokenizer, codebook)
    questions = [(mode, case.question, pair, spec.time_pattern, '') for mode, pair in ARMS.items()]
    if spec.changed_question:
        questions.append(('change_time', spec.changed_question, (1, 1), spec.changed_pattern, spec.time_pattern))
    results = {}
    for arm, question, versions, pattern, old_pattern in questions:
        old = baseline['arms'][arm]
        expected = [case.values[slot][version] for slot, version in enumerate(versions)]
        if old['question'] != question or old['expected'] != expected:
            raise ValueError('The saved positional control uses a different question or states')
        foils = [case.values[slot][1 - version] for slot, version in enumerate(versions)]
        if any(value in question for value in (*expected, *foils)):
            raise ValueError('A hidden fact appeared in the visible question')
        facts = tuple((field, versions_for_field[version]) for field, versions_for_field, version
                      in zip(case.fields, memories, versions, strict=True))
        if arrangement(facts) != arrangement(reversed(facts)):
            raise ValueError('Changing record order changed field bindings')
        span, time_ids = visible_time(tokenizer, question, pattern, model.device)
        runs = {'positional': old['answers']['compose']}
        for mode, head in heads.items():
            candidates = (*facts, ('time', time_ids)) if mode == 'typed_time' else facts
            output = reply(model, tokenizer, head, question, arrangement(candidates),
                           require_coverage=True, cover_present_slots=True, max_new_tokens=BUDGET)
            if arm == 'current':
                reversed_output = reply(model, tokenizer, head, question, arrangement(reversed(candidates)),
                                        require_coverage=True, cover_present_slots=True, max_new_tokens=BUDGET)
                if output != reversed_output:
                    raise ValueError('Reordering the same labeled inputs changed the generated response')
            for copy in output['copied']:
                copy['field'] = SLOTS[copy['slot']]
            checks = grade(output, expected, foils, tokenizer.eos_token_id, pattern, spec.action_groups, old_pattern)
            runs[mode] = {**output, 'checks': checks,
                          'copied_time': any(copy['field'] == 'time' for copy in output['copied'])}
        for mode, output in runs.items():
            checks = output['checks']
            summary[mode].update(both=checks['both'], screened=checks['screened'], time=checks['time'],
                                 stale_time=checks['stale_time'], stale=checks['stale'], ended=checks['ended'])
            if arm == 'change_time':
                summary[mode].update(changed_time=checks['time'] and not checks['stale_time'])
        results[arm] = {'question': question, 'expected': expected, 'visible_time': span, 'answers': runs}
    empty = {mode: reply(model, tokenizer, head, case.question, (None,) * len(SLOTS), max_new_tokens=BUDGET)
             for mode, head in heads.items()}
    if any(output != baseline['empty'] for output in empty.values()):
        raise ValueError('A typed head changed ordinary generation with no memory selected')
    print(case.name, {mode: item['checks']['screened'] for mode, item in results['current']['answers'].items()}, flush=True)
    return {'name': case.name, 'arms': results, 'empty': baseline['empty']}


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    target = args.output / ('replay.json' if args.verify else 'result.json')
    if target.exists():
        raise FileExistsError(target)
    baseline_path = args.reference / 'result.json'
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    database = args.reference / 'versions.sqlite3'
    sources = {str(path): digest(path) for path in (baseline_path, database, Path(__file__),
                 Path(__file__).with_name('audit_spora_dual_version_reply.py'),
                 Path(__file__).with_name('spora_memory_composition_cases.py'),
                 Path(__file__).parents[1] / 'spora/research_versioned_memory_pointer.py')}
    model_sha = next(expected for path, expected in baseline['sources'].items() if Path(path).name == 'model.safetensors')
    writer_sha = next(expected for path, expected in baseline['sources'].items() if Path(path).name == 'adapters.pt')
    if model_sha != base.WEIGHT_SHA or sources[str(database)] != baseline['database_sha256']:
        raise ValueError('The positional control and version store do not match')
    binding = BINDING_PREFIX + writer_sha + ':' + model_sha
    model = load_full_streamed()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    rows = training_rows(tokenizer)
    scaffold = sorted({token for group in rows.values() for _, _, actions, _ in group for kind, token in actions if kind == 'text'})
    heads, checksums = {}, {}
    for mode in MODES:
        artifact = args.output / f'{mode}.pt'
        torch.manual_seed(SEED)
        head = VersionReplyActions(model.config.hidden_size, scaffold, slots=len(SLOTS))
        if args.verify:
            head.load_state_dict(torch.load(artifact, map_location='cpu', weights_only=True))
        else:
            fit(model, tokenizer, head, rows[mode], args.steps)
            atomic_save(head.state_dict(), artifact)
        heads[mode] = head.eval().to(model.device)
        checksums[mode] = digest(artifact)
    embeddings = model.get_input_embeddings().weight.detach()
    columns = compress_dimension(embeddings.shape[1]).to(embeddings.device)
    codebook = F.normalize(embeddings.index_select(1, columns).float(), dim=-1)
    selected = [0, 2, 5] if args.verify else range(len(HELD))
    summary = {mode: Counter() for mode in ('positional', *MODES)}
    with VersionStore(database, binding) as store:
        catalog = store.catalog()
        observed = [run_case(model, tokenizer, heads, store, catalog, codebook, HELD[index],
                             baseline['rows'][index], summary) for index in selected]
    if any(digest(Path(path)) != expected for path, expected in sources.items()):
        raise ValueError('A sealed input changed')
    result = {'sources': sources, 'head_sha256': checksums, 'rows': observed,
              'training_steps': args.steps, 'summary': summary, 'seconds': perf_counter() - started,
              'peak_bytes': torch.cuda.max_memory_allocated()}
    if args.verify:
        prior = json.loads((args.output / 'result.json').read_text(encoding='utf-8'))
        if (sources != prior['sources'] or checksums != prior['head_sha256']
                or json.loads(json.dumps(observed)) != [prior['rows'][index] for index in selected]):
            raise ValueError('Independent replay changed the observations')
        result['training_steps'] = prior['training_steps']
    store_json(target, result)
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--steps', type=int, default=220)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('At least one training step is required')
    main(args)
