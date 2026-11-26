"""Swap persistent version tensors while holding long Chinese dialogue fixed."""

import argparse
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_fast_weight_lexical_write import SOURCE as WRITER_SOURCE, compress_dimension
from scripts.audit_spora_dual_version_reply import reply as span_reply
from scripts.audit_spora_revision_factors import SOURCE, TRANSFER
from scripts.audit_spora_version_coverage import write_fresh
from scripts.audit_spora_versioned_memory_generation import read_versions
from scripts.benchmark_spora_recurrent_dialogue import (
    MODEL, AutoTokenizer, extend_positions, frozen_digest, grade, load_full_streamed, load_integrated_runtime,
    recall_prompts, reply, settings_for, sha, using,
)
from scripts.train_spora_hybrid_hidden_recovery import HELD, evaluate, records, store_json
from spora.research_dialogue_working_state import (
    FIELDS as SLOT_TYPES, CoveredWorkingTurnReader, MixtureWorkingTurnReader, WorkingTurnState,
)
from spora.research_multi_version_route import resolve_address
from spora.research_recurrent_decode import load_recurrent_replacements
from spora.research_ttt_ntp import LowRankFastWeight
from spora.research_version_store import MemoryAddress, VersionStore
from spora.research_version_timeline import VersionQuery
from spora.research_versioned_memory_pointer import VersionReplyActions


FIELDS = {
    'market-sign': ('place', 'person'),
    'radio-rehearsal': ('time', 'person'),
    'library-storage': ('gear', 'place'),
    'printing-handoff': ('quantity', 'person'),
    'seed-workshop': ('gear', 'gear'),
}
OLD_DAY, NEW_DAY = date(2026, 10, 1), date(2026, 10, 2)


def effective_state(store, catalog, codebook, case, query):
    owner = f'{case["case"]}/{case["variant"]}'
    bank = {}
    for field in FIELDS[case['case']]:
        number, reason = resolve_address(catalog, MemoryAddress(owner, field), query, NEW_DAY)
        if reason != 'located':
            raise ValueError(f'No resolved version for {owner}, {field}: {reason}')
        state = store.read(number)
        bank[field] = [(state.keys, state.values)]
    return WorkingTurnState().update({name: tuple(ids[0].tolist()) for name, ids in read_versions(bank, codebook).items()})


def populate(store, base, tokenizer, columns, settings, cases):
    for case in cases:
        fields = FIELDS[case['case']]
        owner = f'{case["case"]}/{case["variant"]}'
        for value_key, field, day, message in (('old', fields[0], OLD_DAY, 0), ('new', fields[0], NEW_DAY, 2),
                                                ('keep', fields[1], OLD_DAY, 0)):
            value = case['values'][value_key]
            utterance = case['history'][message]['content']
            if utterance.count(value) != 1:
                raise ValueError('Unexpected number of value occurrences in the source message')
            keys, values = write_fresh(base, tokenizer, columns, layer=settings['layers'][0], eta=settings['eta'],
                ridge=settings['ridge'], events={field: ((owner, value),)}, facts={field: utterance},
                minimum_tokens=1)[owner][0]
            store.append(f'{owner}/{value_key}', MemoryAddress(owner, field), day, day, LowRankFastWeight(keys, values))


def retained_cases(tokenizer):
    audit = json.loads(SOURCE.read_text(encoding='utf-8'))
    audit['recall'] = json.loads(TRANSFER.read_text(encoding='utf-8'))
    rows, skipped = [], []
    for case in recall_prompts(tokenizer, audit, extended=True):
        if case['length'] != 'extended':
            continue
        reasons = []
        if len(set(FIELDS[case['case']])) != 2:
            reasons.append('shared reader field type')
        if any(not 1 <= len(tokenizer.encode(value, add_special_tokens=False)) <= 4
               for value in case['values'].values()):
            reasons.append('outside the version codec span length')
        if reasons:
            skipped.append(dict(case=case['case'], variant=case['variant'], reasons=reasons))
        else:
            rows.append(case)
    return rows, skipped


def generate(decoder, tokenizer, settings, reader, state, case):
    if isinstance(reader, VersionReplyActions):
        fields = FIELDS[case['case']]
        pieces = [state.slots[SLOT_TYPES.index(field)] for field in fields]
        device = decoder.base.device
        memories = [torch.tensor(part, dtype=torch.long, device=device) if part else None for part in pieces]
        history = () if settings['question_only'] else case['history']
        output = span_reply(decoder.base, tokenizer, reader, case['question'], memories, history=history,
                            require_coverage=settings['require_coverage'], max_new_tokens=settings['limit'],
                            complete_on_eos=settings['complete_on_eos'])
        return {**output, 'eos': output['token_ids'][-1:] == [tokenizer.eos_token_id],
                'generated_tokens': len(output['token_ids'])}
    with reader.active(decoder, state):
        answer = reply(decoder, tokenizer, case['history'], case['question'], **settings)
    if answer['prompt_tokens'] != case['input_tokens']:
        raise ValueError('The generation prompt differed from the registered scenario')
    answer.pop('seconds')
    return answer


def generate_versions(decoder, tokenizer, settings, reader, states, case):
    scored = {}
    for label, state in states.items():
        answer = generate(decoder, tokenizer, settings, reader, state, case)
        values = case['values']
        wanted, foil = ('old', 'new') if label == 'old' else ('new', 'old')
        scored[label] = {**answer, 'mentions': {field: text in answer['answer'] for field, text in values.items()},
                          'passed': grade(answer, [re.escape(values[wanted]), re.escape(values['keep'])],
                                          [re.escape(values[foil])])}
    return scored


def summary(rows):
    counts = {mode: {label: {'passed': sum(row['answers'][mode][label]['passed'] for row in rows),
                            'eos': sum(row['answers'][mode][label]['eos'] for row in rows),
                            'mentions': {field: sum(row['answers'][mode][label]['mentions'][field] for row in rows)
                                         for field in ('new', 'old', 'keep')}}
                     for label in rows[0]['answers'][mode]} for mode in rows[0]['answers']}
    for mode, results in counts.items():
        results['both_versions'] = sum(all(row['answers'][mode][label]['passed'] for label in ('current', 'old'))
                                       for row in rows)
    return counts


@torch.inference_mode()
def main(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    head_path, head_report = args.reader / 'paired.pt', args.reader / 'result.json'
    writer_path = WRITER_SOURCE / 'adapters.pt'
    sources = {path: sha(path) for path in (args.checkpoint, SOURCE, TRANSFER, writer_path, head_path, head_report,
        *(directory / name for directory in args.layers for name in ('attention.pt', 'comparison.json')))}
    prior = None
    if args.span_reader:
        sources[args.reference] = sha(args.reference)
        prior = json.loads(args.reference.read_text(encoding='utf-8'))
        if any(prior['sources'].get(str(path)) != digest for path, digest in sources.items() if path != args.reference):
            raise ValueError('Span and token readers need the same model and data provenance')
    reference = json.loads(head_report.read_text(encoding='utf-8'))
    if (reference['arms']['paired']['sha256'] != sources[head_path]
            or reference['sources'][str(args.checkpoint)] != sources[args.checkpoint]):
        raise ValueError('Reader weights and their parent checkpoint differ from the training report')
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    cases, skipped = retained_cases(tokenizer)
    if not cases:
        raise ValueError('No cases fit the version codec and reader field types')
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    base.lm_head.to(base.device)
    identity = frozen_digest(runtime.organism, runtime.state)
    if identity != reference['individual_sha256']:
        raise ValueError('Reader was trained with a different individual')
    embedding = base.get_input_embeddings().weight.detach()
    columns = compress_dimension(embedding.shape[1]).to(embedding.device)
    codebook = F.normalize(embedding.index_select(1, columns).float(), dim=-1)
    binding = sha256(json.dumps((identity, sources[writer_path], sources[args.checkpoint])).encode()).hexdigest()
    if prior is None:
        writer_settings = torch.load(writer_path, map_location='cpu', weights_only=True)
        database = args.output / 'versions.sqlite'
        with VersionStore(database, binding, create=True) as store:
            populate(store, base, tokenizer, columns, writer_settings, cases)
        readers = {'mixture': MixtureWorkingTurnReader(base.config.hidden_size),
                   'coverage': CoveredWorkingTurnReader(base.config.hidden_size, follow_span=True)}
        weights = torch.load(head_path, map_location='cpu', weights_only=True)
        if weights['mode'] != 'paired':
            raise ValueError('Expected a reader trained on paired states')
        for reader in readers.values():
            reader.load_state_dict(weights['reader'], strict=True)
    else:
        database = args.reference.parent / 'versions.sqlite'
        span_head, span_report = args.span_reader / 'head.pt', args.span_reader / 'result.json'
        for path in (database, span_head, span_report):
            sources[path] = sha(path)
        manifest = json.loads(span_report.read_text(encoding='utf-8'))
        if (manifest['head_sha256'] != sources[span_head]
                or sources[writer_path] not in manifest['sources'].values()):
            raise ValueError('Span reader head or codec provenance differs')
        weights = torch.load(span_head, map_location='cpu', weights_only=True)
        reader = VersionReplyActions(base.config.hidden_size, weights['scaffold_tokens'], slots=2)
        reader.load_state_dict(weights, strict=True)
        readers = {'span': reader}
    for reader in readers.values():
        reader.to(base.device).eval().requires_grad_(False)
    replacements = load_recurrent_replacements(decoder, args.layers, sources[args.checkpoint])
    if set(replacements) != {19, 22}:
        raise ValueError('Expected the two-layer recurrent baseline')
    settings = settings_for(decoder, tokenizer, 48)
    if prior is not None:
        settings['require_coverage'] = not args.allow_early_stop
        settings['complete_on_eos'] = args.complete_on_eos
        settings['question_only'] = args.question_only
    rows = []
    memory_cases = records(runtime.organism, HELD)
    with VersionStore(database, binding) as store, using(replacements, 'recurrent'):
        catalog = store.catalog()
        for case in cases:
            states = {'current': effective_state(store, catalog, codebook, case, VersionQuery('current')),
                      'old': effective_state(store, catalog, codebook, case, VersionQuery('as_of', OLD_DAY)),
                      'empty': WorkingTurnState()}
            fields = FIELDS[case['case']]
            if args.question_only and any(value in case['question'] for value in case['values'].values()):
                raise ValueError('The question contains a stored value')
            decoded = {label: [tokenizer.decode(state.slots[SLOT_TYPES.index(field)]) for field in fields]
                       if label != 'empty' else [] for label, state in states.items()}
            if any(decoded[label] != [case['values'][key], case['values']['keep']]
                   for label, key in (('current', 'new'), ('old', 'old'))):
                raise ValueError(f'Version reconstruction failed for {case["case"]} #{case["variant"]}')
            answers = {label: generate_versions(decoder, tokenizer, settings, reader, states, case)
                       for label, reader in readers.items()}
            if prior is None and answers['mixture']['empty'] != answers['coverage']['empty']:
                raise RuntimeError('Empty state changed normal generation between readers')
            if prior is not None:
                baseline = prior['rows'][len(rows)]
                if (baseline['case'], baseline['variant']) != (case['case'], case['variant']):
                    raise RuntimeError('Span rows do not match the original comparison')
                empty = answers['span']['empty']['answer']
                if args.question_only:
                    control = reply(decoder, tokenizer, [], case['question'], **settings_for(decoder, tokenizer, 48))
                    if empty != control['answer']:
                        raise RuntimeError('Empty-span generation changes the question-only reply')
                elif empty != baseline['answers']['coverage']['empty']['answer']:
                    raise RuntimeError('Empty-span generation differs from the ordinary dialogue baseline')
            row = dict(case=case['case'], variant=case['variant'], versions=decoded, values=case['values'],
                       answers=answers)
            rows.append(row)
            store_json(args.output / f'{case["case"]}-{case["variant"]}.json', row)
            print(case['case'], case['variant'], {mode: {key: row['passed'] for key, row in arm.items()}
                                                  for mode, arm in answers.items()}, flush=True)
        memory = evaluate(decoder, tokenizer, runtime.organism, memory_cases)
        if prior is None:
            with readers['coverage'].active(decoder, states['current']):
                if evaluate(decoder, tokenizer, runtime.organism, memory_cases) != memory:
                    raise RuntimeError('Coverage reader interfered with addressed individual memory')
        elif memory['summary'] != prior['memory']:
            raise RuntimeError('Addressed individual memory changed in the span comparison')
    if identity != frozen_digest(runtime.organism, runtime.state) or any(
            sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('A source or the individual changed during the comparison')
    store_json(args.output / 'result.json', dict(rows=rows, skipped=skipped, summary=summary(rows),
               memory=memory['summary'], individual_sha256=identity, settings=settings,
               sources={str(path): digest for path, digest in sources.items()}, seconds=perf_counter() - started))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('reader', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('layers', type=Path, nargs=2)
    cli.add_argument('--span-reader', type=Path)
    cli.add_argument('--reference', type=Path)
    cli.add_argument('--question-only', action='store_true', help='Run the span reader with the final question and no history')
    completion = cli.add_mutually_exclusive_group()
    completion.add_argument('--allow-early-stop', action='store_true', help='Allow the span reader to choose EOS at any step')
    completion.add_argument('--complete-on-eos', action='store_true', help='Replace premature EOS with a pending span')
    args = cli.parse_args()
    if bool(args.reference) != bool(args.span_reader):
        cli.error('Use both --span-reader and --reference for a matched span-reading comparison')
    if (args.allow_early_stop or args.complete_on_eos or args.question_only) and not args.span_reader:
        cli.error('Completion and question-only comparisons require the span reader')
    main(args)
