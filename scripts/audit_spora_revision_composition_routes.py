"""Route held revision requests through the existing per-field version selector."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.audit_spora_revision_version_state import (
    FIELDS, MODEL, NEW_DAY, OLD_DAY, WRITER_SOURCE, AutoTokenizer, VersionReplyActions, VersionStore,
    WorkingTurnState, compress_dimension, extend_positions, frozen_digest, generate_versions, load_full_streamed,
    load_integrated_runtime, load_recurrent_replacements, read_versions, retained_cases, settings_for, using,
)
from scripts.train_spora_revision_composition import grade_use
from scripts.train_spora_recurrent_counterfactual import sha, store_json
from spora.research_multi_version_route import resolve_address
from spora.research_version_dialogue_scope import advance_scope, dialogue_targets
from spora.research_version_store import MemoryAddress


LABELS = {'place': '地点', 'person': '负责人', 'time': '时间', 'gear': '材料', 'quantity': '数量'}
OWNERS = {'market-sign': ('周泠川', '纪鹤言'), 'radio-rehearsal': ('祝沅一', '何初宁'),
          'library-storage': ('曲江岚', '钟时晴'), 'printing-handoff': ('沈远木', '陈舟昀')}


def resolve(question, catalog, scope=()):
    targets = dialogue_targets(question, {address.name for address in catalog}, LABELS, scope)
    decisions = [resolve_address(catalog, item.address, item.query, NEW_DAY) for item in targets]
    versions = tuple(value for value, reason in decisions if reason == 'located')
    return targets, decisions, versions


def route_cases(catalog):
    rows = []
    for case, owners in OWNERS.items():
        first, second = FIELDS[case]
        for variant, owner in enumerate(owners):
            address = MemoryAddress(owner, first)
            retained = MemoryAddress(owner, second)
            initial, newer = catalog[address]
            same, = catalog[retained]
            if (initial.effective, newer.effective, same.effective) != (OLD_DAY, NEW_DAY, OLD_DAY):
                raise ValueError('Expected the matched old and current versions')
            query = f'查{owner}的{LABELS[first]}当前有效的记录；请核对{owner}的{LABELS[second]}首版。'
            targets, decisions, selected = resolve(query, catalog)
            score = selected == (newer.number, same.number) and len(targets) == len(selected)
            rows.append(dict(case=case, variant=variant, test='current', question=query,
                             decisions=decisions, complete=score))
            scope = advance_scope((), targets, selected if score else ())
            query = f'帮我查{owner}的{LABELS[first]}在2026-10-01有效的记录；再查{owner}的{LABELS[second]}现在这份。'
            _, dated, selected = resolve(query, catalog, scope)
            rows.append(dict(case=case, variant=variant, test='as_of', question=query,
                             decisions=dated, complete=selected == (initial.number, same.number) and len(dated) == 2))
            query = f'把「查{owner}的{LABELS[first]}当前版」改写得自然一点。'
            _, decline, selected = resolve(query, catalog, scope)
            rows.append(dict(case=case, variant=variant, test='rewrite', question=query,
                             decisions=decline, complete=not selected))
            wrong = owner + '云'
            query = f'帮我查{wrong}的{LABELS[first]}首版；顺便查{owner}的{LABELS[second]}当前版。'
            _, unknown, selected = resolve(query, catalog, scope)
            rows.append(dict(case=case, variant=variant, test='unknown', question=query,
                             decisions=unknown, complete=selected == (same.number,) and len(unknown) == 2))
    return rows


@torch.inference_mode()
def generate_routed(args, store, rows, sources, identity):
    torch.set_num_threads(2)
    model = load_full_streamed(stage_output_projection=True)
    rotary = model.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, model, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    model.lm_head.to(model.device)
    if frozen_digest(runtime.organism, runtime.state) != identity:
        raise ValueError('Routed replies use a different individual')
    replacements = load_recurrent_replacements(decoder, args.layers, sources[args.checkpoint])
    if set(replacements) != {19, 22}:
        raise ValueError('Expected the two-layer recurrent baseline')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    cases, _ = retained_cases(tokenizer)
    weights = torch.load(args.composer / 'short.pt', weights_only=True, map_location='cpu')
    head = VersionReplyActions(model.config.hidden_size, weights['scaffold_tokens'], slots=2)
    head.load_state_dict(weights, strict=True)
    head.to(model.device).eval().requires_grad_(False)
    embedding = model.get_input_embeddings().weight.detach()
    columns = compress_dimension(embedding.shape[1]).to(embedding.device)
    codebook = F.normalize(embedding.index_select(1, columns).float(), dim=-1)
    settings = settings_for(decoder, tokenizer, 48) | dict(require_coverage=True, complete_on_eos=True,
                                                           question_only=not args.with_history)
    outputs = []
    with using(replacements, 'recurrent'):
        for case in cases:
            target = (case['case'], case['variant'])
            states, questions = {}, {}
            for row in rows:
                if (row['case'], row['variant']) != target or row['test'] not in ('current', 'as_of'):
                    continue
                if len(row['decisions']) != 2 or any(reason != 'located' for _, reason in row['decisions']):
                    continue
                bank = {}
                for field, (number, _) in zip(FIELDS[case['case']], row['decisions'], strict=True):
                    saved = store.read(number)
                    bank[field] = [(saved.keys, saved.values)]
                label = 'current' if row['test'] == 'current' else 'old'
                states[label] = WorkingTurnState().update({name: tuple(ids[0].tolist())
                                                            for name, ids in read_versions(bank, codebook).items()})
                questions[label] = case['question'] if args.writing_only else row['question'] + case['question']
            if states.keys() != {'current', 'old'} or any(value in question for question in questions.values()
                                                          for value in case['values'].values()):
                raise ValueError('Missing a routed state or a question includes a stored value')
            # Lookups are resolved independently of the text passed to the composer.
            outputs_for_case = {}
            for label, state in states.items():
                question_case = {**case, 'question': questions[label]}
                outputs_for_case.update(generate_versions(decoder, tokenizer, settings, head, {label: state}, question_case))
            grade_use(case, outputs_for_case)
            outputs.append(dict(case=target[0], variant=target[1], questions=questions, outputs=outputs_for_case))
    if frozen_digest(runtime.organism, runtime.state) != identity:
        raise ValueError('Routed replies changed the individual')
    return outputs


def main(args):
    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    database = args.reference.parent / 'versions.sqlite'
    sources = {file: sha(file) for file in (args.reference, args.checkpoint, database, WRITER_SOURCE / 'adapters.pt')}
    writer_hash = sources[WRITER_SOURCE / 'adapters.pt']
    if any(reference['sources'].get(str(file)) != sources[file] for file in (args.checkpoint, WRITER_SOURCE / 'adapters.pt')):
        raise ValueError('Version comparison provenance differs')
    binding = sha256(json.dumps((reference['individual_sha256'], writer_hash, sources[args.checkpoint])).encode()).hexdigest()
    if args.composer:
        path = args.composer / 'result.json'
        sources[path] = sha(path)
        composer = json.loads(path.read_text(encoding='utf-8'))
        artifact = args.composer / 'short.pt'
        sources[artifact] = sha(artifact)
        if (composer['results']['short']['head_sha256'] != sources[artifact]
                or composer['individual_sha256'] != reference['individual_sha256']):
            raise ValueError('The trained composition head does not match its artifact or individual')
        for folder in args.layers:
            for name in ('attention.pt', 'comparison.json'):
                path = folder / name
                sources[path] = sha(path)
                if composer['sources'].get(str(path)) != sources[path]:
                    raise ValueError('Routed generation uses different attention weights')
    with VersionStore(database, binding) as store:
        saved = store.catalog()
        catalog = {MemoryAddress(OWNERS[case][int(variant)], address.field, address.condition): versions
                   for address, versions in saved.items() for case, variant in (address.name.split('/'),)}
        if len(catalog) != len(saved) or {owner for owners in OWNERS.values() for owner in owners} != {
                address.name for address in catalog}:
            raise ValueError('The experimental display names do not correspond one-to-one with saved owners')
        rows = route_cases(catalog)
        numbers = {version.number for versions in saved.values() for version in versions}
        for number in {number for row in rows for number, reason in row['decisions'] if reason == 'located'}:
            if number not in numbers or not len(store.read(number).keys):
                raise ValueError('An accepted route did not read a version tensor')
        generated = generate_routed(args, store, rows, sources, reference['individual_sha256']) if args.composer else []
    if any(sha(file) != digest for file, digest in sources.items()):
        raise RuntimeError('A source or the database changed')
    summary = {test: dict(count=sum(row['test'] == test for row in rows),
                          passed=sum(row['test'] == test and row['complete'] for row in rows))
               for test in ('current', 'as_of', 'rewrite', 'unknown')}
    if generated:
        summary['generation'] = {label: {metric: sum(row['outputs'][label][metric] for row in generated)
                                         for metric in ('passed', 'use', 'eos')} for label in ('current', 'old')}
    store_json(args.output, dict(summary=summary, rows=rows, generated=generated, writing_only=args.writing_only,
                                 with_history=args.with_history,
                                 sources={str(file): value for file, value in sources.items()}))
    print(summary, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('reference', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--composer', type=Path)
    cli.add_argument('--layers', type=Path, nargs=2)
    cli.add_argument('--writing-only', action='store_true', help='Give composition only the writing clause after lookup')
    cli.add_argument('--with-history', action='store_true', help='Keep the fixed long history for the writing stage')
    args = cli.parse_args()
    if bool(args.layers) != bool(args.composer):
        cli.error('Use both --composer and --layers for routed generation')
    if args.writing_only and not args.composer:
        cli.error('Writing-only comparisons require a composition head')
    if args.with_history and not args.writing_only:
        cli.error('History comparison requires --writing-only')
    main(args)
