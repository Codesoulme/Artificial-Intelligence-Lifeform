"""Compare positional, coarse-type and role-aware tensor-backed composition."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_composition_training import encode as encode_actions
from scripts.audit_spora_dual_version_reply import SEED, fit_from_features, reply
from scripts.audit_spora_revision_version_state import (
    HELD, MODEL, NEW_DAY, OLD_DAY, SLOT_TYPES, SOURCE, TRANSFER, WRITER_SOURCE, AutoTokenizer, VersionQuery,
    VersionReplyActions, VersionStore, compress_dimension, effective_state, evaluate, extend_positions,
    frozen_digest, grade, load_full_streamed, load_integrated_runtime, load_recurrent_replacements,
    read_versions, records, retained_cases, sha, store_json, using, write_fresh,
)
from scripts.audit_spora_versioned_memory_generation import frozen_features
from scripts.train_spora_revision_composition import TRAIN, factual_roles_correct
from spora.language_training import atomic_save
from spora.research_multi_version_route import resolve_address
from spora.research_ttt_ntp import LowRankFastWeight
from spora.research_version_store import MemoryAddress


ROLES = {
    'display': 'place', 'collector': 'person', 'record_time': 'time', 'editor': 'person',
    'material': 'gear', 'storage': 'place', 'count': 'quantity', 'recipient': 'person',
    'return_site': 'place', 'recorder': 'person',
}
SCENES = {
    'market-sign': ('display', 'collector'), 'radio-rehearsal': ('record_time', 'editor'),
    'library-storage': ('material', 'storage'), 'printing-handoff': ('count', 'recipient'),
    'two-places': ('display', 'return_site'), 'two-people': ('recorder', 'editor'),
}
TRAIN_SCENES = ('market-sign', 'radio-rehearsal', 'library-storage', 'printing-handoff', 'market-sign', 'radio-rehearsal')
REPEATED_TRAIN = (
    (('提示牌该摆在什么地方，结束后收到哪里？写一句。',
      '给工作人员说一下放牌和还牌的两个位置。'),
     ('请把提示牌放在', 0, '，结束后送回', 1, '。'),
     (('小石亭', '新木楼'), ('梧叶台', '泥墙库'), ('湖西口', '竹棚屋'))),
    (('节目由谁来录，谁负责剪辑？请写成分工提醒。',
      '帮我说明录音和剪辑分别交给哪位同事。'),
     ('请', 0, '负责录音，', 1, '负责剪辑。'),
     (('宋寒桥', '汤叶南'), ('凌霜宁', '孟临舟'), ('秦明坞', '鲍轻池'))),
)
REPEATED_HELD = (
    ('two-places', '写给撤场值班员。先交代展示牌在哪里摆，最后又要把牌送还到哪里。',
     (('北槐门', '白沙亭', '玉井坊'), ('杉风台', '竹雨桥', '青瓦阁'))),
    ('two-people', '拟一条节目组的交班短信。把负责录音的人和最后剪辑的人分别写明。',
     (('许冬桥', '卓南枝', '莫景仪'), ('胡燕回', '林芃舟', '顾问秋'))),
)
FRESH_QUESTIONS = {
    'market-sign': '请拟个摆牌通知。把导览牌的摆放地点和散场后的收牌人一并说明。',
    'radio-rehearsal': '编辑组要发一条录音安排。开录的时段和片头剪辑人都写上。',
    'library-storage': '替图书管理员写句值班提示。标签材料照登记的，未登记的书架位置也要说全。',
    'printing-handoff': '给印务柜台留句话，说明这批样张最后交给哪位，以及应交几份。',
    'two-places': '给布展人员写一行，先交代展示牌摆放点，再说撤展时送回的地方。',
    'two-people': '用一句话交代广播组的分工，录音由哪位做，成片由另一位负责剪辑。',
}
MODES = ('positional', 'type', 'role', 'shared')


def descriptors(mode, roles):
    if mode == 'positional':
        return [0] * len(roles)
    role_aware = mode in ('role', 'shared')
    keys = tuple(ROLES) if role_aware else SLOT_TYPES
    return [1 + keys.index(role if role_aware else ROLES[role]) for role in roles]


def training_rows(tokenizer, cases):
    values = {item for row in cases for item in row['values'].values()}
    questions = {row['question'] for row in cases}
    rows, roles = [], []
    for scene, (forms, answer, choices) in zip((*TRAIN_SCENES, 'two-places', 'two-people'),
                                               (*TRAIN, *REPEATED_TRAIN), strict=True):
        for question in forms:
            for pair in choices:
                if question in questions or values.intersection(pair) or any(value in question for value in pair):
                    raise ValueError('Training repeats a held item or exposes a value in the question')
                rows.append(encode_actions(tokenizer, question, answer, pair, ''))
                roles.append(SCENES[scene])
    return rows, roles


def counterbalance(rows, roles, features, lengths):
    expanded, ordered_roles, spans, sizes = [], [], [], []
    for row, pair, span in zip(rows, roles, features.split(lengths), strict=True):
        question, tail, actions, extra = row
        swapped = [(kind, 1 - value if kind in ('copy', 'skip') else value) for kind, value in actions]
        expanded.extend((row, (question, tail, swapped, extra)))
        ordered_roles.extend((pair, pair[::-1]))
        spans.extend((span, span))
        sizes.extend((len(span), len(span)))
    return expanded, ordered_roles, torch.cat(spans), sizes


def repeated_records(path, binding, model, tokenizer, columns, writer, cases):
    with VersionStore(path, binding, create=True) as store:
        for case in cases:
            owner = f'{case["case"]}/{case["variant"]}'
            first, second = SCENES[case['case']]
            for field, role, day in (('old', first, OLD_DAY), ('new', first, NEW_DAY), ('keep', second, OLD_DAY)):
                value = case['values'][field]
                keys, codes = write_fresh(model, tokenizer, columns, layer=writer['layers'][0], eta=writer['eta'],
                    ridge=writer['ridge'], events={role: ((owner, value),)}, facts={role: '\n{value}'},
                    minimum_tokens=1)[owner][0]
                address = MemoryAddress(owner, ROLES[role], role)
                store.append(f'{owner}/{field}', address, day, day, LowRankFastWeight(keys, codes))


def repeated_values(store, catalog, codebook, case, query):
    owner = f'{case["case"]}/{case["variant"]}'
    records = {}
    for role in SCENES[case['case']]:
        address = MemoryAddress(owner, ROLES[role], role)
        number, status = resolve_address(catalog, address, query, NEW_DAY)
        if status != 'located':
            raise ValueError('Missing a same-type version address')
        record = store.read(number)
        records[role] = [(record.keys, record.values)]
    return tuple(tokens[0] for tokens in read_versions(records, codebook).values())


def score_answer(case, label, output, eos):
    text = output['answer']
    info = dict(eos=output['token_ids'][-1:] == [eos], answer=text)
    wanted, foil = ('new', 'old') if label == 'current' else ('old', 'new')
    first, second = (re.escape(case['values'][name]) for name in (wanted, 'keep'))
    passed = grade(info, (first, second), (re.escape(case['values'][foil]),))
    result = {**output, **info, 'passed': passed}
    if case['case'] in ('two-places', 'two-people'):
        patterns = ((fr'(?:放|摆)(?:在|到){first}', fr'(?:送回|还到|归还到){second}')
                    if case['case'] == 'two-places' else
                    (fr'{first}(?:来|负责)?录音', fr'{second}(?:来|负责)?剪辑'))
        result['use'] = passed and all(re.search(pattern, text) for pattern in patterns) and all(
            text.count(case['values'][name]) == 1 for name in (wanted, 'keep'))
    else:
        # Reuse the same role rules for the eight previously tested prompts.
        result['use'] = factual_roles_correct(case, case['values'][wanted], result)
    return result


def assess(model, tokenizer, head, mode, cases, values):
    results = []
    for case, state in zip(cases, values, strict=True):
        roles = SCENES[case['case']]
        outputs = {}
        for label, pair in state.items():
            versions = {}
            for order, ids in (('forward', (0, 1)), ('reversed', (1, 0))):
                fields = torch.tensor(descriptors(mode, tuple(roles[index] for index in ids)), device=model.device)
                answer = reply(model, tokenizer, head, case['question'], tuple(pair[index] for index in ids),
                               max_new_tokens=48, require_coverage=True, complete_on_eos=True, field_ids=fields)
                versions[order] = score_answer(case, label, answer, tokenizer.eos_token_id)
            outputs[label] = versions
        row = dict(case=case['case'], variant=case['variant'], values=case['values'], outputs=outputs)
        results.append(row)
    metrics = {}
    for label, group in (('distinct', results[:8]), ('same', results[8:])):
        metrics[label] = {order: {key: sum(row['outputs'][version][order][key] for row in group
                                           for version in ('current', 'old'))
                                  for key in ('passed', 'use', 'eos')} for order in ('forward', 'reversed')}
        metrics[label]['both_orders'] = sum(all(result['use'] for result in versions.values())
                                             for row in group for versions in row['outputs'].values())
        metrics[label]['same_tokens'] = sum(versions['forward']['token_ids'] == versions['reversed']['token_ids']
                                            for row in group for versions in row['outputs'].values())
    return dict(summary=metrics, rows=results)


def main(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    archive = args.reference.parent / 'versions.sqlite'
    writer_path = WRITER_SOURCE / 'adapters.pt'
    sources = {path: sha(path) for path in (args.checkpoint, args.reference, archive, writer_path, SOURCE, TRANSFER,
        *(directory / name for directory in args.layers for name in ('attention.pt', 'comparison.json')))}
    replay = None
    if args.replay:
        manifest = args.replay / 'result.json'
        replay = json.loads(manifest.read_text(encoding='utf-8'))
        if (replay['sources'] != {str(path): digest for path, digest in sources.items()} or replay['roles'] != ROLES
                or tuple(replay['roles']) != tuple(ROLES)
                or set(args.mode or MODES) - replay['results'].keys()):
            raise ValueError('Replay requires a matched training result for each selected head')
    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    if any(reference['sources'].get(str(path)) != value for path, value in sources.items()
           if path not in (args.reference, archive)):
        raise ValueError('Input checkpoint, data or attention layers differ from the version experiment')
    if replay:
        sources[manifest] = sha(manifest)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    previous, _ = retained_cases(tokenizer)
    extra = [dict(case=scene, variant=index, question=question, values=dict(zip(('old', 'new', 'keep'), row, strict=True)))
             for scene, question, values in REPEATED_HELD for index, row in enumerate(values)]
    cases = previous + extra
    if args.fresh_questions:
        if any(FRESH_QUESTIONS[case['case']] == case['question'] for case in cases):
            raise ValueError('Fresh questions cannot repeat the original prompts')
        cases = [{**case, 'question': FRESH_QUESTIONS[case['case']]} for case in cases]
    if any(value in case['question'] or not 1 <= len(tokenizer.encode(value, add_special_tokens=False)) <= 4
           for case in cases for value in case['values'].values()):
        raise ValueError('Held facts leak into the question or exceed the version codec budget')
    if len(previous) != 8 or len(extra) != 4:
        raise ValueError('The two evaluation splits changed')
    rows, roles = training_rows(tokenizer, cases)
    scaffold = sorted({token for _, _, actions, _ in rows for kind, token in actions if kind == 'text'})
    model = load_full_streamed(stage_output_projection=True)
    rotary = model.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, model, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    model.lm_head.to(model.device)
    identity = frozen_digest(runtime.organism, runtime.state)
    if identity != reference['individual_sha256'] or (replay and identity != replay['individual_sha256']):
        raise ValueError('Role-aware composition uses a different individual')
    replacements = load_recurrent_replacements(decoder, args.layers, sources[args.checkpoint])
    if set(replacements) != {19, 22}:
        raise ValueError('Expected the two-layer recurrent baseline')
    embedding = model.get_input_embeddings().weight.detach()
    columns = compress_dimension(embedding.shape[1]).to(embedding.device)
    codebook = F.normalize(embedding.index_select(1, columns).float(), dim=-1)
    binding = sha256(json.dumps((identity, sources[writer_path], sources[args.checkpoint])).encode()).hexdigest()
    second = (args.replay or args.output) / 'repeated-roles.sqlite'
    if replay:
        sources[second] = sha(second)
        if replay.get('database_sha256') != sources[second]:
            raise ValueError('The same-type version database differs from the training result')
    else:
        repeated_records(second, binding, model, tokenizer, columns,
                         torch.load(writer_path, map_location='cpu', weights_only=True), extra)
    comparisons = {}
    with ExitStack() as stack:
        old_store = stack.enter_context(VersionStore(archive, binding))
        new_store = stack.enter_context(VersionStore(second, binding))
        old_catalog, new_catalog = old_store.catalog(), new_store.catalog()
        states = []
        for case in cases:
            bank = {}
            for label, wanted, query in (('current', 'new', VersionQuery('current')),
                                         ('old', 'old', VersionQuery('as_of', OLD_DAY))):
                if case['case'] in ('two-places', 'two-people'):
                    pair = repeated_values(new_store, new_catalog, codebook, case, query)
                else:
                    state = effective_state(old_store, old_catalog, codebook, case, query)
                    pair = tuple(torch.tensor(state.slots[SLOT_TYPES.index(ROLES[role])], device=model.device)
                                 for role in SCENES[case['case']])
                if [tokenizer.decode(value.tolist()) for value in pair] != [case['values'][wanted], case['values']['keep']]:
                    raise ValueError('The version store recovered an unexpected role value')
                bank[label] = pair
            states.append(bank)
        stack.enter_context(using(replacements, 'recurrent'))
        if not replay:
            features, lengths = frozen_features(model, tokenizer, rows)
            rows, roles, features, lengths = counterbalance(rows, roles, features, lengths)
        for mode in args.mode or MODES:
            torch.manual_seed(SEED)
            head = VersionReplyActions(model.config.hidden_size, scaffold, slots=2, field_count=1 + len(ROLES),
                                       shared_role_pointer=mode == 'shared')
            artifact = (args.replay or args.output) / f'{mode}.pt'
            if replay:
                sources[artifact] = sha(artifact)
                if sources[artifact] != replay['results'][mode]['head_sha256']:
                    raise ValueError('A saved head differs from the matched training result')
            else:
                field_ids = torch.tensor([descriptors(mode, pair) for pair in roles])
                fit_from_features(head, rows, args.steps, features, lengths, field_ids=field_ids)
                atomic_save(head.state_dict(), artifact)
            head.load_state_dict(torch.load(artifact, map_location='cpu', weights_only=True), strict=True)
            head.to(model.device).eval().requires_grad_(False)
            result = assess(model, tokenizer, head, mode, cases, states)
            if replay and not args.fresh_questions and result['rows'] != replay['results'][mode]['rows']:
                raise RuntimeError('Reload changed a saved evaluation output')
            result['head_sha256'] = sha(artifact)
            comparisons[mode] = result
            store_json(args.output / f'{mode}.json', result)
            print(mode, result['summary'], flush=True)
        memory = evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))['summary']
        if memory != reference['memory']:
            raise RuntimeError('The role experiment changed the addressed memory baseline')
    if identity != frozen_digest(runtime.organism, runtime.state) or any(sha(path) != value for path, value in sources.items()):
        raise RuntimeError('A source tensor or the individual changed')
    store_json(args.output / 'result.json', dict(results=comparisons, steps=replay['steps'] if replay else args.steps,
               training_rows=replay['training_rows'] if replay else len(rows), fresh_questions=args.fresh_questions,
               questions={case['case']: case['question'] for case in cases},
               database_sha256=sha(second),
               roles=ROLES, memory=memory, individual_sha256=identity,
               sources={str(path): value for path, value in sources.items()}, seconds=perf_counter() - started))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('reference', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('layers', nargs=2, type=Path)
    cli.add_argument('--steps', type=int, default=240)
    cli.add_argument('--mode', action='append', choices=MODES)
    cli.add_argument('--replay', type=Path)
    cli.add_argument('--fresh-questions', action='store_true')
    args = cli.parse_args()
    if args.steps < 1:
        cli.error('Expected a positive training step count')
    if args.mode and len(set(args.mode)) != len(args.mode):
        cli.error('Use each mode at most once')
    if args.fresh_questions and not args.replay:
        cli.error('Fresh questions must use a frozen artifact from --replay')
    main(args)
