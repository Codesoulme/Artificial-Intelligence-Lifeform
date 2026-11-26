"""Resolve role-scoped version requests without passing stored values to the planner."""

import argparse
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_dual_version_reply import reply
from scripts.train_spora_revision_roles import (
    HELD, MODEL, NEW_DAY, ROLES, SCENES, SEED, WRITER_SOURCE,
    AutoTokenizer, MemoryAddress, VersionQuery, VersionReplyActions, VersionStore,
    compress_dimension, descriptors, evaluate, extend_positions, frozen_digest,
    load_full_streamed, load_integrated_runtime, load_recurrent_replacements, read_versions,
    records, resolve_address, sha, store_json, using,
)
from spora.language_training import atomic_save


SKIP, CURRENT, FIRST = range(3)
VERSIONS = {CURRENT: VersionQuery('current'), FIRST: VersionQuery('ordinal', 1)}
NAMES = dict(zip(ROLES, ('展示牌摆放地点', '负责收回展示牌的人', '安排的录音时间', '负责片头剪辑的人',
                         '标签使用的材料', '未登记图书存放点', '交付的样张数量', '样张的接收人',
                         '撤展后的归还地点', '负责录音的人'), strict=True))
ALIASES = dict(zip(ROLES, (
    ('指示物陈列地点', '牌子展示点', '指引牌摆设位置'),
    ('收取展牌的人员', '拆展取走标识的同事', '领回提示牌的负责人'),
    ('播音室的录制时刻', '节目收音时段', '开始采音的时间'),
    ('给开场音频做后期的同事', '剪开场白的人员', '编辑片头的负责人'),
    ('标识签的制作材料', '书签用料', '制作图书标贴的素材'),
    ('还没入库的书籍存放处', '未入账书册的暂存点', '待录入图书的所在位置'),
    ('打版交接的份数', '交样的册数', '交印刷小样的总量'),
    ('领取印刷小样的人', '收印制样品的同事', '交样的签收者'),
    ('展后交还指示牌的地方', '收摊送回标牌的地点', '标识牌的还回目的地'),
    ('操作录音机的人员', '给节目采集声音的同事', '负责收音的同事'),
), strict=True))
TRAIN_FORMS = (
    ('请列出{a}现在这份，还要{b}首版。', (CURRENT, FIRST)),
    ('查{a}最早的值，{b}按现行版本。', (FIRST, CURRENT)),
    ('{b}首次记的和{a}目前用的，各是什么？', (CURRENT, FIRST)),
    ('先报{b}有效的，再报{a}最初登记的。', (FIRST, CURRENT)),
    ('{a}和{b}，都查最新的。', (CURRENT, CURRENT)),
    ('帮我看{b}和{a}各自第一版记录。', (FIRST, FIRST)),
    ('本次只核对{a}当前的，{b}不用查。', (CURRENT, SKIP)),
    ('只报{b}最初的记录，略过{a}。', (SKIP, FIRST)),
    ('用{a}的初始记录就行，先别查{b}。', (FIRST, SKIP)),
    ('{b}现在的值发我，{a}等会儿再说。', (SKIP, CURRENT)),
    ('「查{a}的旧版和{b}的新值」帮我改写一下。', (SKIP, SKIP)),
    ('{a}和{b}先放一放，如何写好交接通知？', (SKIP, SKIP)),
)
# This independent split changes both the role wording and the version syntax.
HOLD_FORMS = (
    ('拟条交接提醒，把{a}沿用眼下定稿的那项，把{b}按最早登记的填。', (CURRENT, FIRST)),
    ('请先按眼下有效的记录写{b}；{a}我要核对刚建档时的那份。', (FIRST, CURRENT)),
    ('按最近生效那次的记录写一句，把{a}和{b}都说明。', (CURRENT, CURRENT)),
    ('只看起初定下来的两项，用一句话说明{b}和{a}。', (FIRST, FIRST)),
    ('这次先把{a}刚确认还有效的值说清楚，{b}回头再对。', (CURRENT, SKIP)),
    ('先别动{a}，我只想知道{b}建档时填的是什么。', (SKIP, FIRST)),
    ('不需要翻档。讨论一下怎样区分{a}和{b}的用途。', (SKIP, SKIP)),
    ('请润色「按现在的{a}和起初的{b}写通知」这句要求。', (SKIP, SKIP)),
)


class RolePlan(nn.Module):
    def __init__(self, width, *, role_attention=False):
        super().__init__()
        self.actions = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 3))
        self.query = nn.Linear(width, 48, bias=False) if role_attention else None
        self.key = nn.Linear(width, 48, bias=False) if role_attention else None

    def forward(self, states, context=None, allowed=None):
        if self.query is not None:
            query = self.query(F.rms_norm(states, (states.shape[-1],)))
            keys = self.key(F.rms_norm(context, (context.shape[-1],)))
            score = (keys * query.unsqueeze(-2)).sum(-1) / query.shape[-1] ** .5
            weight = score.masked_fill(~allowed, -torch.inf).softmax(-1)
            states = states + (weight.unsqueeze(-1) * context).sum(-2)
        return self.actions(states)


def query_states(extract, requests, *, role_attention):
    prompts = [question + '\n核对角色：' + NAMES[role] for question, pair, _ in requests for role in pair]
    hidden = extract(prompts)
    states = torch.stack([vector[len(text) - 6:len(text)].float().mean(0)
                          for text, vector in zip(prompts, hidden, strict=True)]).unflatten(0, (-1, 2))
    if not role_attention:
        return (states,)
    lengths = torch.tensor([len(question) for question, _, _ in requests])
    width = int(lengths.max())
    context = hidden[:, :width].float().unflatten(0, (-1, 2))
    allowed = (torch.arange(width) < lengths[:, None]).unsqueeze(1).expand(-1, 2, -1)
    return states, context, allowed


def training_requests():
    return [(form.format(a=ALIASES[pair[0]][index], b=ALIASES[pair[1]][(index + 1) % 3]), pair, labels)
            for pair in SCENES.values() for form, labels in TRAIN_FORMS for index in range(3)]


def hold_requests():
    alternatives = (
        ('导览指示牌要落在哪儿', '散场后撤牌的人'),
        ('开录的档期', '剪片头的那位'),
        ('制作书签标签用的东西', '未登记读物搁在哪儿'),
        ('应该交收的样张数目', '样张最后交给的那位'),
        ('展牌上场时摆的地方', '撤场后放回的地点'),
        ('录音现场负责采音的那位', '给录音剪片头的人'),
    )
    return [(scene, form.format(a=first, b=second), pair, labels)
            for (scene, pair), (first, second) in zip(SCENES.items(), alternatives, strict=True)
            for form, labels in HOLD_FORMS]


def fit(features, requests, steps, *, role_attention):
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)
    head = RolePlan(features[0].shape[-1], role_attention=role_attention)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    targets = torch.tensor([labels for _, _, labels in requests])
    for _ in range(steps):
        batch = torch.randperm(len(requests), generator=rng)[:32]
        loss = F.cross_entropy(head(*(value[batch] for value in features)).flatten(0, 1), targets[batch].flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return head.eval()


def selected_tokens(store, catalog, codebook, addresses, actions):
    selected, bank = [], {}
    for role, address, action in zip(addresses.keys(), addresses.values(), actions, strict=True):
        if action == SKIP:
            continue
        number, reason = resolve_address(catalog, address, VERSIONS[action], NEW_DAY)
        if reason != 'located':
            raise ValueError('The planner selected an unavailable version')
        record = store.read(number)
        bank[role] = [(record.keys, record.values)]
        selected.append(number)
    return selected, {key: values[0] for key, values in read_versions(bank, codebook).items()}


def main(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    archive, shared = args.reference.parent / 'versions.sqlite', args.reader / 'shared.pt'
    repeated, report = args.reader / 'repeated-roles.sqlite', args.reader / 'result.json'
    writer = WRITER_SOURCE / 'adapters.pt'
    inputs = (args.checkpoint, args.reference, archive, shared, repeated, report, writer,
              *(folder / name for folder in args.layers for name in ('attention.pt', 'comparison.json')))
    sources = {path: sha(path) for path in inputs}
    manifest = json.loads(report.read_text(encoding='utf-8'))
    matched = [path for path in inputs if path not in (shared, repeated, report)]
    if (manifest['database_sha256'] != sources[repeated] or manifest['results']['shared']['head_sha256'] != sources[shared]
            or tuple(manifest['roles']) != tuple(ROLES)
            or any(manifest['sources'].get(str(path)) != sources[path] for path in matched)):
        raise ValueError('The saved composition head and version stores do not match')
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed(stage_output_projection=True)
    rotary = model.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, model, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    model.lm_head.to(model.device)
    identity = frozen_digest(runtime.organism, runtime.state)
    if identity != manifest['individual_sha256']:
        raise ValueError('The routing audit uses a different individual')
    replacements = load_recurrent_replacements(decoder, args.layers, sources[args.checkpoint])
    if set(replacements) != {19, 22}:
        raise ValueError('Expected the matched two-layer recurrent model')
    train, hold = training_requests(), hold_requests()
    if {question for question, *_ in train} & {row[1] for row in hold}:
        raise ValueError('Training and held requests overlap')
    extract = FrozenTurnFeatures(decoder, tokenizer)
    features = query_states(extract, train, role_attention=args.role_attention)
    head = fit(features, train, args.steps, role_attention=args.role_attention)
    planner = args.output / 'role-plan.pt'
    atomic_save(head.state_dict(), planner)
    head.load_state_dict(torch.load(planner, weights_only=True, map_location='cpu'), strict=True)
    with torch.inference_mode():
        held_features = query_states(extract, [row[1:] for row in hold], role_attention=args.role_attention)
        predicted = head(*held_features).argmax(-1).tolist()
        swapped = head(*(value.flip(1) for value in held_features)).argmax(-1).flip(1).tolist()
        if predicted != swapped:
            raise RuntimeError('Swapping proposed role order changed the version plan')
        results = evaluate_routes(args, runtime, decoder, model, tokenizer, replacements, manifest,
                                  hold, predicted, sources, identity)
    if identity != frozen_digest(runtime.organism, runtime.state) or any(sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('An input or the individual changed')
    store_json(args.output / 'result.json', dict(**results, train_count=len(train), hold_count=len(hold), steps=args.steps,
               role_attention=args.role_attention,
               individual_sha256=identity, head_sha256=sha(planner), sources={str(path): value for path, value in sources.items()},
               seconds=perf_counter() - started))
    print(results['summary'], flush=True)


def evaluate_routes(args, runtime, decoder, model, tokenizer, replacements, manifest, hold, predicted, sources, identity):
    weights = torch.load(args.reader / 'shared.pt', weights_only=True, map_location='cpu')
    composer = VersionReplyActions(model.config.hidden_size, weights['scaffold_tokens'], slots=2,
                                   field_count=1 + len(ROLES), shared_role_pointer=True)
    composer.load_state_dict(weights, strict=True)
    composer.to(model.device).eval().requires_grad_(False)
    embedding = model.get_input_embeddings().weight.detach()
    columns = compress_dimension(embedding.shape[1]).to(embedding.device)
    codebook = F.normalize(embedding.index_select(1, columns).float(), dim=-1)
    binding = sha256(json.dumps((identity, sources[WRITER_SOURCE / 'adapters.pt'], sources[args.checkpoint])).encode()).hexdigest()
    prior = {(row['case'], row['variant']): row['values'] for row in manifest['results']['shared']['rows']}
    results, stats = [], Counter()
    with ExitStack() as stack:
        old_store = stack.enter_context(VersionStore(args.reference.parent / 'versions.sqlite', binding))
        new_store = stack.enter_context(VersionStore(args.reader / 'repeated-roles.sqlite', binding))
        catalogs = old_store.catalog(), new_store.catalog()
        stack.enter_context(using(replacements, 'recurrent'))
        for index, ((scene, question, roles, expected), actions) in enumerate(zip(hold, predicted, strict=True)):
            variant = index % 2
            owner = f'{scene}/{variant}'
            repeated = int(scene.startswith('two-'))
            store = (old_store, new_store)[repeated]
            catalog = catalogs[repeated]
            addresses = {role: MemoryAddress(owner, ROLES[role], role if repeated else '') for role in roles}
            if not set(addresses.values()) <= catalog.keys():
                raise ValueError('The confirmed owner does not have the expected catalog roles')
            values = prior[scene, variant]
            if any(value in question for value in values.values()):
                raise ValueError('A stored fact occurs in the visible question')
            expected_numbers = [resolve_address(catalog, address, VERSIONS[action], NEW_DAY)[0]
                                for address, action in zip(addresses.values(), expected, strict=True) if action != SKIP]
            # A negative or uncertain prediction is never filled in using the test label.
            selected, memory = selected_tokens(store, catalog, codebook, addresses, actions)
            decoded = {role: tokenizer.decode(ids.tolist()) for role, ids in memory.items()}
            exact_plan = tuple(actions) == expected
            exact_records = selected == expected_numbers
            stats.update(plans=exact_plan, records=exact_records, abstained=sum(actions) == 0,
                         abstain_correct=sum(expected) == sum(actions) == 0,
                         false_read=sum(expected) == 0 and bool(selected))
            output = None
            if selected:
                field_ids = torch.tensor(descriptors('shared', roles), device=model.device)
                output = reply(model, tokenizer, composer, question, tuple(memory.get(role) for role in roles),
                               max_new_tokens=48, require_coverage=True, cover_present_slots=True,
                               complete_on_eos=True, field_ids=field_ids)
                expected_text = {role: (values['keep'] if slot else values['new' if expected[slot] == CURRENT else 'old'])
                                 for slot, role in enumerate(roles) if expected[slot] != SKIP}
                blocked = set(values.values()) - set(expected_text.values())
                text = output['answer']
                stats['facts'] += (bool(expected_text) and output['token_ids'][-1:] == [tokenizer.eos_token_id]
                                   and all(value in text for value in expected_text.values())
                                   and all(value not in text for value in blocked))
                stats['generated'] += 1
            results.append(dict(scene=scene, variant=variant, question=question, expected=expected, actions=actions,
                                expected_records=expected_numbers, selected=selected, facts=decoded, output=output))
        audited = evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))['summary']
    if audited != manifest['memory']:
        raise RuntimeError('The routing audit changed addressed memory')
    return dict(rows=results, summary=dict(stats), memory=audited)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('reference', type=Path)
    cli.add_argument('reader', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('layers', nargs=2, type=Path)
    cli.add_argument('--steps', type=int, default=220)
    cli.add_argument('--role-attention', action='store_true')
    args = cli.parse_args()
    if args.steps < 1:
        cli.error('Expected a positive step count')
    main(args)
