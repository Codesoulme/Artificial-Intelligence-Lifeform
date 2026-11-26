"""Compare individual and joint routing into fresh role-labeled version tensors."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path
import re
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_dual_version_reply import reply
from scripts.audit_spora_revision_address_selection import rank_options, recall
from scripts.audit_spora_revision_version_state import (
    MODEL, NEW_DAY, OLD_DAY, WRITER_SOURCE, AutoTokenizer, VersionReplyActions, VersionStore,
    compress_dimension, extend_positions, frozen_digest, load_full_streamed, load_integrated_runtime,
    load_recurrent_replacements, using, write_fresh,
)
from scripts.train_spora_recurrent_counterfactual import sha, store_json
from scripts.train_spora_revision_roles import ROLES, descriptors
from spora.research_ttt_ntp import LowRankFastWeight
from spora.research_version_store import MemoryAddress


SCENES = {
    '手作展': (
        ('展签摆放地点', 'display', '水榭北窗', '榆树长台', r'(?:摆|放)(?:在|到)?{value}'),
        ('撤场收牌人', 'collector', '潘雪青', None, r'{value}(?:负责)?(?:收|取)')),
    '戏曲采录': (
        ('开录时间', 'record_time', '周四中午', '周日晚间', r'(?:录音|开录)(?:时间)?(?:在|是|定在|改在)?{value}|{value}(?:开始)?(?:录音|开录)'),
        ('片头剪辑者', 'editor', '金野桥', None, r'{value}(?:负责)?剪(?:辑)?(?:片头|开头)')),
    '印刷交接': (
        ('校样份数', 'count', '八份', '十五份', r'{value}(?:校样|样张)|(?:校样|样张)(?:共|有|是)?{value}'),
        ('校样接收人', 'recipient', '江望月', None, r'(?:交|送|递)给{value}|{value}(?:负责)?(?:接收|签收)')),
}
REQUESTS = (
    ('same', '手作展', '戏曲采录', '给手作展写一句交班话，展签摆在哪儿，散场后谁收牌？', (0, 1)),
    ('same', '戏曲采录', '印刷交接', '给采录组留条消息，哪位剪片头，几点开始录？', (1, 0)),
    ('same', '印刷交接', '手作展', '请写清这批校样最终交几份，交到谁手里。', (0, 1)),
    ('cross', '手作展', '戏曲采录', '写一条提醒，告诉我手作展在哪摆展签，戏曲组何时开录。', (0, 2)),
    ('cross', '戏曲采录', '印刷交接', '把剪片头的人和校样接收人合在一句交接消息里。', (1, 3)),
    ('cross', '印刷交接', '手作展', '写给值班员，先说手作展散场后谁收牌，再说校样交几份。', (3, 0)),
    ('missing', '手作展', '戏曲采录', '查皮影演出的灯光师和后门钥匙存放点。', ()),
    ('missing', '印刷交接', '手作展', '查读书会的进场时间和报名负责人。', ()),
)


def populate(path, binding, base, tokenizer, columns, writer):
    expected = {}
    with VersionStore(path, binding, create=True) as store:
        for scene, fields in SCENES.items():
            for label, role, first, revision, _ in fields:
                address = MemoryAddress(scene, ROLES[role], role)
                events = [(OLD_DAY, first, f'{scene}的{label}定为{{value}}。')]
                if revision:
                    events.append((NEW_DAY, revision, f'{scene}的{label}更正为{{value}}。'))
                for day, value, fact in events:
                    keys, codes = write_fresh(base, tokenizer, columns, layer=writer['layers'][0],
                        eta=writer['eta'], ridge=writer['ridge'], events={role: ((scene, value),)},
                        facts={role: fact}, minimum_tokens=1)[scene][0]
                    store.append(f'{scene}/{role}/{day}', address, day, day, LowRankFastWeight(keys, codes))
                expected[address] = revision or first
    return expected


def select(decoder, tokenizer, labels, question, names):
    missing = '档案没有对应记录'
    intro = f'用户请求：{question}\n只回答选项字母。'
    individual = [rank_options(decoder, tokenizer, labels[:5],
        f'{intro}\n请求中提到的{ordinal}信息对应哪条档案？', [*names, missing], ())
        for ordinal in ('第一项', '第二项')]
    pairs = list(combinations(range(len(names)), 2))
    joint = rank_options(decoder, tokenizer, labels, f'{intro}\n哪一组档案可以回答这两个问题？',
                         [*(f'{names[i]} 和 {names[j]}' for i, j in pairs), missing], ())
    # A missing choice for either slot leaves the pair unresolved.
    absent = len(names)
    best = max((pair for pair in product(range(absent + 1), repeat=2) if pair[0] != pair[1] or pair[0] == absent),
               key=lambda pair: sum(individual[slot][0][index] for slot, index in enumerate(pair)))
    return {'individual': (() if absent in best else best, individual),
            'joint': ([*pairs, ()][max(range(len(joint[0])), key=joint[0].__getitem__)], joint)}


def score(output, expected, patterns, disallowed, eos):
    text = output['answer']
    ended = output['token_ids'][-1:] == [eos]
    both = ended and all(text.count(value) == 1 for value in expected) and not any(value in text for value in disallowed)
    return dict(ended=ended, both=both,
                roles=both and all(re.search(pattern.format(value=re.escape(value)), text)
                                   for pattern, value in zip(patterns, expected, strict=True)))


@torch.inference_mode()
def main(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    writer_path = WRITER_SOURCE / 'adapters.pt'
    head_path = args.composer / 'shared.pt'
    manifest_path = args.composer / 'result.json'
    layer_paths = [folder / name for folder in args.layers for name in ('attention.pt', 'comparison.json')]
    sources = {path: sha(path) for path in (args.checkpoint, writer_path, head_path, manifest_path, *layer_paths)}
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest['results']['shared']['head_sha256'] != sources[head_path] or manifest['roles'] != ROLES
            or tuple(manifest['roles']) != tuple(ROLES)
            or any(manifest['sources'].get(str(path)) != sources[path]
                   for path in (args.checkpoint, writer_path, *layer_paths))):
        raise ValueError('Frozen role head provenance differs')
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    labels = [tokenizer.encode(letter, add_special_tokens=False) for letter in 'ABCDEFG']
    if any(len(label) != 1 for label in labels) or len({label[0] for label in labels}) != len(labels):
        raise ValueError('Option labels must be distinct single tokens')
    labels = [label[0] for label in labels]
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    base.lm_head.to(base.device)
    identity = frozen_digest(runtime.organism, runtime.state)
    if identity != manifest['individual_sha256']:
        raise ValueError('Loaded a different individual')
    replacements = load_recurrent_replacements(decoder, args.layers, sources[args.checkpoint])
    if set(replacements) != {19, 22}:
        raise ValueError('Expected the two-layer recurrent baseline')
    embedding = base.get_input_embeddings().weight.detach()
    columns = compress_dimension(embedding.shape[1]).to(embedding.device)
    codebook = F.normalize(embedding.index_select(1, columns).float(), dim=-1)
    weights = torch.load(head_path, map_location='cpu', weights_only=True)
    head = VersionReplyActions(base.config.hidden_size, weights['scaffold_tokens'], slots=2,
                               field_count=1 + len(ROLES), shared_role_pointer=True)
    head.load_state_dict(weights, strict=True)
    head.to(base.device).eval().requires_grad_(False)
    writer = torch.load(writer_path, map_location='cpu', weights_only=True)
    binding = sha256(json.dumps((identity, sources[writer_path], sources[args.checkpoint])).encode()).hexdigest()
    database = args.output / 'fresh-versions.sqlite'
    expected = populate(database, binding, base, tokenizer, columns, writer)
    database_hash = sha(database)
    rows, summary = [], {}
    with VersionStore(database, binding) as store, using(replacements, 'recurrent'):
        catalog = store.catalog()
        for group, first, second, question, target in REQUESTS:
            fields = [field for scene in (first, second) for field in SCENES[scene]]
            candidates = [(f'{scene}的{label}', MemoryAddress(scene, ROLES[role], role))
                          for scene in (first, second) for label, role, *_ in SCENES[scene]]
            if any(value in question for _, _, old, new, _ in fields for value in (old, new) if value):
                raise ValueError('The question exposes a stored value')
            selected = select(decoder, tokenizer, labels, question, [label for label, _ in candidates])
            wanted = [expected[candidates[index][1]] for index in target]
            patterns = [fields[index][-1] for index in target]
            disallowed = {value for _, _, old, new, _ in fields for value in (old, new) if value} - set(wanted)
            generations, arms = {}, {}
            for mode, (indexes, scores) in {'given': (target, None), **selected}.items():
                key = tuple(sorted(indexes))
                if indexes and key not in generations:
                    numbers, tokens, decoded = recall(store, catalog, codebook, tokenizer, candidates, key)
                    roles = tuple(candidates[index][1].condition for index in key)
                    field_ids = torch.tensor(descriptors('shared', roles), device=base.device)
                    output = reply(base, tokenizer, head, question, tokens, require_coverage=True,
                                   complete_on_eos=True, max_new_tokens=48, field_ids=field_ids)
                    if decoded != [expected[candidates[index][1]] for index in key]:
                        raise ValueError('A selected address returned the wrong current value')
                    generations[key] = dict(numbers=numbers, values=decoded, output=output)
                generated = generations.get(key, {})
                checks = score(generated['output'], wanted, patterns, disallowed, tokenizer.eos_token_id) if target and generated else {}
                hit = set(indexes) == set(target)
                summary.setdefault(group, {}).setdefault(mode, Counter()).update(
                    total=1, route=hit, reply=checks.get('both', False), roles=checks.get('roles', False),
                    withheld=not indexes)
                arms[mode] = dict(selected=indexes, scores=scores, checks=checks, **generated)
            rows.append(dict(group=group, question=question, candidates=[label for label, _ in candidates],
                             expected=target, arms=arms))
            print(group, question, {mode: arm['selected'] for mode, arm in arms.items()}, flush=True)
    if (frozen_digest(runtime.organism, runtime.state) != identity or sha(database) != database_hash
            or any(sha(path) != digest for path, digest in sources.items())):
        raise RuntimeError('A protected input changed in the comparison')
    store_json(args.output / 'result.json', dict(rows=rows, summary=summary, seconds=perf_counter() - started,
               individual_sha256=identity, database_sha256=database_hash,
               sources={str(path): digest for path, digest in sources.items()}))
    print(summary, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('composer', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('layers', type=Path, nargs=2)
    main(cli.parse_args())
