"""Distinguish address ranking from abstention using the same dense catalog."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch

from scripts import audit_spora_dense_joint_address as dense
from scripts import audit_spora_spoken_query_plan as plan
from scripts.audit_spora_natural_lookup_cli import digest
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.spora_version_chat import FrozenTurnFeatures, VersionStore, artifacts, base, local_model
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.research_version_store import MemoryAddress


def address_json(address):
    return None if address is None else [address.name, address.field]


@torch.inference_mode()
def numbered_choice(model, tokenizer, question, candidates):
    by_number = {token: address for token, address in zip('12345678', candidates, strict=False)}
    listing = '\n'.join(f'{token} {address.name}的{plan.FIELDS[address.field]}' for token, address in by_number.items())
    prompt = ('请判断用户要查哪项已登记记录。目录只含姓名和事项。请看清用户有没有取消某一项、改问另一项。'
              '用户请你查阅且目录能唯一确定时，输出该项编号。目标不在目录、没有查阅要求或目标不明确时，输出0。'
              f'只填一个数字。\n目录\n{listing}\n用户原话\n{question}')
    tokens = ['0', *by_number]
    pieces = [tokenizer.encode(token, add_special_tokens=False) for token in tokens]
    if any(len(piece) != 1 for piece in pieces) or len({piece[0] for piece in pieces}) != len(tokens):
        raise ValueError('Catalog choices must be distinct single tokens')
    inputs = closed_think_inputs(tokenizer, prompt, model.device)['input_ids']
    suffix = tokenizer.encode('编号：', add_special_tokens=False)
    inputs = torch.cat((inputs, torch.tensor([suffix], device=model.device)), dim=1)
    logits = model(input_ids=inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    distribution = logits[[piece[0] for piece in pieces]].softmax(-1)
    selected = tokens[int(distribution.argmax())]
    return by_number.get(selected), dict(zip(tokens, distribution.tolist(), strict=True))


@torch.inference_mode()
def evaluate(model, tokenizer, extract, catalog, heads, old):
    cases = [(text, MemoryAddress(dense.NAMES[owner], field) if owner is not None else None)
             for text, owner, field in dense.HELD]
    states, spans, owners = dense.align(extract, [text for text, _ in cases], dense.NAMES)
    masks = torch.stack([dense.mask_for(names, catalog) for names in owners])
    ranks = {mode: head(states, spans if mode == 'local' else torch.zeros_like(spans), masks)[:, :-1].argmax(-1).tolist()
             for mode, head in heads.items()}
    _, owners_logits, fields_logits, _ = old(states)
    independent = list(zip(owners_logits.argmax(-1).tolist(), fields_logits.argmax(-1).tolist(), strict=True))
    rows, scores = [], Counter()
    for index, ((question, truth), names) in enumerate(zip(cases, owners, strict=True)):
        allowed = sorted((entry for entry in catalog if entry.name in names), key=lambda value: (value.name, value.field))
        if len(allowed) != dense.NONE:
            raise ValueError('Expected two mentioned people with all four fields each')
        owner, field = independent[index]
        picks = {mode: dense.addressed(slot[index], names) for mode, slot in ranks.items()}
        picks['old'] = dense.addressed(owner * dense.WIDTH + field, names)
        if truth is not None:
            scores.update({mode + '_given_read': value == truth for mode, value in picks.items()})
        views = {}
        for name, items, expected in (('full', allowed, truth), ('target_removed', [x for x in allowed if x != truth], None)):
            if name == 'target_removed' and truth is None:
                continue
            forward, p1 = numbered_choice(model, tokenizer, question, items)
            reverse, p2 = numbered_choice(model, tokenizer, question, items[::-1])
            category = ('target_removed' if name == 'target_removed' else 'present' if truth is not None else 'cancel')
            scores.update({category + '_forward': forward == expected, category + '_reverse': reverse == expected,
                           category + '_both': forward == reverse == expected, category + '_changed': forward != reverse})
            views[name] = {'forward': address_json(forward), 'reverse': address_json(reverse),
                           'probabilities_forward': p1, 'probabilities_reverse': p2}
        rows.append(dict(question=question, correct=address_json(truth), given_read={name: address_json(value)
                         for name, value in picks.items()}, full_decoder=views))
    return rows, scores


def main(args):
    started = perf_counter()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    previous = json.loads((args.source / 'result.json').read_text(encoding='utf-8'))
    known = {Path(path).resolve(): value for path, value in previous['sources'].items()}
    database = args.source / 'profile' / 'versions.sqlite3'
    old_file, = (path for path in known if path.name == 'public.pt')
    inputs = {database: digest(database), old_file: digest(old_file), Path(__file__): digest(Path(__file__))}
    if inputs[database] != known[database.resolve()] or inputs[old_file] != known[old_file]:
        raise ValueError('The backbone probe or dense profile changed')
    for path in (args.recurrent[0], *(folder / 'attention.pt' for folder in args.recurrent[1:])):
        inputs[path] = digest(path)
        if inputs[path] != known[path.resolve()]:
            raise ValueError('Diagnostic must use the same recurrent weights')
    _, _, _, binding = artifacts(args.reader, args.normalizer)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    with local_model(args.recurrent) as model, VersionStore(database, binding) as store:
        width = model.config.hidden_size
        heads = {}
        for mode in ('global', 'local'):
            path = args.source / f'{mode}.pt'
            inputs[path] = digest(path)
            if inputs[path] != previous['sha256'][mode]:
                raise ValueError('A learned pointer changed')
            head = heads[mode] = dense.JointPointer(width).eval()
            head.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        old = plan.QueryPlanProbe(width).eval()
        old.load_state_dict(torch.load(old_file, map_location='cpu', weights_only=True))
        extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=plan.DEPTH)
        rows, scores = evaluate(model, tokenizer, extract, store.catalog(), heads, old)
    if any(digest(path) != expected for path, expected in inputs.items()):
        raise ValueError('Diagnostic changed a bound input')
    store_json(args.output, {'metrics': scores, 'rows': rows, 'seconds': perf_counter() - started,
                             'sources': {str(path): value for path, value in inputs.items()}})
    print(scores, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reader', type=Path, default=Path('runs/spora-dual-version-reply-v1'))
    parser.add_argument('--normalizer', type=Path, default=Path('runs/spora-version-normalizer-v1'))
    parser.add_argument('--recurrent', type=Path, nargs=3, required=True)
    main(parser.parse_args())
