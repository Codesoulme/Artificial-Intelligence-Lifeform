"""Locate which parts of the public field improvement transfer to private lookups."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from zipfile import ZipFile

import torch
from torch import nn

from scripts import audit_spora_crosswoz_field_transfer as transfer
from scripts import audit_spora_dense_joint_address as dense
from scripts import audit_spora_spoken_query_plan as plan
from scripts.audit_spora_crosswoz_positive_fields import PINNED, requested
from scripts.audit_spora_natural_lookup_cli import digest
from scripts.spora_version_chat import FrozenTurnFeatures, VersionStore, artifacts, base, local_model
from scripts.train_spora_recurrent_counterfactual import store_json


@torch.inference_mode()
def address_components(head, inputs, cases, catalog):
    logits = head(inputs[0], inputs[1], transfer.masks_for(inputs, catalog))
    rows, summary = [], Counter()
    for result, (_, gold), owners in zip(logits, cases, inputs[2], strict=True):
        if gold is None:
            continue
        target = dense.index_of(gold, owners)
        owner, field = divmod(target, dense.WIDTH)
        slots = result[:-1].reshape(2, dense.WIDTH)
        field_match = int(slots[owner].argmax()) == field
        owner_match = int(slots[:, field].argmax()) == owner
        summary.update(field_given_owner=field_match, owner_given_field=owner_match,
                       joint_given_read=int(result[:-1].argmax()) == target,
                       raw_read=int(result.argmax()) != dense.NONE)
        rows.append({'expected': [gold.name, gold.field], 'field_given_owner': field_match,
                     'owner_given_field': owner_match, 'scores': result.tolist()})
    return {'counts': summary, 'rows': rows}


def test_rows(path, manifest):
    with ZipFile(path) as arc:
        data = json.loads(arc.read('test.json'))
    with_history, direct = [], []
    for name, turn, expected in manifest:
        messages = data[name]['messages']
        if messages[turn]['role'] != 'usr':
            raise ValueError('Expected an official user turn')
        prompt = transfer.context(messages, turn)
        if transfer.fingerprint(prompt) != expected:
            raise ValueError('Test prompt differs from training report')
        user = f"用户：{messages[turn]['content'].strip()}"
        labels = tuple(f'{domain}/{slot}' for domain, slot in requested(messages[turn]))
        with_history.append((name, turn, prompt, labels))
        direct.append((name, turn, user, labels))
    return with_history, direct


def main(args):
    started = perf_counter()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    source = args.transfer / 'result.json'
    reference = json.loads(source.read_text(encoding='utf-8'))
    originals = {Path(path).resolve(): value for path, value in reference['sources'].items()}
    test = args.corpus / 'test.json.zip'
    archive = args.profile / 'profile' / 'versions.sqlite3'
    paths = (test, archive, args.recurrent[0], *(path / 'attention.pt' for path in args.recurrent[1:]))
    bound = {path: digest(path) for path in paths}
    if bound[test] != PINNED['test'] or any(bound[path] != originals[path.resolve()] for path in paths):
        raise ValueError('Reload requires the same held set, archive, and backbone')
    bound.update({source: digest(source), Path(__file__): digest(Path(__file__))})
    labels = reference['labels']
    public_history, public_direct = test_rows(test, reference['corpus']['test'])
    targets = transfer.multilabel(public_history, labels)
    _, _, _, binding = artifacts(args.reader, args.normalizer)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    private = {'previous': transfer.make_cases(dense.HELD), 'fresh': transfer.make_cases(transfer.FRESH)}
    results = {}
    with local_model(args.recurrent) as model, VersionStore(archive, binding) as store:
        extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=plan.DEPTH)
        inputs = {name: dense.align(extract, [text for text, _ in cases], dense.NAMES) for name, cases in private.items()}
        users = {name: plan.vectors(extract, [row[2] for row in items], (), 'user-span')[0]
                 for name, items in (('context', public_history), ('user_only', public_direct))}
        catalog = store.catalog()
        for mode, arm in reference['arms'].items():
            path = args.transfer / f'{mode}.pt'
            bound[path] = digest(path)
            if bound[path] != arm['sha256']:
                raise ValueError('Unexpected transfer checkpoint')
            weight = torch.load(path, map_location='cpu', weights_only=True)
            pointer = dense.JointPointer(model.config.hidden_size).eval()
            pointer.load_state_dict(weight['pointer'])
            private_scores = {name: address_components(pointer, inputs[name], cases, catalog) for name, cases in private.items()}
            public_scores = None
            if arm['public'] is not None:
                head = nn.Linear(pointer.output.in_features, len(labels)).eval()
                head.load_state_dict(weight['public'])
                with torch.inference_mode():
                    public_scores = {name: transfer.public_score(head(pointer.shared(state)).sigmoid(), targets,
                                                                   arm['public']['threshold']) for name, state in users.items()}
                if public_scores['context'] != arm['public']['test']:
                    raise ValueError('Reload differs on the unchanged public test input')
            results[mode] = {'public': public_scores, 'personal': private_scores}
    if any(digest(path) != expected for path, expected in bound.items()):
        raise ValueError('A bound artifact changed')
    store_json(args.output, {'arms': results, 'sources': {str(p): sha for p, sha in bound.items()},
                             'seconds': perf_counter() - started})
    print({name: {'public': row['public'], 'personal': {key: val['counts'] for key, val in row['personal'].items()}}
           for name, row in results.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('transfer', type=Path)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('profile', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reader', type=Path, default=Path('runs/spora-dual-version-reply-v1'))
    parser.add_argument('--normalizer', type=Path, default=Path('runs/spora-version-normalizer-v1'))
    parser.add_argument('--recurrent', type=Path, nargs=3, required=True)
    main(parser.parse_args())
