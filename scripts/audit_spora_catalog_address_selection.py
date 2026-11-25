"""Measure the effect of restricting address selection to existing metadata."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch

from scripts import audit_spora_spoken_query_plan as plan
from scripts.audit_spora_natural_lookup_cli import digest
from scripts.audit_spora_spoken_address_authority import held_data
from scripts.spora_version_chat import FrozenTurnFeatures, VersionStore, artifacts, base, local_model, spoken_lookup_planner
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.research_version_store import MemoryAddress


def restriction(names, owners, fields, allowed):
    ranked = [(MemoryAddress(name, field), float(owners[i] * fields[j]))
              for i, name in enumerate(names) for j, field in enumerate(plan.FIELDS_ORDER)
              if MemoryAddress(name, field) in allowed]
    return max(ranked, key=lambda item: item[1], default=(None, 0.))


@torch.inference_mode()
def audit(head, extract, catalog):
    positives = [example for example in held_data() if example.address is not None]
    vectors, names = plan.vectors(extract, [row.text for row in positives], {key.name for key in catalog}, 'user-span')
    _, people, fields, _ = [tensor.softmax(-1) for tensor in head(vectors)]
    options = (0., .1, .2, .35, .5, .7)
    counter = Counter()
    cutoffs = {level: Counter() for level in options}
    rows = []
    for row, candidates, names_p, fields_p in zip(positives, names, people, fields, strict=True):
        if row.address not in catalog:
            raise ValueError('Expected address is absent from the test catalog')
        owner_index, field_index = int(names_p.argmax()), int(fields_p.argmax())
        unconstrained = (MemoryAddress(candidates[owner_index], plan.FIELDS_ORDER[field_index])
                         if owner_index < len(candidates) else None)
        allowed, probability = restriction(candidates, names_p, fields_p, catalog)
        omitted, omitted_probability = restriction(candidates, names_p, fields_p, set(catalog) - {row.address})
        dense = {MemoryAddress(owner, field) for owner in candidates for field in plan.FIELDS_ORDER}
        enriched, _ = restriction(candidates, names_p, fields_p, dense)
        counter.update(independent_no_address=unconstrained not in catalog, independent_correct=unconstrained == row.address,
                       sparse_correct=allowed == row.address, enriched_correct=enriched == row.address)
        for threshold, stats in cutoffs.items():
            accepted = allowed is not None and probability >= threshold
            stats.update(correct=accepted and allowed == row.address, wrong=accepted and allowed != row.address,
                         substitute=omitted is not None and omitted_probability >= threshold)

        def slot(address):
            return [address.name, address.field] if address is not None else None

        rows.append(dict(question=row.text, expected=slot(row.address), independent=slot(unconstrained),
                         available=slot(allowed), available_probability=probability, dense=slot(enriched),
                         unavailable_target=slot(omitted), unavailable_target_probability=omitted_probability))
    return rows, counter, cutoffs


def main(args):
    started = perf_counter()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    report = json.loads((args.profile / 'result.json').read_text(encoding='utf-8'))
    archive = args.profile / 'profile' / 'versions.sqlite3'
    checkpoint = args.previous / 'public.pt'
    _, _, _, binding = artifacts(args.reader, args.normalizer)
    audited = {Path(path).resolve(): value for path, value in report['sources'].items()}
    bound = {p: digest(p) for p in (archive, checkpoint, Path(__file__))}
    if bound[archive] != audited[archive.resolve()]:
        raise ValueError('The isolated memory metadata changed')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    with local_model(args.recurrent) as model, VersionStore(archive, binding) as store:
        spoken_lookup_planner(args.previous, args.recurrent, model, tokenizer)
        head = plan.QueryPlanProbe(model.config.hidden_size)
        head.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=plan.DEPTH)
        rows, counts, cutoffs = audit(head.eval(), extract, store.catalog())
    if any(digest(path) != value for path, value in bound.items()):
        raise ValueError('A source changed during the catalog audit')
    store_json(args.output, dict(rows=rows, counts=counts, cutoffs=cutoffs,
                                 sources={str(p): h for p, h in bound.items()}, seconds=perf_counter() - started))
    print(counts, cutoffs, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path)
    parser.add_argument('previous', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reader', type=Path, default=Path('runs/spora-dual-version-reply-v1'))
    parser.add_argument('--normalizer', type=Path, default=Path('runs/spora-version-normalizer-v1'))
    parser.add_argument('--recurrent', type=Path, nargs=3, required=True)
    main(parser.parse_args())
