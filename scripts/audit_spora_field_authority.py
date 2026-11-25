"""Measure field-level exclusions and verify selected values in a sealed profile."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from scripts.audit_spora_composition_addresses import resolve_current
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_dual_version_reply import FIELDS
from scripts.audit_spora_fast_weight_lexical_write import base, compress_dimension
from scripts.audit_spora_field_evidence import (
    BINDING_PREFIX, check_explicit, explicit_route, field_features, field_head,
    profile_values, read_reply, semantic_route,
)
from scripts.audit_spora_fresh_dialogue import digest
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_composition_address_cases import HELD as PRIOR
from scripts.spora_field_authority_cases import HELD, training
from scripts.spora_field_evidence_cases import FRESH as PRIOR_FRESH
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_version_store import MemoryAddress, VersionStore
from spora.research_versioned_memory_pointer import VersionReplyActions


UNMENTIONED, EXCLUDED, REQUESTED = range(3)
SEED = 84711


def labels_for(rows):
    if any(set(row.fields) & set(row.excluded) for row in rows):
        raise ValueError('A field cannot be both requested and excluded')
    return torch.tensor([[REQUESTED if field in row.fields else EXCLUDED if field in row.excluded else UNMENTIONED
                          for field in FIELDS] for row in rows])


def train(head, features, labels, *, explicit_exclusion):
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    generator = torch.Generator().manual_seed(SEED)
    for epoch in range(12):
        losses = []
        for subset in torch.randperm(len(labels), generator=generator).split(24):
            prediction = head(features[subset])
            target = labels[subset]
            if explicit_exclusion:
                loss = F.cross_entropy(prediction.flatten(0, 1), target.flatten())
            else:
                loss = F.binary_cross_entropy_with_logits(prediction.squeeze(-1), (target == REQUESTED).float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch in (0, 5, 11):
            print('train', explicit_exclusion, epoch + 1, sum(losses) / len(losses), flush=True)


@torch.inference_mode()
def evaluate(rows, vectors, heads, model, tokenizer, store, codebook, reader, archived=None):
    catalog = store.catalog()
    facts = profile_values(catalog)
    addresses = {entry.number: address for address, versions in catalog.items() for entry in versions}
    probabilities = {
        'previous': heads['previous'](vectors).sigmoid().squeeze(-1).tolist(),
        'matched_binary': heads['matched_binary'](vectors).sigmoid().squeeze(-1).tolist(),
        'field_authority': (heads['field_authority'](vectors).argmax(-1) == REQUESTED).float().tolist(),
    }
    stats = {mode: Counter() for mode in probabilities}
    observed = []
    cache = {}
    for index, row in enumerate(rows):
        if any(value in row.question for value in facts.values()):
            raise ValueError('A registered value appeared in the request')
        expected = resolve_current(catalog, [MemoryAddress(row.owner, field) for field in row.fields])
        excluded = set(getattr(row, 'excluded', ()))
        explicit = explicit_route(row.question, catalog)
        arms = {}
        for mode, scores in probabilities.items():
            route = explicit if explicit is not None else semantic_route(row, scores[index], catalog)
            if archived is not None and mode == 'previous':
                recorded = archived[index]
                if row.question != recorded['request']['question'] or list(route) != recorded['decisions'][-1]['route']:
                    raise ValueError('The previous selector did not reproduce its recorded routing')
            wrong = set(route) - set(expected)
            missing = set(expected) - set(route)
            key = (row.question, route)
            if key not in cache:
                cache[key] = read_reply(model, tokenizer, store, codebook, reader, row.question, route, expected, facts)
            output = cache[key]
            stats[mode].update(cases=1, positive=bool(expected), exact_positive=bool(expected) and not wrong and not missing,
                               false_read=not expected and bool(route), extra_fields=len(wrong), missing_fields=len(missing),
                               excluded_reads=sum(addresses[number].field in excluded for number in route),
                               value_pass=output['value_pass'])
            arms[mode] = {'route': route, 'extra': sorted(wrong), 'missing': sorted(missing), **output}
        observed.append({'request': asdict(row), 'expected': expected, 'arms': arms})
    return {'summary': stats, 'rows': observed}


def main(args):
    start = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / ('replay.json' if args.verify else 'result.json')
    if destination.exists():
        raise FileExistsError(destination)
    archive = args.records / 'versions.sqlite3'
    readout = args.reader / 'head.pt'
    previous = args.previous / 'field.pt'
    records_report, reader_report, previous_report = (
        folder / 'result.json' for folder in (args.records, args.reader, args.previous)
    )
    model_path = base.MODEL / 'model.safetensors'
    paths = (archive, readout, previous, records_report, reader_report, previous_report, model_path,
             base.MODEL / 'tokenizer.json', Path(__file__), Path(__file__).with_name('spora_field_authority_cases.py'))
    sources = {str(path): digest(path) for path in paths}
    records, reader_audit, previous_audit = [
        json.loads(path.read_text(encoding='utf-8')) for path in (records_report, reader_report, previous_report)
    ]
    expected = (records['database_sha256'], reader_audit['head_sha256'], previous_audit['head_sha256'])
    if tuple(sources[str(path)] for path in (archive, readout, previous)) != expected or sources[str(model_path)] != base.WEIGHT_SHA:
        raise ValueError('The profile, readout, reference or backbone differs from its audit')
    writer = next(value for path, value in records['sources'].items() if Path(path).name == 'adapters.pt')
    model = load_full_streamed()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer)
    rows = tuple(training())
    seen = {row.question for row in rows}
    if len(seen) != len(rows) or seen.intersection(row.question for row in (*PRIOR, *PRIOR_FRESH, *HELD)):
        raise ValueError('Training contains duplicate or evaluation requests')
    features = None if args.verify else field_features(extract, rows)
    labels = labels_for(rows)
    heads = {}
    weights = {}
    for mode in ('previous', 'matched_binary', 'field_authority'):
        torch.manual_seed(SEED)
        head = field_head(model.config.hidden_size)
        typed = mode == 'field_authority'
        if typed:
            head[-1] = nn.Linear(64, 3)
        path = previous if mode == 'previous' else args.output / f'{mode}.pt'
        if args.verify or mode == 'previous':
            head.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        else:
            train(head, features, labels, explicit_exclusion=typed)
            atomic_save(head.state_dict(), path)
        heads[mode] = head.eval()
        weights[mode] = digest(path)
    payload = torch.load(readout, map_location='cpu', weights_only=True)
    reader = VersionReplyActions(model.config.hidden_size, payload['scaffold_tokens'], slots=2)
    reader.load_state_dict(payload)
    reader.eval().to(model.device)
    embeddings = model.get_input_embeddings().weight.detach()
    columns = compress_dimension(embeddings.shape[1]).to(embeddings.device)
    codebook = F.normalize(embeddings.index_select(1, columns).float(), dim=-1)
    result = {'sources': sources, 'weights_sha256': weights, 'training_count': len(rows), 'splits': {}}
    with VersionStore(archive, BINDING_PREFIX + writer + ':' + base.WEIGHT_SHA) as store:
        for name, group in (('prior', PRIOR), ('fresh', PRIOR_FRESH), ('exclusions', HELD)):
            vectors = field_features(extract, group)
            result['splits'][name] = evaluate(group, vectors, heads, model, tokenizer, store, codebook, reader,
                                               archived=previous_audit['rows'].get(name))
            print(name, result['splits'][name]['summary'], flush=True)
        result['explicit'] = check_explicit(model, tokenizer, store, codebook, reader)
    if not all(row['passes'] for row in result['explicit']):
        raise ValueError('An explicit historical-version regression failed')
    if any(digest(path) != sources[str(path)] for path in paths):
        raise ValueError('A sealed input changed')
    if args.verify:
        prior = json.loads((args.output / 'result.json').read_text(encoding='utf-8'))
        if any(json.loads(json.dumps(item)) != prior[key] for key, item in result.items()):
            raise ValueError('Independent replay changed records or replies')
    store_json(destination, {**result, 'seconds': perf_counter() - start,
                             'peak_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('records', 'reader', 'previous', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
