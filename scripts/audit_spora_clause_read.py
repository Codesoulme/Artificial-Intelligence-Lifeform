"""Select memory using field-conditioned clause evidence and replay version values."""

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
from scripts.audit_spora_fast_weight_lexical_write import base, compress_dimension
from scripts.audit_spora_field_evidence import (
    BINDING_PREFIX, DESCRIPTIONS, FIELDS, check_explicit, explicit_route, field_features,
    field_head, profile_values, read_reply, semantic_route,
)
from scripts.audit_spora_fresh_dialogue import digest
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_clause_read_cases import EXCLUDE, HELD, READ, SKIP, clauses, training
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_version_store import MemoryAddress, VersionStore
from spora.research_versioned_memory_pointer import VersionReplyActions


SEED = 38621


def features_for(extract, rows, names):
    """Compute one causal representation per field and available clause boundary."""
    bounds = [clauses(row.question) for row in rows]
    if any(not spans for spans in bounds):
        raise ValueError('Every request needs a nonempty clause')
    prepared = []
    for index, row in enumerate(rows):
        candidates = sorted(name for name in names if name in row.question)
        owner = candidates[0] if len(candidates) == 1 else ''
        for column, field in enumerate(FIELDS):
            prefix = f'候选记录是{owner}的{DESCRIPTIONS[field]}。请看用户的要求：'
            ends = [len(prefix) + end - 1 for _, end in bounds[index]]
            prepared.append((index, column, ends, prefix + row.question))
    steps = max(map(len, bounds))
    vectors = None
    for start in range(0, len(prepared), 24):
        batch = prepared[start:start + 24]
        hidden = extract([text for _, _, _, text in batch])
        if vectors is None:
            vectors = hidden.new_zeros((len(rows), len(FIELDS), steps, hidden.shape[-1]))
        for i, (row, field, ends, _) in enumerate(batch):
            vectors[row, field, :len(ends)] = hidden[i, ends]
    return vectors.float(), bounds


def targets_for(rows, bounds, steps):
    labels = torch.full((len(rows), len(FIELDS), steps), -100, dtype=torch.long)
    resolved = torch.zeros(len(rows), len(FIELDS), dtype=torch.float32)
    for i, (row, spans) in enumerate(zip(rows, bounds, strict=True)):
        labels[i, :, :len(spans)] = SKIP
        for index, field, label in row.clauses:
            if index >= len(spans) or labels[i, tuple(FIELDS).index(field), index] != SKIP:
                raise ValueError('Invalid or repeated clause evidence')
            labels[i, tuple(FIELDS).index(field), index] = label
        for column, field in enumerate(FIELDS):
            resolutions = [(index, label) for index, name, label in row.clauses if name == field]
            should_read = bool(resolutions and max(resolutions)[1] == READ)
            if should_read != (field in row.fields):
                raise ValueError('Clause labels disagree with the final request')
            resolved[i, column] = should_read
    return labels, resolved


def fit(head, features, bounds, targets, *, local):
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    rng = torch.Generator().manual_seed(SEED)
    weights = torch.tensor([1., 2., 2.])
    last = torch.tensor([len(spans) - 1 for spans in bounds])
    for epoch in range(12):
        losses = []
        for batch in torch.randperm(len(features), generator=rng).split(24):
            if local:
                scores = head(features[batch])
                loss = F.cross_entropy(scores.flatten(0, 2), targets[batch].flatten(), weight=weights)
            else:
                tails = features[batch, :, last[batch]]
                loss = F.binary_cross_entropy_with_logits(head(tails).squeeze(-1), targets[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch in (0, 5, 11):
            print('clause' if local else 'whole', epoch + 1, sum(losses) / len(losses), flush=True)


def read_evidence(question, spans, choices):
    records = {}
    for field, labels in zip(FIELDS, choices, strict=True):
        for i, (bounds, evidence) in enumerate(zip(spans, labels[:len(spans)], strict=True)):
            if evidence != SKIP:
                start, end = bounds
                records[field] = {'index': i, 'range': [start, end], 'clause': question[start:end], 'decision': evidence}
    return records


@torch.inference_mode()
def evaluate(rows, extract, heads, model, tokenizer, store, codebook, reader):
    catalog = store.catalog()
    names = {a.name for a in catalog}
    registered = profile_values(catalog)
    vectors, bounds = features_for(extract, rows, names)
    targets_for(rows, bounds, vectors.shape[2])
    last = torch.tensor([len(spans) - 1 for spans in bounds])
    learned = heads['whole'](vectors[torch.arange(len(rows)), :, last]).squeeze(-1).sigmoid().tolist()
    choices = heads['clause'](vectors).argmax(-1).tolist()
    prior = heads['previous'](field_features(extract, rows)).squeeze(-1).sigmoid().tolist()
    address_by_id = {entry.number: address for address, entries in catalog.items() for entry in entries}
    summaries = {mode: Counter() for mode in heads}
    outputs = []
    for row, spans, baseline, whole, local in zip(rows, bounds, prior, learned, choices, strict=True):
        if any(value in row.question for value in registered.values()):
            raise ValueError('A registered value appeared in the request')
        expected = resolve_current(catalog, [MemoryAddress(row.owner, field) for field in row.fields])
        evidence = read_evidence(row.question, spans, local)
        predicted = [int(evidence.get(field, {}).get('decision') == READ) for field in FIELDS]
        explicit = explicit_route(row.question, catalog)
        gold = {field: (i, label) for i, field, label in row.clauses}
        reply_cache, arms = {}, {}
        for mode, decisions in (('previous', baseline), ('whole', whole), ('clause', predicted)):
            route = explicit if explicit is not None else semantic_route(row, decisions, catalog)
            if route not in reply_cache:
                reply_cache[route] = read_reply(model, tokenizer, store, codebook, reader, row.question, route,
                                                expected, registered)
            reply = reply_cache[route]
            fields = [address_by_id[number].field for number in route]
            exact = set(route) == set(expected)
            grounded = sum(gold.get(field) == (evidence.get(field, {}).get('index'), READ) for field in fields)
            summaries[mode].update(
                exact_positive=bool(expected) and exact, value_pass=reply['value_pass'],
                false_read=not expected and bool(route), extra=len(set(route) - set(expected)),
                missing=len(set(expected) - set(route)),
                excluded=sum(gold.get(field, (None, SKIP))[1] == EXCLUDE for field in fields),
            )
            arms[mode] = {'route': route, **reply}
            if mode == 'clause':
                summaries[mode].update(grounded_reads=grounded, all_reads=len(fields))
        outputs.append({'request': asdict(row), 'expected': expected, 'evidence': evidence, 'arms': arms})
    return summaries, outputs


def main(args):
    start = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / ('replay.json' if args.verify else 'result.json')
    if destination.exists():
        raise FileExistsError(destination)
    database = args.records / 'versions.sqlite3'
    reader_file = args.reader / 'head.pt'
    baseline_file = args.previous / 'matched_binary.pt'
    reports = tuple(directory / 'result.json' for directory in (args.records, args.reader, args.previous))
    records, readout, baseline = [json.loads(path.read_text(encoding='utf-8')) for path in reports]
    model_path = base.MODEL / 'model.safetensors'
    paths = (database, reader_file, baseline_file, model_path, base.MODEL / 'tokenizer.json', *reports,
             Path(__file__), Path(__file__).with_name('spora_clause_read_cases.py'))
    sources = {str(path): digest(path) for path in paths}
    expected = (records['database_sha256'], readout['head_sha256'],
                baseline['weights_sha256']['matched_binary'], base.WEIGHT_SHA)
    if tuple(sources[str(path)] for path in paths[:4]) != expected:
        raise ValueError('Unexpected versions of the store, readout, baseline or backbone')
    writer = next(value for path, value in records['sources'].items() if Path(path).name == 'adapters.pt')
    model = load_full_streamed()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer)
    train = tuple(dict.fromkeys(training()))
    if {row.question for row in train} & {row.question for row in HELD}:
        raise ValueError('Training contains a held request')
    if not args.verify:
        features, bounds = features_for(extract, train, {row.owner for row in train})
        local_labels, whole_labels = targets_for(train, bounds, features.shape[2])
    heads, weights = {}, {}
    for mode in ('previous', 'whole', 'clause'):
        torch.manual_seed(SEED)
        head = field_head(model.config.hidden_size)
        local = mode == 'clause'
        if local:
            head[-1] = nn.Linear(64, 3)
        path = baseline_file if mode == 'previous' else args.output / f'{mode}.pt'
        if args.verify or mode == 'previous':
            head.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        else:
            fit(head, features, bounds, local_labels if local else whole_labels, local=local)
            atomic_save(head.state_dict(), path)
        heads[mode], weights[mode] = head.eval(), digest(path)
    payload = torch.load(reader_file, map_location='cpu', weights_only=True)
    reader = VersionReplyActions(model.config.hidden_size, payload['scaffold_tokens'], slots=2)
    reader.load_state_dict(payload)
    reader.eval().to(model.device)
    embeddings = model.get_input_embeddings().weight.detach()
    columns = compress_dimension(embeddings.shape[1]).to(embeddings.device)
    codebook = F.normalize(embeddings.index_select(1, columns).float(), dim=-1)
    with VersionStore(database, BINDING_PREFIX + writer + ':' + base.WEIGHT_SHA) as store:
        summary, rows = evaluate(HELD, extract, heads, model, tokenizer, store, codebook, reader)
        explicit = check_explicit(model, tokenizer, store, codebook, reader)
    if not all(item['passes'] for item in explicit) or any(digest(path) != sources[str(path)] for path in paths):
        raise ValueError('An explicit version failed or a sealed input changed')
    result = {'sources': sources, 'heads': weights, 'summary': summary, 'rows': rows, 'explicit': explicit,
              'training_count': len(train)}
    if args.verify:
        previous = json.loads((args.output / 'result.json').read_text(encoding='utf-8'))
        if any(json.loads(json.dumps(value)) != previous[key] for key, value in result.items()):
            raise ValueError('The reloaded experiment changed its decisions or replies')
    store_json(destination, {**result, 'seconds': perf_counter() - start,
                             'peak_bytes': torch.cuda.max_memory_allocated()})
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('records', 'reader', 'previous', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
