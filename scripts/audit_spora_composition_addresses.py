"""Learn which existing records an ordinary Chinese request calls for."""

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_dual_version_reply import FIELDS
from scripts.audit_spora_dual_version_routing import reply_to_route
from scripts.audit_spora_fast_weight_lexical_write import base, compress_dimension
from scripts.audit_spora_fresh_dialogue import digest
from scripts.audit_spora_typed_composition import BINDING_PREFIX
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_composition_address_cases import HELD, TRAIN_NAMES, train_cases
from scripts.spora_memory_composition_cases import HELD as MEMORY_CASES
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_multi_version_route import resolve_address
from spora.research_version_dialogue_scope import dialogue_targets
from spora.research_version_store import MemoryAddress, VersionStore
from spora.research_version_timeline import VersionQuery
from spora.research_versioned_memory_pointer import VersionReplyActions


SEED = 84711
TODAY = date(2026, 11, 25)


def features_for(extract, requests):
    texts = [r.question + '\n完成这条请求需要查阅哪些已经登记的事项？' for r in requests]
    vectors = []
    for start in range(0, len(texts), 24):
        batch = texts[start:start + 24]
        hidden = extract(batch)
        vectors.extend(hidden[i, len(text) - 1].clone() for i, text in enumerate(batch))
    return torch.stack(vectors).float()


def make_head(width):
    return nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Linear(64, len(FIELDS)))


def fit(head, requests, vectors):
    labels = torch.tensor([[field in item.fields for field in FIELDS] for item in requests], dtype=torch.float32)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    rng = torch.Generator().manual_seed(SEED)
    for epoch in range(12):
        losses = []
        for batch in torch.randperm(len(requests), generator=rng).split(24):
            scores = head(vectors[batch])
            loss = F.binary_cross_entropy_with_logits(scores, labels[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch in (0, 5, 11):
            print('addresses', epoch + 1, sum(losses) / len(losses), flush=True)


def addresses_for(question, fields, catalog):
    owners = {address.name for address in catalog if address.name in question}
    if len(owners) != 1 or not 1 <= len(fields) <= 2:
        return ()
    owner, = owners
    addresses = tuple(MemoryAddress(owner, field) for field in fields)
    return addresses if all(address in catalog for address in addresses) else ()


def resolve_current(catalog, addresses):
    current = VersionQuery('current')
    result = tuple(resolve_address(catalog, address, current, TODAY) for address in addresses)
    if any(reason != 'located' for _, reason in result):
        raise ValueError('Expected a current record for each selected field')
    return tuple(number for number, _ in result)


@torch.inference_mode()
def evaluate(model, tokenizer, codebook, head, reader, store, vectors):
    catalog = store.catalog()
    enrolled = {address.name for address in catalog}
    full = {(c.scenario.name, field): values for c in MEMORY_CASES
            for field, values in zip(c.scenario.fields, c.scenario.values, strict=True)}
    protected = tuple(value for pair in full.values() for value in pair)
    scores = head(vectors).sigmoid().tolist()
    rows, totals = [], Counter()
    for task, score in zip(HELD, scores, strict=True):
        if any(value in task.question for value in protected):
            raise ValueError('A stored value leaked into a user prompt')
        intended = {MemoryAddress(task.owner, field) for field in task.fields}
        old_targets = dialogue_targets(task.question, enrolled, FIELDS, ())
        old_route = {
            t.address for t in old_targets
            if t.address in catalog and t.query is not None and t.query.kind == 'current'
        }
        fields = tuple(field for field, confidence in zip(FIELDS, score, strict=True) if confidence >= .5)
        predicted = addresses_for(task.question, fields, catalog)
        route = resolve_current(catalog, predicted)
        if route:
            selected, output = reply_to_route(model, tokenizer, store, codebook, reader, task.question, list(route))
            if tuple(selected) != route:
                raise ValueError('Decoder did not use the selected record identities')
        else:
            output = {'answer': '', 'token_ids': [], 'copied': []}
        wanted = {full[item.name, item.field][1] for item in intended}
        observed = {value for value in protected if value in output['answer']}
        old_correct = old_route == intended
        new_correct = set(predicted) == intended
        generation_ok = bool(wanted) and observed == wanted and output['token_ids'][-1:] == [tokenizer.eos_token_id]
        totals.update(old_correct=old_correct, old_positive=bool(intended) and old_correct,
                      learned_correct=new_correct, learned_positive=bool(intended) and new_correct,
                      unnecessary_read=not intended and bool(route), all_requested_values=generation_ok)
        rows.append({'question': task.question, 'expected_owner': task.owner, 'expected_fields': task.fields,
                     'scores': dict(zip(FIELDS, score, strict=True)), 'route': route,
                     'old_fields': sorted(item.field for item in old_route), 'predicted': [item.field for item in predicted],
                     'old_correct': old_correct, 'correct': new_correct, 'wanted': sorted(wanted),
                     'values_appeared': sorted(observed), 'all_values': generation_ok, **output})
    return rows, totals


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    target = args.output / ('replay.json' if args.verify else 'result.json')
    if target.exists():
        raise FileExistsError(target)
    data = args.records / 'versions.sqlite3'
    reader = args.reader / 'head.pt'
    paths = (data, reader, args.records / 'result.json', args.reader / 'result.json',
             base.MODEL / 'model.safetensors', base.MODEL / 'tokenizer.json', Path(__file__),
             Path(__file__).with_name('spora_composition_address_cases.py'))
    sources = {str(path): digest(path) for path in paths}
    records, audit = [json.loads(path.read_text(encoding='utf-8')) for path in paths[2:4]]
    writer = next(value for file, value in records['sources'].items() if Path(file).name == 'adapters.pt')
    if (sources[str(data)] != records['database_sha256'] or sources[str(reader)] != audit['head_sha256']
            or sources[str(paths[4])] != base.WEIGHT_SHA):
        raise ValueError('Expected the fixed profile, reader and model')
    train = tuple(train_cases())
    if set(TRAIN_NAMES) & {row.owner for row in HELD} or {r.question for r in HELD} & {r.question for r in train}:
        raise ValueError('An evaluation person or request appears in training')
    model = load_full_streamed()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer)
    torch.manual_seed(SEED)
    head = make_head(model.config.hidden_size)
    artifact = args.output / 'addresses.pt'
    if args.verify:
        head.load_state_dict(torch.load(artifact, map_location='cpu', weights_only=True))
    else:
        fit(head, train, features_for(extract, train))
        atomic_save(head.state_dict(), artifact)
    state = torch.load(reader, map_location='cpu', weights_only=True)
    readout = VersionReplyActions(model.config.hidden_size, state['scaffold_tokens'], slots=2)
    readout.load_state_dict(state)
    embeddings = model.get_input_embeddings().weight.detach()
    columns = compress_dimension(embeddings.shape[1]).to(embeddings.device)
    codebook = F.normalize(embeddings.index_select(1, columns).float(), dim=-1)
    with VersionStore(data, BINDING_PREFIX + writer + ':' + base.WEIGHT_SHA) as store:
        rows, totals = evaluate(model, tokenizer, codebook, head.eval(), readout.eval().to(model.device), store,
                                features_for(extract, HELD))
    if any(digest(path) != sources[str(path)] for path in paths):
        raise ValueError('A sealed artifact changed')
    result = {'sources': sources, 'reader_sha256': sources[str(reader)], 'head_sha256': digest(artifact),
              'training_count': len(train), 'rows': rows, 'summary': totals, 'seconds': perf_counter() - started,
              'peak_bytes': torch.cuda.max_memory_allocated()}
    if args.verify:
        expected = json.loads((args.output / 'result.json').read_text(encoding='utf-8'))
        if any(json.loads(json.dumps(result[key])) != expected[key] for key in ('sources', 'head_sha256', 'rows', 'summary')):
            raise ValueError('Independent replay changed address decisions or replies')
    store_json(target, result)
    print(totals, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reader', 'records', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
