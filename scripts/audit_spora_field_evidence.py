"""Compare field-conditioned read decisions and preserve explicit version reads."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import re
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from scripts.audit_spora_composition_addresses import TODAY, addresses_for, features_for, make_head, resolve_current
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_dual_version_reply import FIELDS
from scripts.audit_spora_dual_version_routing import reply_to_route
from scripts.audit_spora_fast_weight_lexical_write import base, compress_dimension
from scripts.audit_spora_fresh_dialogue import digest
from scripts.audit_spora_typed_composition import BINDING_PREFIX
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_composition_address_cases import HELD, TRAIN_NAMES, train_cases
from scripts.spora_field_evidence_cases import EXPLICIT, FRESH
from scripts.spora_memory_composition_cases import HELD as MEMORY_CASES
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_multi_version_route import resolve_address
from spora.research_version_dialogue_scope import dialogue_targets
from spora.research_version_store import MemoryAddress, VersionStore
from spora.research_versioned_memory_pointer import VersionReplyActions


SEED = 84711
DESCRIPTIONS = {
    'drinks': '已登记的饮品或常喝的那一款',
    'places': '已登记的见面地点或原定碰头处',
    'items': '已登记的用品或说好要带的物件',
    'books': '已登记的书名或选中的读物',
}
VERSION = re.compile(r'当前版|现行版|首版|初版|上一版|前一版|第[一二三四五六七八九0-9]+版')


def field_features(extract, rows):
    marked = [row.question + '\n完成这个请求应当查阅' + DESCRIPTIONS[field] + '吗？'
              for row in rows for field in FIELDS]
    vectors = []
    for start in range(0, len(marked), 24):
        batch = marked[start:start + 24]
        hidden = extract(batch)
        vectors.extend(hidden[i, len(text) - 1].clone() for i, text in enumerate(batch))
    return torch.stack(vectors).float().reshape(len(rows), len(FIELDS), -1)


def field_head(width):
    return nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))


def fit(head, rows, vectors):
    targets = torch.tensor([[field in row.fields for field in FIELDS] for row in rows], dtype=torch.float32)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    rng = torch.Generator().manual_seed(SEED)
    for epoch in range(12):
        losses = []
        for subset in torch.randperm(len(rows), generator=rng).split(24):
            estimate = head(vectors[subset]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(estimate, targets[subset])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch in (0, 5, 11):
            print('field evidence', epoch + 1, sum(losses) / len(losses), flush=True)


def explicit_route(question, catalog):
    if not VERSION.search(question):
        return None
    enrolled = {address.name for address in catalog}
    targets = dialogue_targets(question, enrolled, FIELDS, ())
    if not 1 <= len(targets) <= 2:
        return ()
    decisions = tuple(resolve_address(catalog, row.address, row.query, TODAY, retain_same_day=True)
                      for row in targets)
    if any(reason != 'located' for _, reason in decisions):
        return ()
    numbers = tuple(number for number, _ in decisions)
    return numbers if len(set(numbers)) == len(numbers) else ()


def semantic_route(request, probabilities, catalog):
    chosen = tuple(field for field, prob in zip(FIELDS, probabilities, strict=True) if prob >= .5)
    return resolve_current(catalog, addresses_for(request.question, chosen, catalog))


def grade_route(mode, route, expected):
    return {'mode': mode, 'route': route, 'correct': len(route) == len(expected) and set(route) == set(expected)}


def profile_values(catalog):
    values = {}
    for spec in MEMORY_CASES:
        case = spec.scenario
        for field, versions in zip(case.fields, case.values, strict=True):
            entries = catalog[MemoryAddress(case.name, field)]
            for entry, value in zip(entries, versions, strict=True):
                values[entry.number] = value
    total = sum(map(len, catalog.values()))
    if total != len(values):
        raise ValueError('Unexpected version count in the sealed profile')
    return values


def read_reply(model, tokenizer, store, codebook, reader, question, route, expected, values):
    output = {'answer': '', 'token_ids': [], 'copied': []}
    if route:
        picked, output = reply_to_route(model, tokenizer, store, codebook, reader, question, list(route))
        if picked != list(route):
            raise ValueError('Readout used different records than the planned addresses')
    requested = {values[number] for number in expected}
    observed = {value for value in values.values() if value in output['answer']}
    passed = bool(expected) and requested == observed and output['token_ids'][-1:] == [tokenizer.eos_token_id]
    return {**output, 'value_pass': passed, 'observed': sorted(observed)}


@torch.inference_mode()
def evaluate_split(model, tokenizer, store, codebook, reader, rows, old_scores, new_scores):
    catalog = store.catalog()
    values = profile_values(catalog)
    logs, counts = [], {name: Counter() for name in ('whole_request', 'per_field', 'explicit_first')}
    for request, earlier, current in zip(rows, old_scores, new_scores, strict=True):
        if any(value in request.question for value in values.values()):
            raise ValueError('The question contains a registered value')
        wanted = resolve_current(catalog, [MemoryAddress(request.owner, field) for field in request.fields])
        original = semantic_route(request, earlier, catalog)
        learned = semantic_route(request, current, catalog)
        explicit = explicit_route(request.question, catalog)
        hybrid = learned if explicit is None else explicit
        decisions = []
        for label, route in zip(counts, (original, learned, hybrid), strict=True):
            grade = grade_route(label, route, wanted)
            counts[label].update(all=grade['correct'], correct_positive=bool(wanted) and grade['correct'],
                                 false_read=not wanted and bool(route), wrong_or_partial=bool(wanted) and bool(route)
                                 and not grade['correct'])
            decisions.append(grade)
        reply = read_reply(model, tokenizer, store, codebook, reader, request.question, hybrid, wanted, values)
        counts['explicit_first'].update(value_pass=reply['value_pass'])
        logs.append({'request': asdict(request), 'expected': wanted, 'whole_scores': earlier,
                     'field_scores': current, 'decisions': decisions, **reply})
    return logs, counts


def check_explicit(model, tokenizer, store, codebook, reader):
    catalog = store.catalog()
    values = profile_values(catalog)
    rows = []
    for question, name, field, version in EXPLICIT:
        wanted = () if name is None else (catalog[MemoryAddress(name, field)][version].number,)
        route = explicit_route(question, catalog)
        if route != wanted:
            print('explicit mismatch', question, route, wanted, flush=True)
        output = read_reply(model, tokenizer, store, codebook, reader, question, route or (), wanted, values)
        rows.append({'question': question, 'route': route, 'expected': wanted,
                     'passes': route == wanted and (not wanted or output['value_pass']), **output})
    return rows


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    target = args.output / ('replay.json' if args.verify else 'result.json')
    if target.exists():
        raise FileExistsError(target)
    database = args.records / 'versions.sqlite3'
    anchor = args.whole / 'addresses.pt'
    reader_path = args.reader / 'head.pt'
    files = (database, anchor, reader_path, args.records / 'result.json', args.whole / 'result.json',
             args.reader / 'result.json', base.MODEL / 'model.safetensors', base.MODEL / 'tokenizer.json',
             Path(__file__), Path(__file__).with_name('spora_field_evidence_cases.py'))
    sources = {str(path): digest(path) for path in files}
    records, baseline, audit = [json.loads(path.read_text(encoding='utf-8')) for path in files[3:6]]
    if (sources[str(database)] != records['database_sha256'] or sources[str(anchor)] != baseline['head_sha256']
            or sources[str(reader_path)] != audit['head_sha256'] or sources[str(files[6])] != base.WEIGHT_SHA):
        raise ValueError('A frozen model or archive does not match its audit')
    writer = next(value for file, value in records['sources'].items() if Path(file).name == 'adapters.pt')
    train = tuple(train_cases())
    if set(TRAIN_NAMES) & {row.owner for row in (*HELD, *FRESH)} or (
        {r.question for r in (*HELD, *FRESH)} & {r.question for r in train}
    ):
        raise ValueError('Training and evaluation requests overlap')
    model = load_full_streamed()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer)
    torch.manual_seed(SEED)
    whole = make_head(model.config.hidden_size).eval()
    whole.load_state_dict(torch.load(anchor, map_location='cpu', weights_only=True))
    head = field_head(model.config.hidden_size)
    artifact = args.output / 'field.pt'
    if args.verify:
        head.load_state_dict(torch.load(artifact, map_location='cpu', weights_only=True))
    else:
        fit(head, train, field_features(extract, train))
        atomic_save(head.state_dict(), artifact)
    head.eval()
    state = torch.load(reader_path, map_location='cpu', weights_only=True)
    reader = VersionReplyActions(model.config.hidden_size, state['scaffold_tokens'], slots=2)
    reader.load_state_dict(state)
    reader.eval().to(model.device)
    embeddings = model.get_input_embeddings().weight.detach()
    columns = compress_dimension(embeddings.shape[1]).to(embeddings.device)
    codebook = F.normalize(embeddings.index_select(1, columns).float(), dim=-1)
    splits, counts = {}, {}
    with VersionStore(database, BINDING_PREFIX + writer + ':' + base.WEIGHT_SHA) as store:
        for name, rows in (('prior', HELD), ('fresh', FRESH)):
            with torch.inference_mode():
                old_scores = whole(features_for(extract, rows)).sigmoid().tolist()
                scores = head(field_features(extract, rows)).squeeze(-1).sigmoid().tolist()
            splits[name], counts[name] = evaluate_split(model, tokenizer, store, codebook, reader,
                                                        rows, old_scores, scores)
            print(name, counts[name], flush=True)
        explicit = check_explicit(model, tokenizer, store, codebook, reader)
    if any(digest(path) != sources[str(path)] for path in files):
        raise ValueError('A sealed input changed')
    result = {'sources': sources, 'head_sha256': digest(artifact), 'rows': splits, 'summary': counts,
              'explicit': explicit, 'seconds': perf_counter() - started,
              'peak_bytes': torch.cuda.max_memory_allocated()}
    if args.verify:
        previous = json.loads((args.output / 'result.json').read_text(encoding='utf-8'))
        for key in ('sources', 'head_sha256', 'rows', 'summary', 'explicit'):
            if json.loads(json.dumps(result[key])) != previous[key]:
                raise ValueError(f'Independent replay changed {key}')
    store_json(target, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reader', 'records', 'whole', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
