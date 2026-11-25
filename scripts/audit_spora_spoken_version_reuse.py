"""Test the installed temporal normalizer with read permission and address supplied."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import date
import json
from pathlib import Path
import re
from time import perf_counter
from types import SimpleNamespace

import torch

from scripts.audit_spora_natural_lookup_cli import digest
from scripts.audit_spora_spoken_query_data import FRESH_READS, RECORDS
from scripts.audit_spora_spoken_query_plan import VERSIONS
from scripts.spora_version_chat import (FIELD_IDS, FrozenTurnFeatures, TemporalProposals, VersionStore,
                                        artifacts, base, local_model)
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.research_multi_version_route import parse_lookup_scope, resolve_address, spoken_scope_evidence
from spora.research_version_store import MemoryAddress


@torch.inference_mode()
def evaluate(catalog, proposer, baseline):
    totals = {mode: Counter() for mode in ('learned_coordinates', 'explicit_time_rules', 'rules_then_normalizer')}
    aliases = re.compile('|'.join({owner for owner, _, _ in RECORDS}))
    rows = []
    for (number, version, question), prior in zip(FRESH_READS, baseline[:len(FRESH_READS)], strict=True):
        if (question, number) != (prior['question'], prior['expected_record']):
            raise ValueError('Expected the independent held set from the data audit')
        owner, field, _ = RECORDS[number - 1]
        address = MemoryAddress(owner, FIELD_IDS[field])
        rule = parse_lookup_scope(question)
        evidence = spoken_scope_evidence(aliases.sub('那个人', question))
        learned = proposer([evidence])[0] if rule is None and evidence else None
        queries = {'learned_coordinates': VERSIONS[prior['scored'][3][0]], 'explicit_time_rules': rule,
                   'rules_then_normalizer': rule or learned}
        arms = {}
        for mode, query in queries.items():
            recorded, reason = resolve_address(catalog, address, query, date.today(), retain_same_day=True)
            totals[mode].update(exact_version=query == VERSIONS[version], exact_record=recorded == number,
                                abstained=query is None, wrong_record=recorded is not None and recorded != number)
            arms[mode] = dict(query=asdict(query) if query else None, record=recorded, reason=reason)
        rows.append(dict(question=question, number=number, owner=owner, field=field,
                         expected=asdict(VERSIONS[version]), evidence=evidence, arms=arms))
    return rows, totals


def main(args):
    started = perf_counter()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    report = args.prior / 'result.json'
    prior = json.loads(report.read_text(encoding='utf-8'))
    database = args.prior / 'profile' / 'versions.sqlite3'
    _, normalizer, _, binding = artifacts(args.reader, args.normalizer)
    paths = (report, database, normalizer, Path(__file__), args.recurrent[0],
             *(directory / 'attention.pt' for directory in args.recurrent[1:]))
    sources = {path: digest(path) for path in paths}
    if sources[database] != prior['sources'][str(database.resolve())]:
        raise ValueError('Expected the unchanged independent profile')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    with local_model(args.recurrent) as model, VersionStore(database, binding) as store:
        extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=len(model.model.layers))
        state = torch.load(normalizer, map_location='cpu', weights_only=True)['final_layer']
        proposer = TemporalProposals(extract, model.config.hidden_size, state)
        rows, metrics = evaluate(store.catalog(), proposer, prior['results']['public']['rows'])
    if any(digest(path) != expected for path, expected in sources.items()):
        raise ValueError('A source changed during the temporal diagnostic')
    store_json(args.output, {'rows': rows, 'metrics': metrics, 'seconds': perf_counter() - started,
                             'sources': {str(p): h for p, h in sources.items()}})
    print(metrics, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reader', type=Path, default=Path('runs/spora-dual-version-reply-v1'))
    parser.add_argument('--normalizer', type=Path, default=Path('runs/spora-version-normalizer-v1'))
    parser.add_argument('--recurrent', required=True, nargs=3, type=Path)
    main(parser.parse_args())
