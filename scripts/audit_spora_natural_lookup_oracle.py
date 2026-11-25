"""Check whether verified memory coordinates suffice for natural-language replies."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from scripts.audit_spora_natural_lookup_cli import PAIRS, RECORDS, digest
from scripts.spora_version_chat import (FIELD_IDS, VersionReplyActions, VersionStore, artifacts, base,
                                        compress_dimension, local_model, reply_to_route)
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.research_version_store import MemoryAddress
from torch.nn.functional import normalize


@torch.inference_mode()
def evaluate(model, tokenizer, store, reader):
    weight = model.get_input_embeddings().weight
    codebook = normalize(weight.index_select(1, compress_dimension(weight.shape[1]).to(weight.device)).float(), dim=-1)
    catalog = store.catalog()
    rows = []
    values = [record[-1] for record in RECORDS]
    for number, explicit, natural, _ in PAIRS:
        owner, label, value = RECORDS[number - 1]
        address = MemoryAddress(owner, FIELD_IDS[label])
        if number not in {row.number for row in catalog[address]} or any(term in natural for term in values):
            raise ValueError('Expected a bound address and a question free of record values')
        for mode, question, labels in (('explicit', explicit, [label]), ('natural', natural, [label]),
                                        ('unconstrained', natural, None)):
            try:
                route, output = reply_to_route(model, tokenizer, store, codebook, reader, question, [number], labels=labels)
                shown = [item for item in values if item in output['answer']]
                passed = route == [number] and shown == [value] and len(output['copied']) == 1
            except ValueError as error:
                output, passed = {'error': str(error)}, False
            rows.append({'mode': mode, 'question': question, 'record': number, 'value': value,
                         'output': output, 'passed': passed})
    print({mode: sum(row['mode'] == mode and row['passed'] for row in rows)
           for mode in ('explicit', 'natural', 'unconstrained')}, flush=True)
    return rows


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    started = perf_counter()
    torch.set_num_threads(2)
    prior = json.loads((args.prior / 'result.json').read_text(encoding='utf-8'))
    reader_path, _, _, binding = artifacts(args.reader, args.normalizer)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    state = torch.load(reader_path, map_location='cpu', weights_only=True)
    results = {}
    for label, recurrent in (('default', None), ('recurrent', args.recurrent)):
        source = args.prior / label / 'versions.sqlite3'
        before = digest(source)
        if before != prior[label]['archive_sha256']:
            raise ValueError('A prior replay archive changed')
        with local_model(recurrent) as model, VersionStore(source, binding) as store:
            reader = VersionReplyActions(model.config.hidden_size, state['scaffold_tokens'], slots=2)
            reader.load_state_dict(state)
            results[label] = evaluate(model, tokenizer, store, reader.eval().to(model.device))
        del model, reader
        torch.cuda.empty_cache()
        if digest(source) != before:
            raise ValueError('The oracle audit changed a source archive')
    store_json(args.output, {**results, 'reader_sha256': digest(reader_path), 'seconds': perf_counter() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reader', type=Path, default=Path('runs/spora-dual-version-reply-v1'))
    parser.add_argument('--normalizer', type=Path, default=Path('runs/spora-version-normalizer-v1'))
    parser.add_argument('--recurrent', required=True, nargs=3, type=Path)
    main(parser.parse_args())
