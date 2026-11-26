"""Isolate typed memory readout from route selection and chat history."""

import argparse
from datetime import date
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_dual_version_reply import reply
from scripts.audit_spora_paired_lookup_clarification import CASES, RESTART, VALUES
from scripts.spora_version_chat import (
    FIELDS, VersionChat, VersionReplyActions, VersionStore, artifacts, base, compress_dimension,
    dialogue_targets, generate, local_model, read_versions, resolve_address,
)
from scripts.train_spora_recurrent_counterfactual import store_json


def messages(question, result):
    return [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': result['answer']}]


def reported_case(rows, question, kind):
    return next(row['result'] for row in rows if row['question'] == question and row['result']['kind'] == kind)


def read_pairs(store, model, tokenizer, reader, history):
    catalog = store.catalog()
    enrolled = {address.name for address in catalog}
    embedding = model.get_input_embeddings().weight.detach()
    columns = compress_dimension(embedding.shape[1]).to(embedding.device)
    codebook = F.normalize(embedding.index_select(1, columns).float(), dim=-1)
    cases = [*CASES['complete'], (RESTART[1],)]
    rows = []
    for ((question, _, route),) in cases:
        targets = dialogue_targets(question, enrolled, FIELDS, ())
        actual = tuple(resolve_address(catalog, item.address, item.query, date.today(), retain_same_day=True)[0]
                       for item in targets)
        if actual != route or any(value in question for value in VALUES.values()):
            raise ValueError('The fixture no longer resolves to the expected internal records')
        stored = {number: store.read(number) for number in route}
        slots = read_versions({n: [(state.keys, state.values)] for n, state in stored.items()}, codebook)
        values = tuple(slots[number][0] for number in route)
        expected = [VALUES[number] for number in route]
        if [tokenizer.decode(value.tolist()) for value in values] != expected:
            raise ValueError('The archived tensors reconstructed different values')
        results = {}
        for context, text in (('prior', history), ('empty', ())):
            for complete in (False, True):
                started = perf_counter()
                output = reply(model, tokenizer, reader, question, values, history=text, require_coverage=True,
                               cover_present_slots=True, complete_on_eos=complete)
                verified = {item['slot'] for item in output['copied']
                            if output['token_ids'][item['at']:item['at'] + len(values[item['slot']])]
                            == values[item['slot']].tolist()}
                results[f'{context}_{"complete" if complete else "mask"}'] = dict(
                    **output, exact_slots=sorted(verified), all_slots=verified == {0, 1},
                    eos=output['token_ids'][-1:] == [tokenizer.eos_token_id], seconds=perf_counter() - started)
        print(question, {name: result['exact_slots'] for name, result in results.items()}, flush=True)
        rows.append(dict(question=question, route=route, expected=expected, results=results))
    return rows


@torch.inference_mode()
def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    report = json.loads((args.prior / 'result.json').read_text(encoding='utf-8'))
    archive = args.prior / 'profile' / 'versions.sqlite3'
    if base.sha(archive) != report['archive_sha256']:
        raise ValueError('The input archive differs from the completed replay')
    reader_path, _, writer_path, binding = artifacts(args.reader, args.normalizer)
    history = messages(RESTART[0][0], report['restart'][0]['result'])
    preceding = '请查唐砚平的见面地点当前版。'
    prior_read = messages(preceding, reported_case(report['rows'], preceding, 'read'))
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    with local_model(args.recurrent) as model, VersionStore(archive, binding) as store:
        weights = torch.load(reader_path, map_location='cpu', weights_only=True)
        reader = VersionReplyActions(model.config.hidden_size, weights['scaffold_tokens'], slots=2)
        reader.load_state_dict(weights)
        reader.eval().to(model.device)
        paired = read_pairs(store, model, tokenizer, reader, history)
        bare = {label: generate(model, tokenizer, RESTART[0][0], history=text, max_new_tokens=24)
                for label, text in (('prior_read', prior_read), ('empty', ()))}
        chat = VersionChat(store, model, tokenizer, reader, lambda _: [],
                           torch.load(writer_path, map_location='cpu', weights_only=True), ordinary_tokens=24,
                           typed_lookup=True)
        chat.history = list(history)
        try:
            actual = chat.turn(RESTART[1][0], date.today())
        except ValueError as error:
            actual = {'kind': 'error', 'answer': str(error)}
    if base.sha(archive) != report['archive_sha256']:
        raise RuntimeError('The diagnostic changed the input archive')
    print('entrypoint', actual, 'bare turns', {key: row['answer'] for key, row in bare.items()}, flush=True)
    store_json(args.output / 'result.json', dict(paired=paired, bare=bare, entrypoint=actual,
               archive_sha256=report['archive_sha256'], seconds=perf_counter() - started))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('prior', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--reader', type=Path, default=Path('runs/spora-dual-version-reply-v1'))
    cli.add_argument('--normalizer', type=Path, default=Path('runs/spora-version-normalizer-v1'))
    cli.add_argument('--recurrent', type=Path, nargs=3, required=True)
    main(cli.parse_args())
