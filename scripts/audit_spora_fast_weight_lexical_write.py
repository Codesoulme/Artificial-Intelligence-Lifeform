"""Compare a compact, lexical write target against contextual fast-weight targets."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_fast_weight_value_decoder import (
    ALTERNATE, FACT, FRESH_NAMES, FRESH_VALUES, load_capture_model, target_vectors, tokens,
)
from scripts.audit_spora_isolated_fast_weights import SOURCE, base, event_batches
from scripts.deepseek_spora_ttt_joint_writer_reader_pilot import fact_cache
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_ttt_ntp import LowRankFastWeight, install


WIDTH = 128
SEED = 26312


def compress_dimension(width):
    generator = torch.Generator().manual_seed(SEED)
    return torch.randperm(width, generator=generator)[:WIDTH]


def lexical_write(state, keys, codes, *, eta, ridge):
    # Keep the existing delta rule while using a fixed-width code from the frozen embedding.
    return state.write(F.normalize(keys.float(), dim=-1), codes, eta=eta, ridge=ridge)


@torch.no_grad()
def events(model, tokenizer, site, columns_for_keys, columns, eta, ridge):
    name_specs = [event for _, part in event_batches() for event in part]
    first = list(zip(FRESH_NAMES, FRESH_VALUES[:4], strict=True))
    second = list(zip(FRESH_NAMES, FRESH_VALUES[4:], strict=True))
    banks, versions = {}, {}
    layer = next(iter(site))
    for name, value in (*name_specs, *first, *second):
        cache = fact_cache(model, tokenizer, site, (FACT if name not in FRESH_NAMES else ALTERNATE).format(
            name=name, value=value), value, decoder_only=True)
        # Use the source-token embedding targets captured by fact_cache, never an answer lookup table.
        keys = cache['keys'][layer].index_select(-1, columns_for_keys)
        codes = F.normalize(cache['value_embeddings'].index_select(-1, columns), dim=-1)
        if len(codes) != len(tokens(tokenizer, value)):
            raise ValueError('Contextual span uses a different number of tokens')
        state = banks.get(name)
        if state is None:
            state = LowRankFastWeight(keys.new_empty(0, WIDTH), codes.new_empty(0, WIDTH))
        before = versions.get(name, [])
        updated = lexical_write(state, keys, codes, eta=eta, ridge=ridge)
        banks[name] = updated
        versions[name] = [*before, len(codes)]
    return banks, versions


def recover(state, lengths, eta, ridge):
    return target_vectors({0: state}, sum(lengths), eta, ridge)


def grade(banks, versions, codebook, tokenizer, eta, ridge):
    expected = {}
    for _, batch in event_batches():
        for name, value in batch:
            expected.setdefault(name, []).append(value)
    for i, name in enumerate(FRESH_NAMES):
        expected[name] = [FRESH_VALUES[i], FRESH_VALUES[i + 4]]
    reads = {}
    for name, state in banks.items():
        vectors = recover(state, versions[name], eta, ridge).to(codebook.device)
        scores, choices = (F.normalize(vectors, dim=-1) @ codebook.T).topk(2)
        rows = []
        offset = 0
        for n, value in zip(versions[name], expected[name], strict=True):
            ids = choices[offset:offset + n, 0].tolist()
            wanted = tokens(tokenizer, value)
            rows.append({'expected_ids': wanted, 'decoded_ids': ids, 'value': tokenizer.decode(ids),
                         'exact': ids == wanted, 'minimum_margin': float((scores[offset:offset+n, 0]
                                                                          - scores[offset:offset+n, 1]).min())})
            offset += n
        reads[name] = rows
    return reads


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    state_file, report_file = args.output / 'states.pt', args.output / 'result.json'
    base_path = base.MODEL / 'model.safetensors'
    source = SOURCE / 'adapters.pt'
    inputs = (base_path, source, SOURCE / 'result.json')
    hashes = {str(path): base.sha(path) for path in inputs}
    meta = json.loads(inputs[-1].read_text(encoding='utf-8'))
    if hashes[str(base_path)] != base.WEIGHT_SHA or hashes[str(source)] != meta['adapter_sha256']:
        raise ValueError('Frozen write hyperparameters or base changed')
    settings = torch.load(source, map_location='cpu', weights_only=True)
    eta, ridge = settings['eta'], settings['ridge']
    expected = json.loads(report_file.read_text(encoding='utf-8')) if args.verify else None
    if expected is not None and (expected['inputs'] != hashes or expected['state_sha256'] != base.sha(state_file)):
        raise ValueError('Saved lexical state does not match the experiment')
    model = load_capture_model(settings['layers'][:1])
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    weights = model.get_input_embeddings().weight.detach()
    columns = compress_dimension(weights.shape[1]).to(weights.device)
    codebook = F.normalize(weights.index_select(-1, columns).float(), dim=-1)
    site = install(model, tuple(settings['layers'][:1]))
    columns_for_keys = compress_dimension(next(iter(site.values())).base.in_features).to(weights.device)
    if args.verify:
        payload = torch.load(state_file, weights_only=True, map_location='cpu')
        versions = payload['event_lengths']
        banks = {name: LowRankFastWeight(keys.to(weights.device), values.to(weights.device))
                 for name, (keys, values) in payload['factors'].items()}
    else:
        banks, versions = events(model, tokenizer, site, columns_for_keys, columns, eta, ridge)
        atomic_save({'factors': {name: (state.keys.cpu(), state.values.cpu()) for name, state in banks.items()},
                     'event_lengths': versions}, state_file)
    rows = grade(banks, versions, codebook, tokenizer, eta, ridge)
    latest = [values[-1] for values in rows.values()]
    historical = [entry for values in rows.values() for entry in values[:-1]]
    corrections = [values[-1] for values in rows.values() if len(values) > 1]
    report = {'inputs': hashes, 'state_sha256': base.sha(state_file), 'state_bytes': state_file.stat().st_size,
              'factor_bytes': sum((state.keys.numel() + state.values.numel()) * state.keys.element_size()
                                  for state in banks.values()),
              'latest': [sum(row['exact'] for row in latest), len(latest)],
              'historical': [sum(row['exact'] for row in historical), len(historical)],
              'corrected': [sum(row['exact'] for row in corrections), len(corrections)],
              'minimum_margin': min(row['minimum_margin'] for values in rows.values() for row in values),
              'records': rows}
    if any(base.sha(path) != hashes[str(path)] for path in inputs):
        raise ValueError('A protected input changed')
    if expected is not None and any(expected[k] != v for k, v in report.items()):
        raise ValueError('Independent reload changed a lexical read')
    print({key: report[key] for key in ('latest', 'historical', 'corrected', 'state_bytes', 'factor_bytes',
                                       'minimum_margin')}, flush=True)
    output = args.output / ('reload.json' if args.verify else 'result.json')
    if args.verify and output.exists():
        previous = json.loads(output.read_text(encoding='utf-8'))
        if any(previous[k] != v for k, v in report.items()):
            raise ValueError('The earlier reload artifact disagrees with this run')
    else:
        store_json(output, {**report, 'seconds': perf_counter() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
