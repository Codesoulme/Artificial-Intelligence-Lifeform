"""Compare frozen-model address selection with known addresses on versioned recall."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_dual_version_reply import reply as span_reply
from scripts.audit_spora_hybrid_language_recovery import FIELD_NAMES
from scripts.audit_spora_revision_version_state import (
    FIELDS, MODEL, NEW_DAY, WRITER_SOURCE, AutoTokenizer, VersionReplyActions, VersionStore,
    compress_dimension, extend_positions, frozen_digest, load_full_streamed, load_integrated_runtime,
    load_recurrent_replacements, read_versions, retained_cases, using,
)
from scripts.benchmark_spora_recurrent_dialogue import encode
from scripts.train_spora_recurrent_counterfactual import sha, store_json
from scripts.train_spora_revision_composition import factual_roles_correct
from spora.research_multi_version_route import resolve_address
from spora.research_version_store import MemoryAddress
from spora.research_version_timeline import VersionQuery


def rank_options(decoder, tokenizer, labels, question, options, history):
    size = len(options)
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:size]
    scores, votes = [0.] * size, []
    for rotation in range(size):
        order = [(rotation + offset) % size for offset in range(size)]
        choices = '\n'.join(f'{label}. {options[index]}' for label, index in zip(letters, order, strict=True))
        prefix = encode(tokenizer, [*history, {'role': 'user', 'content': f'{question}\n{choices}'}])
        if len(prefix) > decoder.position_table_max_position:
            raise ValueError('Routing prompt exceeds the model position table')
        inputs = torch.tensor([prefix], device=decoder.base.device)
        logits = decoder(input_ids=inputs, use_cache=True, logits_to_keep=1).logits[0, -1].float()
        option_scores = F.log_softmax(logits[labels], dim=0).tolist()
        votes.append(order[max(range(size), key=option_scores.__getitem__)])
        for index, score in zip(order, option_scores, strict=True):
            scores[index] += score / size
    return scores, votes


def locate(decoder, tokenizer, labels, case, candidates, *, with_history):
    history = case['history'] if with_history else []
    options = [candidate[0] for candidate in candidates]
    scores, votes = zip(*(rank_options(decoder, tokenizer, labels,
        f'用户请求：{case["question"]}\n下面哪条记录对应请求中的{ordinal}信息？'
        '按请求中提到的先后顺序判断，只回答选项字母。', options, history)
        for ordinal in ('第一项', '第二项')), strict=True)
    size = len(candidates)
    selected = max(permutations(range(size), 2), key=lambda pair: scores[0][pair[0]] + scores[1][pair[1]])
    return selected, dict(scores=scores, votes=votes)


def recall(store, catalog, codebook, tokenizer, candidates, selected):
    numbers, tokens = [], []
    for index in selected:
        _, address = candidates[index]
        number, reason = resolve_address(catalog, address, VersionQuery('current'), NEW_DAY)
        if reason != 'located':
            raise ValueError(f'No current record at {address}: {reason}')
        saved = store.read(number)
        decoded = read_versions({'value': [(saved.keys, saved.values)]}, codebook)['value'][0]
        numbers.append(number)
        tokens.append(decoded)
    return numbers, tokens, [tokenizer.decode(value) for value in tokens]


@torch.inference_mode()
def main(args):
    started = perf_counter()
    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    database = args.reference.parent / 'versions.sqlite'
    writer = WRITER_SOURCE / 'adapters.pt'
    head_file, head_report = args.composer / 'short.pt', args.composer / 'result.json'
    layer_files = [directory / name for directory in args.layers for name in ('attention.pt', 'comparison.json')]
    sources = {path: sha(path) for path in (args.checkpoint, args.reference, database, writer, head_file,
                                           head_report, *layer_files)}
    composer = json.loads(head_report.read_text(encoding='utf-8'))
    identity = reference['individual_sha256']
    if (any(reference['sources'].get(str(path)) != sources[path] for path in (args.checkpoint, writer, *layer_files))
            or any(composer['sources'].get(str(path)) != sources[path] for path in (args.checkpoint, *layer_files))
            or composer['results']['short']['head_sha256'] != sources[head_file]
            or composer['individual_sha256'] != identity):
        raise ValueError('Model, memory, and composition artifacts must match')
    binding = sha256(json.dumps((identity, sources[writer], sources[args.checkpoint])).encode()).hexdigest()
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    labels = [tokenizer.encode(label, add_special_tokens=False) for label in 'ABCD']
    if any(len(label) != 1 for label in labels) or len({label[0] for label in labels}) != 4:
        raise ValueError('Expected four distinct single-token options')
    labels = [label[0] for label in labels]
    cases, skipped = retained_cases(tokenizer)
    case_names = list(dict.fromkeys(case['case'] for case in cases))
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    base.lm_head.to(base.device)
    if frozen_digest(runtime.organism, runtime.state) != identity:
        raise ValueError('Loaded a different individual')
    embedding = base.get_input_embeddings().weight.detach()
    columns = compress_dimension(embedding.shape[1]).to(embedding.device)
    codebook = F.normalize(embedding.index_select(1, columns).float(), dim=-1)
    weights = torch.load(head_file, weights_only=True, map_location='cpu')
    head = VersionReplyActions(base.config.hidden_size, weights['scaffold_tokens'], slots=2)
    head.load_state_dict(weights, strict=True)
    head.to(base.device).eval().requires_grad_(False)
    replacements = load_recurrent_replacements(decoder, args.layers, sources[args.checkpoint])
    if set(replacements) != {19, 22}:
        raise ValueError('Expected the two-layer recurrent baseline')
    rows, counts = [], {mode: Counter() for mode in ('given', 'question', 'history')}
    with VersionStore(database, binding) as store, using(replacements, 'recurrent'):
        catalog = store.catalog()
        for case in cases:
            name, variant = case['case'], case['variant']
            foil = case_names[(case_names.index(name) + 1 + variant) % len(case_names)]
            candidates = [(label, MemoryAddress(f'{scenario}/{variant}', field))
                          for scenario in (name, foil)
                          for label, field in zip(FIELD_NAMES[scenario], FIELDS[scenario], strict=True)]
            candidates.sort(key=lambda item: item[0])
            expected = tuple(next(index for index, (_, address) in enumerate(candidates)
                                  if address == MemoryAddress(f'{name}/{variant}', field)) for field in FIELDS[name])
            if any(value in case['question'] or any(value in label for label, _ in candidates)
                   for value in case['values'].values()):
                raise ValueError('A question or address label leaks a stored value')
            variants, generated = {}, {}
            for mode, count in counts.items():
                selected, routing = (expected, {}) if mode == 'given' else locate(
                    decoder, tokenizer, labels, case, candidates, with_history=mode == 'history')
                if selected not in generated:
                    numbers, tokens, decoded = recall(store, catalog, codebook, tokenizer, candidates, selected)
                    output = span_reply(base, tokenizer, head, case['question'], tokens, history=case['history'],
                                        require_coverage=True, complete_on_eos=True, max_new_tokens=48)
                    text = output['answer']
                    values = case['values']
                    eos = output['token_ids'][-1:] == [tokenizer.eos_token_id]
                    passed = eos and values['new'] in text and values['keep'] in text and values['old'] not in text
                    output.update(eos=eos, passed=passed, use=factual_roles_correct(case, values['new'],
                                                                                  dict(answer=text, passed=passed)))
                    generated[selected] = dict(numbers=numbers, decoded=decoded, output=output)
                result = generated[selected]
                read_correct = result['decoded'] == [case['values']['new'], case['values']['keep']]
                count.update(total=1, exact_pair=selected == expected, pair_unordered=set(selected) == set(expected),
                             address_hits=len(set(selected) & set(expected)), read_pair=read_correct,
                             eos=result['output']['eos'], answer=result['output']['passed'],
                             role_correct=result['output']['use'], stale=case['values']['old'] in result['output']['answer'])
                variants[mode] = dict(selected=selected, routing=routing, **result)
            if variants['given']['decoded'] != [case['values']['new'], case['values']['keep']]:
                raise ValueError('The given address did not reconstruct the expected tensors')
            rows.append(dict(case=name, variant=variant, question=case['question'], values=case['values'],
                             candidates=[dict(label=label, name=address.name, field=address.field)
                                         for label, address in candidates], expected=expected, variants=variants))
            print(name, variant, {mode: row['selected'] for mode, row in variants.items()}, flush=True)
    if frozen_digest(runtime.organism, runtime.state) != identity or any(
            sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('A source or individual changed in the address comparison')
    store_json(args.output, dict(rows=rows, summary=counts, skipped=skipped, individual_sha256=identity,
                                sources={str(path): digest for path, digest in sources.items()},
                                seconds=perf_counter() - started))
    print(counts, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('reference', type=Path)
    cli.add_argument('composer', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('layers', type=Path, nargs=2)
    args = cli.parse_args()
    if args.output.exists():
        cli.error('Choose a new output path')
    main(args)
