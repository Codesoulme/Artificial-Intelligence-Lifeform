"""Compare the same address selector with learned, oracle, or absent read intent."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from zipfile import ZipFile

import torch
from torch.nn import functional as F

from scripts import audit_spora_natural_read_evidence as natural
from scripts import audit_spora_paired_read_authority as paired
from scripts.audit_spora_composition_addresses import resolve_current
from scripts.audit_spora_field_evidence import check_explicit


SEED = 38612
THRESHOLD = 0.


def intent_features(extract, questions):
    texts = [f'用户说：{question}\n现在必须查以前登记的个人事实才能完成要求吗？' for question in questions]
    rows = []
    for start in range(0, len(texts), 24):
        batch = texts[start:start + 24]
        hidden = extract(batch)
        rows.extend(hidden[i, len(text) - 1].clone() for i, text in enumerate(batch))
    return torch.stack(rows).float()


def train_gate(vectors, labels):
    torch.manual_seed(SEED)
    head = natural.field_head(vectors.shape[-1])
    opt = torch.optim.AdamW(head.parameters(), lr=.002)
    rng = torch.Generator().manual_seed(SEED)
    for _ in range(12):
        for batch in torch.randperm(len(labels), generator=rng).split(24):
            scores = head(vectors[batch]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(scores, labels[batch])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return head.eval()


def corpus_split(path, previous):
    with ZipFile(path) as archive:
        train = natural.corpus_cases(json.loads(archive.read('source-data/all_train10000.json')), count=144)
        dev = natural.corpus_cases(json.loads(archive.read('source-data/all_dev600.json')), count=96,
                                   excluded_texts=[row[2] for row in train])
    expected_train = [tuple(row) for row in previous['natural_train']]
    expected_dev = [(row['dialogue'], row['turn']) for row in previous['corpus']]
    if [row[:2] for row in train] != expected_train or [row[:2] for row in dev] != expected_dev:
        raise ValueError('Natural requests must reproduce the prior corpus split')
    return train, dev


@torch.inference_mode()
def measure(model, tokenizer, store, reader, codebook, extract, select, gate, dev):
    catalog = store.catalog()
    values = natural.profile_values(catalog)
    requests = [sample.question for sample in natural.HELD]
    intent = gate(intent_features(extract, requests)).squeeze(-1).tolist()
    dev_scores = gate(intent_features(extract, [row[2] for row in dev])).squeeze(-1).tolist()
    cases = []
    for sample in natural.HELD:
        addresses = [a for a in sorted(catalog) if a.name in sample.question]
        expected = resolve_current(catalog, [natural.MemoryAddress(sample.owner, field) for field in sample.fields])
        cases.append((sample, addresses, expected))
    candidate_vectors = iter(paired.features(extract, [(s.question, a) for s, addresses, _ in cases for a in addresses]))
    metrics = {label: Counter() for label in ('selector', 'learned_gate', 'oracle_gate')}
    rows = []
    gate_stats = Counter()
    for (sample, addresses, expected), score in zip(cases, intent, strict=True):
        vectors = [next(candidate_vectors) for _ in addresses]
        scores = select(torch.stack(vectors)).squeeze(-1).tolist() if vectors else []
        selected = [a for a, logit in zip(addresses, scores, strict=True) if logit >= THRESHOLD]
        allowed = score >= THRESHOLD
        gate_stats.update(positive=bool(expected), accepted_positive=bool(expected) and allowed,
                          negative=not expected, accepted_negative=not expected and allowed)
        if any(value in sample.question for value in values.values()):
            raise ValueError('A stored value is in the question')
        selected_route = resolve_current(catalog, selected) if 1 <= len(selected) <= 2 else ()
        arms, cached = {}, {}
        for label, opened in (('selector', True), ('learned_gate', allowed), ('oracle_gate', bool(expected))):
            route = selected_route if opened else ()
            if route not in cached:
                cached[route] = natural.read_reply(model, tokenizer, store, codebook, reader, sample.question,
                                                   route, expected, values)
            actual = cached[route]
            exact = set(route) == set(expected)
            metrics[label].update(exact_positive=bool(expected) and exact,
                                  correct_values=bool(expected) and exact and actual['value_pass'],
                                  false_read=not expected and bool(route), missing=len(set(expected) - set(route)),
                                  extra=len(set(route) - set(expected)))
            arms[label] = {'route': route, **actual}
        rows.append({'question': sample.question, 'expected': expected, 'gate_score': score, 'arms': arms,
                     'candidates': [(a.name, a.field, s) for a, s in zip(addresses, scores, strict=True)]})
    gate_stats.update(corpus_negative=len(dev), corpus_accepted=sum(s >= THRESHOLD for s in dev_scores))
    dev_rows = [{'dialogue': d, 'turn': t, 'gate_score': s} for (d, t, _), s in zip(dev, dev_scores, strict=True)]
    return metrics, gate_stats, rows, dev_rows, check_explicit(model, tokenizer, store, codebook, reader)


def main(args):
    start = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    db, reader_file = args.records / 'versions.sqlite3', args.reader / 'head.pt'
    record_report, reader_report = args.records / 'result.json', args.reader / 'result.json'
    selector_file, selector_report = args.selector / 'heads.pt', args.selector / 'result.json'
    records, reader_meta, selector_meta, prior = [
        json.loads(path.read_text(encoding='utf-8'))
        for path in (record_report, reader_report, selector_report, args.prior)
    ]
    model_path = natural.MODEL / 'model.safetensors'
    paths = (db, reader_file, record_report, reader_report, selector_file, selector_report, args.prior, args.corpus,
             model_path, natural.MODEL / 'tokenizer.json', natural.MODEL / 'tokenizer_config.json', Path(__file__),
             Path(natural.__file__), Path(paired.__file__))
    sources = {path: natural.sha(path) for path in paths}
    if (sources[db], sources[reader_file], sources[selector_file], sources[args.corpus], sources[model_path]) != (
        records['database_sha256'], reader_meta['head_sha256'], selector_meta['head_sha256'],
        natural.SOURCE_HASHES[0], natural.WEIGHT_SHA,
    ):
        raise ValueError('A bound research artifact does not match')
    training, dev = corpus_split(args.corpus, prior)
    synthetic = tuple(dict.fromkeys(natural.training()))
    questions = [row.question for row in synthetic] + [row[2] for row in training]
    labels = torch.tensor([bool(row.fields) for row in synthetic] + [False] * len(training), dtype=torch.float32)
    if set(questions) & ({row.question for row in natural.HELD} | {row[2] for row in dev}):
        raise ValueError('Gate training overlaps held evaluation')
    model = natural.load_full_streamed()
    tokenizer = natural.AutoTokenizer.from_pretrained(natural.MODEL, local_files_only=True, trust_remote_code=False)
    extract = natural.FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer)
    gate = train_gate(intent_features(extract, questions), labels)
    select = natural.field_head(model.config.hidden_size)
    select.load_state_dict(torch.load(selector_file, map_location='cpu', weights_only=True)['binary'])
    state = torch.load(reader_file, map_location='cpu', weights_only=True)
    reader = natural.VersionReplyActions(model.config.hidden_size, state['scaffold_tokens'], slots=2)
    reader.load_state_dict(state)
    embedding = model.get_input_embeddings().weight
    columns = natural.compress_dimension(embedding.shape[1]).to(embedding.device)
    codebook = F.normalize(embedding.index_select(1, columns).float(), dim=-1)
    writer = next(sha for path, sha in records['sources'].items() if Path(path).name == 'adapters.pt')
    with natural.VersionStore(db, natural.BINDING_PREFIX + writer + ':' + natural.WEIGHT_SHA) as store:
        metrics, gate_stats, rows, corpus, explicit = measure(
            model, tokenizer, store, reader.eval().to(model.device), codebook, extract, select.eval(), gate, dev,
        )
    baseline = selector_meta['held']
    for row, previous in zip(rows, baseline, strict=True):
        if (row['question'], list(row['expected']), list(row['arms']['selector']['route'])) != (
            previous['question'], previous['expected'], previous['arms']['binary']['route'],
        ):
            raise RuntimeError('Ungated address selection differs from the frozen baseline')
    if not all(row['passes'] for row in explicit):
        raise RuntimeError('An explicit version read failed its control')
    if any(natural.sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('A sealed source changed')
    checkpoint = args.output / 'gate.pt'
    natural.atomic_save(gate.state_dict(), checkpoint)
    natural.store_json(args.output / 'result.json', {
        'metrics': metrics, 'gate': gate_stats, 'rows': rows, 'corpus': corpus, 'explicit': explicit,
        'training': {'synthetic': len(synthetic), 'natural': len(training), 'positives': int(labels.sum())},
        'gate_sha256': natural.sha(checkpoint), 'sources': {str(path): digest for path, digest in sources.items()},
        'seconds': perf_counter() - start,
    })
    print(metrics, gate_stats, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('records', type=Path)
    parser.add_argument('reader', type=Path)
    parser.add_argument('selector', type=Path)
    parser.add_argument('prior', type=Path)
    parser.add_argument('output', type=Path)
    main(parser.parse_args())
