"""Compare learned memory-read authority with and without natural hard negatives."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
from time import perf_counter
from types import SimpleNamespace
from zipfile import ZipFile

import torch
from torch.nn import functional as F

from scripts.audit_spora_clause_read import HELD
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_field_evidence import (
    BINDING_PREFIX, DESCRIPTIONS, FIELDS, explicit_route, field_head,
    profile_values, read_reply, semantic_route,
)
from scripts.audit_spora_composition_addresses import resolve_current
from scripts.audit_spora_risawoz_updates import SOURCE_HASHES
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.audit_spora_fast_weight_lexical_write import compress_dimension
from scripts.spora_clause_read_cases import Task, training
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_version_store import MemoryAddress, VersionStore
from spora.research_versioned_memory_pointer import VersionReplyActions
from transformers import AutoTokenizer


SEED = 34017
MOVES = ('Request', 'Inform')
TOPICS = re.compile(r'喝|饮|餐厅|奶茶|咖啡|碰面|见面|位置|地方|地址|地点|哪儿|哪里|带|用品|书|读')


def corpus_cases(dialogues, *, count, excluded_texts=()):
    excluded_texts = set(excluded_texts)
    candidates = []
    for dialogue in dialogues:
        for turn in dialogue['dialogue']:
            raw = turn['user_utterance'].strip()
            acts = {act.strip() for act, domain, _, _ in turn['user_actions'] if domain != '通用'}
            if (8 <= len(raw) <= 90 and raw not in excluded_texts and TOPICS.search(raw)
                    and acts and acts <= set(MOVES)):
                candidates.append((dialogue['dialogue_id'], turn['turn_id'], raw))
    random.Random(SEED).shuffle(candidates)
    selected = {}
    for dialogue, turn, raw in candidates:
        selected.setdefault(dialogue, (dialogue, turn, raw))
        if len(selected) == count:
            return list(selected.values())
    raise ValueError('Insufficient independently sourced ordinary dialogue examples')


def features(extract, rows, owners):
    texts = [
        f'给{owner}处理请求。用户说：{row.question}\n'
        f'需要读取{owner}的{description}吗？'
        for row, owner in zip(rows, owners, strict=True) for description in DESCRIPTIONS.values()
    ]
    vectors = []
    for batch in range(0, len(texts), 24):
        local = texts[batch:batch + 24]
        hidden = extract(local)
        vectors.extend(hidden[i, len(text) - 1].clone() for i, text in enumerate(local))
    return torch.stack(vectors).float().reshape(len(rows), len(FIELDS), -1)


def fit(vectors, rows, *, extra_start, include_natural):
    torch.manual_seed(SEED)
    head = field_head(vectors.shape[-1])
    labels = torch.tensor([[field in row.fields for field in FIELDS] for row in rows], dtype=torch.float32)
    # Preserve every original positive and match the number of added all-negative rows.
    indices = torch.arange(extra_start)
    if include_natural:
        extra = torch.arange(extra_start, len(rows))
    else:
        eligible = torch.tensor([i for i, row in enumerate(rows[:extra_start]) if not row.fields])
        extra = eligible[torch.arange(len(rows) - extra_start) % len(eligible)]
    indices = torch.cat((indices, extra))
    rng = torch.Generator().manual_seed(SEED)
    opt = torch.optim.AdamW(head.parameters(), lr=.002)
    for _ in range(12):
        for subset in indices[torch.randperm(len(indices), generator=rng)].split(24):
            logits = head(vectors[subset]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels[subset], pos_weight=torch.tensor(2.))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return head.eval()


@torch.inference_mode()
def measure(heads, held, negative, held_vectors, negative_vectors, model, tokenizer, store, reader, codebook):
    catalog = store.catalog()
    values = profile_values(catalog)
    metrics = {name: Counter() for name in heads}
    outputs = []
    heads_probs = {name: (head(held_vectors).squeeze(-1).sigmoid().tolist(),
                          head(negative_vectors).squeeze(-1).sigmoid().tolist())
                   for name, head in heads.items()}
    for i, sample in enumerate(held):
        expected = resolve_current(catalog, [MemoryAddress(sample.owner, field) for field in sample.fields])
        explicit = explicit_route(sample.question, catalog)
        arms, cache = {}, {}
        for name, (predictions, _) in heads_probs.items():
            route = explicit if explicit is not None else semantic_route(sample, predictions[i], catalog)
            if route not in cache:
                cache[route] = read_reply(model, tokenizer, store, codebook, reader, sample.question,
                                         route, expected, values)
            metrics[name].update(exact_positive=bool(expected) and set(route) == set(expected),
                                 correct_values=bool(expected) and cache[route]['value_pass'],
                                 false_read=not expected and bool(route), missing=len(set(expected) - set(route)),
                                 extra=len(set(route) - set(expected)))
            arms[name] = {'route': route, **cache[route]}
        outputs.append({'question': sample.question, 'expected': expected, 'arms': arms})
    negatives = []
    for i, (dialogue, turn, _) in enumerate(negative):
        decisions = {name: [field for field, score in zip(FIELDS, probabilities[i], strict=True) if score >= .5]
                     for name, (_, probabilities) in heads_probs.items()}
        for name, fields in decisions.items():
            metrics[name]['natural_false_read'] += bool(fields)
        negatives.append({'dialogue': dialogue, 'turn': turn, 'predicted_fields': decisions})
    return metrics, outputs, negatives


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = perf_counter()
    torch.set_num_threads(2)
    db, head_path = args.records / 'versions.sqlite3', args.reader / 'head.pt'
    record_report_path, reader_report_path = args.records / 'result.json', args.reader / 'result.json'
    record_report, reader_report = [json.loads(path.read_text(encoding='utf-8'))
                                    for path in (record_report_path, reader_report_path)]
    sources = {path: sha(path) for path in (args.corpus, db, head_path, MODEL / 'model.safetensors',
                                           MODEL / 'tokenizer.json', MODEL / 'tokenizer_config.json',
                                           record_report_path, reader_report_path, Path(__file__))}
    if [sources[path] for path in (args.corpus, db, head_path, MODEL / 'model.safetensors')] != [
        SOURCE_HASHES[0], record_report['database_sha256'], reader_report['head_sha256'], WEIGHT_SHA
    ]:
        raise ValueError('Source corpus, model, archive or read head did not match its recorded digest')
    with ZipFile(args.corpus) as corpus:
        natural = corpus_cases(json.loads(corpus.read('source-data/all_train10000.json')), count=144)
        negative = corpus_cases(json.loads(corpus.read('source-data/all_dev600.json')), count=96,
                                excluded_texts=[row[2] for row in natural])
    if ({row[0] for row in natural} & {row[0] for row in negative}
            or {row[2] for row in natural} & {row[2] for row in negative}):
        raise ValueError('Training and evaluation corpus selections overlap')
    synthetic = tuple(dict.fromkeys(training()))
    # No current-record authority is granted by a self-contained corpus request.
    train = (*synthetic, *(Task(raw, None, ()) for _, _, raw in natural))
    model = load_full_streamed()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer)
    names = tuple(dict.fromkeys(row.owner for row in synthetic))
    train_owners = [row.owner or names[index % len(names)] for index, row in enumerate(train)]
    train_vectors = features(extract, train, train_owners)
    heads = {key: fit(train_vectors, train, extra_start=len(synthetic), include_natural=flag)
             for key, flag in (('matched_synthetic', False), ('natural_negatives', True))}
    held_vectors = features(extract, HELD, [row.owner or '林语' for row in HELD])
    negative_vectors = features(extract, [Task(raw, None, ()) for _, _, raw in negative], ['林语'] * len(negative))
    state = torch.load(head_path, map_location='cpu', weights_only=True)
    reader = VersionReplyActions(model.config.hidden_size, state['scaffold_tokens'], slots=2)
    reader.load_state_dict(state)
    embedding = model.get_input_embeddings().weight
    columns = compress_dimension(embedding.shape[1]).to(embedding.device)
    codebook = F.normalize(embedding.index_select(1, columns).float(), dim=-1)
    writer = next(digest for path, digest in record_report['sources'].items() if Path(path).name == 'adapters.pt')
    with VersionStore(db, BINDING_PREFIX + writer + ':' + WEIGHT_SHA) as store:
        summary, held, corpus = measure(heads, HELD, negative, held_vectors, negative_vectors,
                                        model, tokenizer, store, reader.eval().to(model.device), codebook)
    if any(sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('A sealed source was modified')
    atomic_save({name: head.state_dict() for name, head in heads.items()}, args.output / 'heads.pt')
    store_json(args.output / 'result.json', {
        'summary': summary, 'held': held, 'corpus': corpus, 'synthetic': len(synthetic),
        'natural_train': [(dialogue, turn) for dialogue, turn, _ in natural],
        'sources': {str(path): digest for path, digest in sources.items()}, 'seconds': perf_counter() - started,
    })
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('records', type=Path)
    parser.add_argument('reader', type=Path)
    parser.add_argument('output', type=Path)
    main(parser.parse_args())
