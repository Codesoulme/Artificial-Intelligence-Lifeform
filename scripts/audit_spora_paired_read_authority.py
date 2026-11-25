"""Measure paired read authority changes while freezing the language backbone."""

import argparse
from collections import Counter
from itertools import permutations
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from scripts.audit_spora_composition_addresses import resolve_current
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_field_evidence import BINDING_PREFIX, DESCRIPTIONS, field_head, profile_values, read_reply
from scripts.audit_spora_fast_weight_lexical_write import compress_dimension
from scripts.audit_spora_natural_read_evidence import HELD, MODEL, WEIGHT_SHA, AutoTokenizer, load_full_streamed, sha
from scripts.spora_composition_address_cases import TRAIN_NAMES
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_version_store import MemoryAddress, VersionStore
from spora.research_versioned_memory_pointer import VersionReplyActions


SEED = 71422
ALIASES = {
    'drinks': ('常喝的饮料', '选的那杯'),
    'places': ('约好的会面地点', '原定的集合点'),
    'items': ('说好要带的东西', '备好的用品'),
    'books': ('借的那本书', '挑中的读物'),
}


def training_pairs():
    for index, (first, second) in enumerate(permutations(TRAIN_NAMES, 2)):
        for field, terms in ALIASES.items():
            term = terms[index % len(terms)]
            reversals = (
                (f'请查{first}记下的{term}，{second}那项就别提。',
                 f'请查{second}记下的{term}，{first}那项就别提。'),
                (f'{first}的{term}先放着，{second}也先别查。想好了，报{first}原来那项。',
                 f'{first}的{term}先放着，{second}也先别查。想好了，报{second}原来那项。'),
                (f'把{first}以前登记的{term}说出来，{second}这次不用查。',
                 f'把{first}以前登记的{term}留空，{second}这次也不用查。'),
            )
            yes, no = reversals[index % len(reversals)]
            yield yes, no, MemoryAddress(first, field)
    for index, person in enumerate(TRAIN_NAMES):
        for first, second in permutations(ALIASES, 2):
            a, b = ALIASES[first][index % 2], ALIASES[second][1 - index % 2]
            address = MemoryAddress(person, first)
            yield (f'{person}的{a}按已经登记的写，{b}先别填。',
                   f'{person}的{a}先别填，{b}按已经登记的写。', address)
            yield (f'{person}原先说的{a}和{b}都给我核对一下。',
                   f'{person}原先说的{a}先不要读，先只核对{b}。', address)


FRESH = (
    ('地点我留给赵茗初那边再选。何青玟倒还照上次的碰头处，麻烦报一下。',
     '地点我留给何青玟那边再选。赵茗初倒还照上次的碰头处，麻烦报一下。', MemoryAddress('何青玟', 'places')),
    ('有变动，饮料给许昀芝去翻旧记录，黎书然这一杯重选吧。',
     '有变动，饮料给黎书然去翻旧记录，许昀芝这一杯重选吧。', MemoryAddress('许昀芝', 'drinks')),
    ('岑婉听过去说要带的那样先不用给我，她以前约的地方得填上。',
     '岑婉听过去说要带的那样得填上，她以前约的地方先不用给我。', MemoryAddress('岑婉听', 'places')),
    ('我已经知道顾绮言约在哪了，把他那册书的原名补给我。',
     '我已经知道顾绮言那册叫什么了，把他约的碰头处补给我。', MemoryAddress('顾绮言', 'books')),
    ('碰头的地方另定吧，何青玟那件需要捎的还是原来的，报个名称。',
     '要捎的东西另外定吧，何青玟碰头这项还是原来的，报个名称。', MemoryAddress('何青玟', 'items')),
    ('给黎书然备原来那种喝的，借阅那栏我会自己问。',
     '给黎书然备点没喝过的，借阅那栏我会自己问。', MemoryAddress('黎书然', 'drinks')),
    ('赵茗初之前登记在哪儿见？聊完用品我还是想核一下碰头地址。',
     '赵茗初之前登记在哪儿见？先撤掉这个问题，我这里只聊选用品的方法。', MemoryAddress('赵茗初', 'places')),
    ('借阅这项暂时放下，刚想起来还是得说一下许昀芝过去挑的书叫什么。',
     '刚想起来许昀芝过去挑的书叫什么，不过借阅这项暂时放下。', MemoryAddress('许昀芝', 'books')),
)


def features(extract, examples):
    texts = [f'给{a.name}处理请求。用户说：{q}\n需要读取{a.name}的{DESCRIPTIONS[a.field]}吗？'
             for q, a in examples]
    vectors = []
    for begin in range(0, len(texts), 24):
        batch = texts[begin:begin + 24]
        hidden = extract(batch)
        vectors.extend(hidden[index, len(text) - 1].clone() for index, text in enumerate(batch))
    return torch.stack(vectors).float()


def fit(vectors, *, pair_weight):
    torch.manual_seed(SEED)
    head = field_head(vectors.shape[-1])
    rng = torch.Generator().manual_seed(SEED)
    opt = torch.optim.AdamW(head.parameters(), lr=.002)
    targets = torch.tensor([1., 0.])
    for _ in range(12):
        for batch in torch.randperm(len(vectors), generator=rng).split(24):
            logits = head(vectors[batch]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, targets.expand_as(logits))
            if pair_weight:
                loss = loss + pair_weight * F.softplus(1. - (logits[:, 0] - logits[:, 1])).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return head.eval()


@torch.inference_mode()
def measure(store, model, tokenizer, reader, codebook, heads, extract):
    catalog = store.catalog()
    addresses = sorted(catalog)
    values = profile_values(catalog)
    if not all(address in catalog for _, _, address in FRESH):
        raise ValueError('A paired candidate is missing from the archive')
    pair_vectors = features(extract, [(q, a) for yes, no, a in FRESH for q in (yes, no)])
    cases = []
    for sample in HELD:
        candidates = tuple(address for address in addresses if address.name in sample.question)
        expected = resolve_current(catalog, [MemoryAddress(sample.owner, field) for field in sample.fields])
        cases.append((sample.question, candidates, expected))
    case_vectors = iter(features(extract, [(q, a) for q, candidates, _ in cases for a in candidates]))
    metrics = {name: Counter() for name in heads}
    out = []
    pair_results = {}
    for name, head in heads.items():
        pair_scores = head(pair_vectors).reshape(-1, 2).tolist()
        metrics[name]['pair_rank'] = sum(a > b for a, b in pair_scores)
        metrics[name]['pair_both'] = sum(a >= 0 > b for a, b in pair_scores)
        pair_results[name] = pair_scores
    for question, candidates, expected in cases:
        rows = [next(case_vectors) for _ in candidates]
        vectors = torch.stack(rows) if rows else None
        if any(value in question for value in values.values()):
            raise ValueError('A memory value is already in the request')
        arms, cached = {}, {}
        for name, head in heads.items():
            positive = [] if vectors is None else head(vectors).squeeze(-1).ge(0).tolist()
            selected = [address for address, yes in zip(candidates, positive, strict=True) if yes]
            route = resolve_current(catalog, selected) if 1 <= len(selected) <= 2 else ()
            if route not in cached:
                cached[route] = read_reply(model, tokenizer, store, codebook, reader, question, route, expected, values)
            actual = cached[route]
            correct = set(route) == set(expected)
            metrics[name].update(exact_positive=bool(expected) and correct,
                                 correct_values=bool(expected) and correct and actual['value_pass'],
                                 false_read=not expected and bool(route), missing=len(set(expected) - set(route)),
                                 extra=len(set(route) - set(expected)))
            arms[name] = {'route': route, 'proposed': [(a.name, a.field) for a in selected], **actual}
        out.append({'question': question, 'expected': expected, 'arms': arms})
    return metrics, out, pair_results


def main(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    db, read_head = args.records / 'versions.sqlite3', args.reader / 'head.pt'
    record_report = args.records / 'result.json'
    reader_report = args.reader / 'result.json'
    record_meta, reader_meta = [json.loads(path.read_text(encoding='utf-8')) for path in (record_report, reader_report)]
    paths = (db, read_head, record_report, reader_report, MODEL / 'model.safetensors',
             MODEL / 'tokenizer.json', MODEL / 'tokenizer_config.json', Path(__file__))
    sources = {path: sha(path) for path in paths}
    if (sources[db], sources[read_head], sources[MODEL / 'model.safetensors']) != (
        record_meta['database_sha256'], reader_meta['head_sha256'], WEIGHT_SHA
    ):
        raise ValueError('Unexpected study artifacts')
    model = load_full_streamed()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer)
    pairs = list(dict.fromkeys(training_pairs()))
    train = [(q, a) for yes, no, a in pairs for q in (yes, no)]
    held_questions = {r.question for r in HELD} | {q for yes, no, _ in FRESH for q in (yes, no)}
    if {q for q, _ in train} & held_questions or {a.name for _, _, a in FRESH} & set(TRAIN_NAMES):
        raise ValueError('Training and paired evaluation overlap')
    vectors = features(extract, train).reshape(len(pairs), 2, model.config.hidden_size)
    heads = {name: fit(vectors, pair_weight=weight) for name, weight in (('binary', 0.), ('paired', .5))}
    state = torch.load(read_head, map_location='cpu', weights_only=True)
    reader = VersionReplyActions(model.config.hidden_size, state['scaffold_tokens'], slots=2)
    reader.load_state_dict(state)
    embedding = model.get_input_embeddings().weight
    columns = compress_dimension(embedding.shape[1]).to(embedding.device)
    codebook = F.normalize(embedding.index_select(1, columns).float(), dim=-1)
    writer = next(digest for path, digest in record_meta['sources'].items() if Path(path).name == 'adapters.pt')
    with VersionStore(db, BINDING_PREFIX + writer + ':' + WEIGHT_SHA) as store:
        metrics, held, paired = measure(store, model, tokenizer, reader.eval().to(model.device), codebook, heads, extract)
    if any(sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('A sealed source changed')
    head_path = args.output / 'heads.pt'
    atomic_save({name: head.state_dict() for name, head in heads.items()}, head_path)
    store_json(args.output / 'result.json', {
        'metrics': metrics, 'training_pairs': len(pairs), 'held': held,
        'fresh_pairs': [(yes, no, a.name, a.field) for yes, no, a in FRESH], 'paired': paired,
        'head_sha256': sha(head_path), 'sources': {str(path): digest for path, digest in sources.items()},
        'seconds': perf_counter() - started,
    })
    print(metrics, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('records', type=Path)
    parser.add_argument('reader', type=Path)
    parser.add_argument('output', type=Path)
    main(parser.parse_args())
