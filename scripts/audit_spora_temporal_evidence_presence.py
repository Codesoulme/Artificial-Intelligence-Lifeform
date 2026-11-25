"""Test whether a separate evidence check can stop unsupported temporal reads."""

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

from scripts import audit_spora_spoken_query_plan as plan
from scripts import audit_spora_spoken_temporal_head as prior
from scripts.audit_spora_natural_lookup_cli import digest
from scripts.spora_version_chat import (FIELD_IDS, FrozenTurnFeatures, VersionReplyActions, VersionStore,
                                        artifacts, base, compress_dimension, local_model, reply_to_route)
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_multi_version_route import resolve_address
from spora.research_version_store import MemoryAddress


TRAIN_ENDS = ('请报那一项的内容', '我想核实一下具体是什么', '想确认究竟是哪一个', '那件事还记着吗')
CAL_ENDS = ('可以给我一个具体答案吗', '方便告诉我那份是什么吗')
DELETION_PAIRS = (
    (0, prior.CURRENT, '顺手给罗映荷带杯喝的。', '这会儿照她刚说定的那种买，叫什么？', '需要买哪一种？'),
    (1, prior.PREVIOUS, '我想向罗映禾核实之前约见的地方。', '就问这回迁走以后废掉的那个点。', '之前的地点是什么？'),
    (2, prior.FIRST, '穆云川让我顺便找一下提过的物件。', '我要的是最开始打算拿的那样。', '她到底要的是哪件？'),
    (3, prior.CURRENT, '提到叶盼山选的书，我还得找一下。', '照她最后定下的那一本找。', '她订的那本究竟叫什么？'),
)


def evidence_pairs(cues, ends):
    rows = []
    for index, versions in enumerate(cues):
        label = tuple(FIELD_IDS)[index % len(FIELD_IDS)]
        for ending in ends:
            stem = f'甲在我这儿留过{label}。{ending}。'
            for version in versions:
                rows.append((stem + version + '？', True))
            rows.extend(((stem, False), (stem + f'{versions[0]}，还是{versions[1]}？我还要确定。', False),
                         (stem + f'{versions[2]}？这个我只是打个比方，还没说要用哪次。', False)))
    return rows


def fit(features, labels):
    torch.manual_seed(70821)
    rng = torch.Generator().manual_seed(70821)
    head = nn.Sequential(nn.LayerNorm(features.shape[-1]), nn.Linear(features.shape[-1], 48),
                         nn.GELU(), nn.Linear(48, 1))
    opt = torch.optim.AdamW(head.parameters(), lr=.002)
    for _ in range(15):
        for index in torch.randperm(len(labels), generator=rng).split(32):
            score = head(features[index]).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(score, labels[index])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return head.eval()


def check_pairs():
    rows = []
    for group, target, prefix, full, incomplete in DELETION_PAIRS:
        rows.extend(((group, target, prefix + full), (group, prior.UNCLEAR, prefix + incomplete)))
    return rows


@torch.inference_mode()
def evaluate(model, tokenizer, store, reader, temporal, evidence, cut, requests):
    catalog = store.catalog()
    embedding = model.get_input_embeddings().weight
    columns = compress_dimension(embedding.shape[1]).to(embedding.device)
    codebook = F.normalize(embedding.index_select(1, columns).float(), dim=-1)
    values = [value for _, _, value in prior.RECORDS]
    totals = {mode: Counter() for mode in ('raw', 'guarded')}
    rows = []
    for (group, target, question), (guess, _), confidence in zip(requests, temporal, evidence, strict=True):
        if any(value in question for value in values):
            raise ValueError('A question contains a stored value')
        owner, field, _ = prior.TIMELINES[group]
        expected = prior.record_at(group, target)
        query = plan.VERSIONS[guess] if guess != prior.UNCLEAR else None
        record, _ = resolve_address(catalog, MemoryAddress(owner, FIELD_IDS[field]), query, date.today(), retain_same_day=True)
        results = {}
        for mode, opened in (('raw', record is not None), ('guarded', record is not None and confidence >= cut)):
            found = record if opened else None
            answer = None
            if opened:
                try:
                    _, output = reply_to_route(model, tokenizer, store, codebook, reader, question, [record], labels=[field])
                    answer = output['answer']
                except ValueError as error:
                    answer = str(error)
            shown = [value for value in values if answer and value in answer]
            correct = found is not None and found == expected and shown == [values[found - 1]]
            totals[mode].update(exact_read=correct, wrong_read=opened and found != expected,
                                correctly_asked=target == prior.UNCLEAR and not opened, no_read=not opened)
            results[mode] = {'record': found, 'answer': answer if opened else None, 'correct': correct}
        rows.append(dict(question=question, expected=expected, guessed=guess, evidence=confidence, arms=results))
    return rows, totals


def main(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    report = json.loads((args.prior / 'result.json').read_text(encoding='utf-8'))
    model_file = args.prior / 'mean.pt'
    database = args.prior / 'profile' / 'versions.sqlite3'
    if digest(model_file) != report['sha256']['mean'] or digest(database) != report['sources'][str(database)]:
        raise ValueError('Prior assets changed')
    reader_file, _, _, binding = artifacts(args.reader, args.normalizer)
    paths = (model_file, database, reader_file, Path(__file__), Path(prior.__file__))
    sources = {path: digest(path) for path in paths}
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    train = evidence_pairs(prior.TEMPORAL_TRAIN, TRAIN_ENDS)
    cal = evidence_pairs(prior.TEMPORAL_CAL, CAL_ENDS)
    requests = check_pairs()
    names = {owner for owner, *_ in prior.TIMELINES}
    selected = [plan.anonymize(question, names)[0].rsplit('\n', 1)[0] for _, _, question in requests]
    if {text for text, _ in train} & ({text for text, _ in cal} | set(selected)):
        raise ValueError('Evidence supervision overlaps with evaluation')
    with local_model(args.recurrent) as model, VersionStore(database, binding) as store:
        extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=plan.DEPTH)
        train_x, _ = plan.vectors(extract, [text for text, _ in train], (), 'user-span')
        cal_x, _ = plan.vectors(extract, [text for text, _ in cal], (), 'user-span')
        test_x, _ = plan.vectors(extract, selected, (), 'user-span')
        gate = fit(train_x, torch.tensor([target for _, target in train], dtype=torch.float32))
        head = prior.TemporalHead(model.config.hidden_size, focus=False)
        head.load_state_dict(torch.load(model_file, map_location='cpu', weights_only=True))
        test = prior.extracted(extract, [(text, None, None) for text in selected])
        with torch.inference_mode():
            cal_probs = gate(cal_x).sigmoid().squeeze(1).tolist()
            test_probs = gate(test_x).sigmoid().squeeze(1).tolist()
        missing = max(prob for prob, (_, yes) in zip(cal_probs, cal, strict=True) if not yes)
        candidates = [x for x in (0., .5, .6, .7, .8, .9, .95, .99, 1.) if x > missing]
        threshold = min(candidates) if candidates else 1.000001
        state = torch.load(reader_file, map_location='cpu', weights_only=True)
        reader = VersionReplyActions(model.config.hidden_size, state['scaffold_tokens'], slots=2)
        reader.load_state_dict(state)
        rows, metrics = evaluate(model, tokenizer, store, reader.eval().to(model.device), prior.prediction(head.eval(), test),
                                 test_probs, threshold, requests)
    if any(digest(path) != expected for path, expected in sources.items()):
        raise ValueError('Source files changed')
    gate_path = args.output / 'evidence.pt'
    atomic_save(gate.state_dict(), gate_path)
    store_json(args.output / 'result.json', dict(rows=rows, metrics=metrics, threshold=threshold,
               training=len(train), calibration=len(cal), calibration_positive=sum(p >= threshold and expected
              for p, (_, expected) in zip(cal_probs, cal, strict=True)), head_sha256=digest(gate_path),
               sources={str(path): value for path, value in sources.items()}, seconds=perf_counter() - started))
    print(metrics, threshold, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reader', type=Path, default=Path('runs/spora-dual-version-reply-v1'))
    parser.add_argument('--normalizer', type=Path, default=Path('runs/spora-version-normalizer-v1'))
    parser.add_argument('--recurrent', required=True, nargs=3, type=Path)
    main(parser.parse_args())
