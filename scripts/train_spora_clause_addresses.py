"""Compare supplied and automatic clause spans for matching version addresses."""

import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from scripts.spora_version_chat import FrozenTurnFeatures, base, local_model
from scripts.train_spora_address_sets import deletion_summary, matched_rows
from scripts.train_spora_candidate_relevance import FRAMES, HELD, SEED, TRAIN, UNAVAILABLE, features, suite
from scripts.train_spora_recurrent_counterfactual import sha, store_json
from spora.language_training import atomic_save
from spora.research_version_roles import clause_spans


def examples(rows, mode):
    data = []
    for row in rows:
        question = row['question']
        if mode == 'provided':
            parts = [part for part, _ in row['parts']]
        else:
            if mode == 'conjunction':
                question = '请帮我核实' + '还有'.join(part for part, _ in row['parts']) + '。'
            parts = [question[start:end].strip() for start, end in clause_spans(question)]
        data.extend(dict(question=text, candidates=row['candidates']) for text in parts if text)
    return data


def fit_head(inputs, labels):
    torch.manual_seed(SEED)
    model = nn.Sequential(nn.LayerNorm(inputs.shape[-1]), nn.Linear(inputs.shape[-1], 24),
                          nn.GELU(), nn.Linear(24, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.03)
    for _ in range(160):
        logits = F.pad(model(inputs).squeeze(-1), (0, 1))
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval().requires_grad_(False)


def score_rows(rows, clauses, logits):
    totals = {}
    cases = []
    start = 0
    for row, parts in zip(rows, clauses, strict=True):
        scores = logits[start:start + len(parts)].tolist()
        start += len(parts)
        picks = [max(range(len(score)), key=score.__getitem__) for score in scores]
        selected = {index for index in picks if index < len(row['candidates'])}
        truth = set(row['target'])
        count = totals.setdefault(str(len(truth)), Counter())
        count.update(total=1, exact=selected == truth, count_correct=len(selected) == len(truth),
                     accepted=len(selected & truth), erroneous=len(selected - truth), clauses=len(parts))
        cases.append({**row, 'selected': sorted(selected), 'segments': [part['question'] for part in parts],
                      'scores': scores, 'picks': picks})
    if start != len(logits):
        raise RuntimeError('Scored clauses could not be assigned to their requests')
    return dict(totals=totals, rows=cases)


def main(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {path: sha(path) for path in (args.checkpoint, *(folder / name for folder in args.layers
                                                            for name in ('attention.pt', 'comparison.json')))}
    torch.set_num_threads(2)
    training = suite(TRAIN, FRAMES[0], UNAVAILABLE[0], annotate=True) + matched_rows(0, annotate=True)
    evaluation = dict(ordinary=suite(HELD, FRAMES[2], UNAVAILABLE[2], annotate=True),
                      matched=matched_rows(2, annotate=True), fresh=matched_rows(2, fresh=True, annotate=True))
    modes = ('provided', 'punctuation', 'conjunction')
    groups = {(split, mode): [examples([row], mode) for row in rows]
              for split, rows in evaluation.items() for mode in modes}
    flat = examples(training, 'provided')
    labels = torch.tensor([4 if index is None else index for row in training for _, index in row['parts']])
    if len(flat) != len(labels) or set(labels.tolist()) != set(range(5)):
        raise ValueError('Training data must cover every address position and abstention')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    with local_model([args.checkpoint, *args.layers]) as model:
        extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=23, single_sequence=True)
        inputs = features(extract, flat)
        # Cache repeated request/candidate combinations across the segmentation conditions.
        unique = {}
        for group in groups.values():
            for parts in group:
                for part in parts:
                    unique.setdefault((part['question'], tuple(part['candidates'])), part)
        hidden = dict(zip(unique, features(extract, list(unique.values())), strict=True))
    head = fit_head(inputs, labels)
    artifact = args.output / 'clause_addresses.pt'
    atomic_save(head.state_dict(), artifact)
    head.load_state_dict(torch.load(artifact, map_location='cpu', weights_only=True), strict=True)
    results = {}
    with torch.no_grad():
        for (split, mode), clauses in groups.items():
            data = torch.stack([hidden[part['question'], tuple(part['candidates'])]
                                for parts in clauses for part in parts])
            logits = F.pad(head(data).squeeze(-1), (0, 1))
            if not torch.equal(logits[:, :4].flip(dims=(1,)).argmax(dim=1),
                               head(data.flip(dims=(1,))).squeeze(-1).argmax(dim=1)):
                raise RuntimeError('Reversing candidate order changed the underlying field scores')
            result = score_rows(evaluation[split], clauses, logits)
            if split in ('matched', 'fresh'):
                result['deletions'] = deletion_summary(result['rows'])
            results.setdefault(split, {})[mode] = result
            print(split, mode, result['totals'], result.get('deletions'), flush=True)
    if any(sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('A source checkpoint changed')
    store_json(args.output / 'result.json', dict(results=results, checkpoint_sha256=sha(artifact),
               training_clauses=len(flat), sources={str(path): digest for path, digest in sources.items()},
               seconds=perf_counter() - started))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('layers', type=Path, nargs=2)
    main(cli.parse_args())
