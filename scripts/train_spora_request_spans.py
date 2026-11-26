"""Learn request spans and ablate hard negatives in clause-address matching."""

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
from scripts.train_spora_candidate_relevance import CALIBRATION, FRAMES, HELD, SEED, TRAIN, UNAVAILABLE, features, suite
from scripts.train_spora_clause_addresses import examples, fit_head, score_rows
from scripts.train_spora_recurrent_counterfactual import sha, store_json
from spora.language_training import atomic_save


def rephrase(rows, frame):
    return [{**row, 'question': frame.format(*(part for part, _ in row['parts']))} for row in rows]


def tags(rows, width):
    result = torch.full((len(rows), width), -100, dtype=torch.long)
    for row, labels in zip(rows, result, strict=True):
        text = row['question']
        labels[:len(text)] = 0
        previous_end = 0
        for part, _ in row['parts']:
            begin = text.index(part, previous_end)
            previous_end = begin + len(part)
            labels[begin] = 1
            labels[begin + 1:previous_end] = 2
    return result


class RequestSpans(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 48), nn.GELU())
        self.context = nn.GRU(48, 32, batch_first=True, bidirectional=True)
        self.classify = nn.Linear(64, 3)

    def forward(self, hidden, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(self.project(hidden), lengths, batch_first=True, enforce_sorted=False)
        vectors, _ = self.context(packed)
        padded, _ = nn.utils.rnn.pad_packed_sequence(vectors, batch_first=True, total_length=hidden.shape[1])
        return self.classify(padded)


def span_features(extract, rows):
    texts = list(dict.fromkeys(row['question'] for row in rows))
    hidden = extract(texts).float()
    index = {text: i for i, text in enumerate(texts)}
    return hidden[[index[row['question']] for row in rows]], [len(row['question']) for row in rows]


def fit_spans(hidden, lengths, labels):
    torch.manual_seed(SEED)
    head = RequestSpans(hidden.shape[-1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    for _ in range(100):
        logits = head(hidden, lengths)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), weight=torch.tensor([1., 3., 1.]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return head.eval().requires_grad_(False)


def predicted_parts(text, labels):
    parts, start = [], None
    for index, label in enumerate([*labels, 0]):
        if start is not None and label != 2:
            parts.append(text[start:index])
            start = None
        if label == 1 or (label == 2 and start is None):
            start = index
    return parts


def save_reload(module, path):
    atomic_save(module.state_dict(), path)
    module.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    return sha(path)


def address_labels(rows):
    return torch.tensor([4 if target is None else target for row in rows for _, target in row['parts']])


def abstention_metrics(logits, labels, threshold):
    score, address = logits.max(dim=-1)
    selected = score.gt(threshold)
    missing = labels.eq(logits.shape[-1])
    correct = selected & address.eq(labels)
    return dict(present=int((~missing).sum()), missing=int(missing.sum()), correct=int(correct.sum()),
                wrong=int((selected & ~correct).sum()), missing_wrong=int((selected & missing).sum()))


def calibrate_abstention(head, calibration, labels):
    logits = head(calibration).squeeze(-1)
    thresholds = [0., *sorted(set(x for x in logits.amax(dim=-1).tolist() if x > 0))]

    def utility(threshold):
        metrics = abstention_metrics(logits, labels, threshold)
        return metrics['correct'] - metrics['wrong'], -metrics['wrong'], -threshold

    return max(thresholds, key=utility)


def main(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {path: sha(path) for path in (args.checkpoint, *(folder / part for folder in args.layers
                                                            for part in ('attention.pt', 'comparison.json')))}
    torch.set_num_threads(2)
    train = suite(TRAIN, FRAMES[0], UNAVAILABLE[0], annotate=True) + matched_rows(0, annotate=True)
    hard_train = matched_rows(1, annotate=True)
    span_train = train + rephrase(train, '请核对{}还有{}。') + rephrase(train, '我想知道{}以及{}。')
    fresh = matched_rows(2, fresh=True, annotate=True)
    evaluation = dict(ordinary=suite(HELD, FRAMES[2], UNAVAILABLE[2], annotate=True),
                      matched=matched_rows(2, annotate=True), fresh=fresh,
                      conjunction=rephrase(fresh, '帮忙确认{}再告诉我{}。'))
    span_sets = {'learned': span_train}
    calibration = []
    if args.boundaries:
        span_sets['varied'] = span_train + rephrase(train, '我打算把{}和{}记在交接纸上。') + rephrase(
            train, '请告诉我{}并核实{}，随后我会转给别人。')
        evaluation['handoff'] = rephrase(fresh, '麻烦把{}和{}查清楚，我来填交接单。')
        evaluation['append'] = rephrase(fresh, '请报{}并顺带说说{}，我整理好再发。')
        calibration = suite(CALIBRATION, FRAMES[1], UNAVAILABLE[1], annotate=True)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    with local_model([args.checkpoint, *args.layers]) as model:
        extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=23, single_sequence=True)
        segmenters, hashes = {}, {}
        for name, rows in span_sets.items():
            hidden, lengths = span_features(extract, rows)
            segmenters[name] = fit_spans(hidden, lengths, tags(rows, hidden.shape[1]))
            del hidden
            path = args.output / ('spans.pt' if name == 'learned' else f'{name}.pt')
            hashes[path.stem] = save_reload(segmenters[name], path)
        groups, span_metrics = {}, {}
        for split, rows in evaluation.items():
            hidden, lengths = span_features(extract, rows)
            groups[split] = dict(provided=[examples([row], 'provided') for row in rows],
                                 punctuation=[examples([row], 'punctuation') for row in rows])
            metrics = {}
            for name, segmenter in segmenters.items():
                labels = segmenter(hidden, lengths).argmax(dim=-1).tolist()
                predicted = [predicted_parts(row['question'], prediction[:length])
                             for row, prediction, length in zip(rows, labels, lengths, strict=True)]
                metrics[name] = dict(total=len(rows), exact=sum(
                    parts == [part for part, _ in row['parts']] for row, parts in zip(rows, predicted, strict=True)))
                groups[split][name] = [[dict(question=part, candidates=row['candidates']) for part in parts]
                                       for row, parts in zip(rows, predicted, strict=True)]
            span_metrics[split] = metrics if args.boundaries else metrics['learned']
            print(split, span_metrics[split], flush=True)
        train_data = features(extract, examples(train + hard_train, 'provided'))
        calibration_data = features(extract, examples(calibration, 'provided')) if calibration else None
        unique = {(part['question'], tuple(part['candidates'])): part for modes in groups.values()
                  for group in modes.values() for parts in group for part in parts}
        vectors = dict(zip(unique, features(extract, list(unique.values())), strict=True))
    labels = address_labels(train + hard_train)
    regular = len(address_labels(train))
    heads = dict(base=fit_head(train_data[:regular], labels[:regular]), augmented=fit_head(train_data, labels))
    results, abstention = {}, {}
    for name, head in heads.items():
        hashes[name] = save_reload(head, args.output / f'{name}.pt')
        results[name] = {}
        if args.boundaries:
            calibration_labels = address_labels(calibration)
            threshold = calibrate_abstention(head, calibration_data, calibration_labels)
            abstention[name] = dict(threshold=threshold, splits={})
        for split, modes in groups.items():
            results[name][split] = {}
            for mode, group in modes.items():
                flat = [vectors[part['question'], tuple(part['candidates'])] for parts in group for part in parts]
                logits = F.pad(head(torch.stack(flat)).squeeze(-1), (0, 1)) if flat else torch.empty(0, 5)
                result = score_rows(evaluation[split], group, logits)
                if args.boundaries and mode == 'provided':
                    expected = address_labels(evaluation[split])
                    abstention[name]['splits'][split] = dict(
                        default=abstention_metrics(logits[:, :4], expected, 0.),
                        calibrated=abstention_metrics(logits[:, :4], expected, threshold))
                if split != 'ordinary':
                    result['deletions'] = deletion_summary(result['rows'])
                results[name][split][mode] = result
                numbers = Counter()
                for bucket in result['totals'].values():
                    numbers.update(bucket)
                print(name, split, mode, numbers, flush=True)
    if any(sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('A protected checkpoint changed')
    store_json(args.output / 'result.json', dict(results=results, spans=span_metrics, checkpoints=hashes,
               abstention=abstention, span_training_counts={name: len(rows) for name, rows in span_sets.items()},
               training_requests=len(span_train), added_hard_clauses=len(labels) - regular,
               sources={str(path): digest for path, digest in sources.items()}, seconds=perf_counter() - started))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('layers', type=Path, nargs=2)
    cli.add_argument('--boundaries', action='store_true', help='Compare varied boundary training and calibrated abstention')
    main(cli.parse_args())
