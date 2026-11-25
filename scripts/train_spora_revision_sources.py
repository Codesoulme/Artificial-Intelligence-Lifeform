"""Learn the status of source-linked dialogue values in the DeepSeek SPORA decoder."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_spora_dialogue_evidence import load_spora_for_eval, store_json
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.deepseek_local_baseline import sha, torch
from scripts.spora_revision_source_data import ARCHIVE, HELD, NULL, Status, training_rows
from scripts.train_spora_revision_ranker import summarize_saved_responses
from spora.language_training import atomic_save
from torch import nn
from torch.nn import functional as F


FORMAT = 'spora-revision-sources-v1'
SEED = 20261126
EPOCHS = 26
VIEWS = ('source_spans', 'source_turns', 'no_latest')


def layout(row):
    name, turns, terms, roles = row
    annotated = [f'用户第{index}轮：{text}\n' for index, text in enumerate(turns, 1)]
    starts = [0, len(annotated[0]), len(annotated[0]) + len(annotated[1])]
    body_starts = [start + len(part) - len(text) - 1 for start, part, text in zip(starts, annotated, turns, strict=True)]
    sources = []
    for term in terms:
        match = next(((index, text.index(term)) for index, text in enumerate(turns[:2]) if term in text), None)
        if match is None:
            raise ValueError(f'Missing source for {name}: {term}')
        index, offset = match
        begin = body_starts[index] + offset
        whole = slice(body_starts[index], body_starts[index] + len(turns[index]))
        sources.append((slice(begin, begin + len(term)), whole, index))
    return ''.join(annotated), sources, slice(body_starts[2], body_starts[2] + len(turns[2])), roles


def capture(extractor, rows, *, arrange=layout):
    layouts = [arrange(row) for row in rows]
    width = max(item[2].stop - item[2].start for item in layouts)
    exact, whole, order, latest, masks, gold = [], [], [], [], [], []
    for start in range(0, len(layouts), 12):
        batch = layouts[start:start + 12]
        activations = extractor([row[0] for row in batch]).float()
        for vector, (_, sources, last, roles) in zip(activations, batch, strict=True):
            exact.append(torch.stack([vector[span].mean(dim=0) for span, _, _ in sources]))
            whole.append(torch.stack([vector[turn].mean(dim=0) for _, turn, _ in sources]))
            order.append([index for _, _, index in sources])
            recent = vector[last]
            latest.append(F.pad(recent, (0, 0, 0, width - len(recent))))
            masks.append(torch.arange(width) < len(recent))
            gold.append(roles)
    return {key: torch.stack(value) for key, value in (('source_spans', exact), ('source_turns', whole),
                                                      ('latest', latest), ('mask', masks))} | {
        'order': torch.tensor(order), 'roles': torch.tensor(gold)}


class SourceStatusHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        reduced = 48
        self.source = nn.Sequential(nn.LayerNorm(width, elementwise_affine=False), nn.Linear(width, reduced))
        self.recent = nn.Sequential(nn.LayerNorm(width, elementwise_affine=False), nn.Linear(width, reduced))
        self.classify = nn.Sequential(nn.Linear(4 * reduced + 1, 72), nn.GELU(), nn.Linear(72, len(Status)))

    def forward(self, sources, latest, mask, order):
        source, query = self.source(sources), self.recent(latest)
        similarity = source @ query.transpose(-1, -2) / (query.shape[-1] ** .5)
        attention = similarity.masked_fill(~mask[:, None], float('-inf')).softmax(dim=-1)
        support = attention @ query
        recent = (query * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        return self.classify(torch.cat((source, support, source * support, recent[:, None].expand_as(source),
                                        order.unsqueeze(-1)), dim=-1))


def arguments(data, view, subset=slice(None)):
    source = data['source_turns' if view == 'source_turns' else 'source_spans'][subset]
    latest, mask = data['latest'][subset], data['mask'][subset]
    if view == 'no_latest':
        latest = torch.zeros_like(latest[:, :1])
        mask = torch.ones_like(mask[:, :1])
    return source, latest, mask, data['order'][subset]


def active_scores(logits):
    return logits[..., Status.ACTIVE] - logits[..., 1:].logsumexp(dim=-1)


def train(head, features, view):
    statuses = features['roles']
    rng = torch.Generator().manual_seed(SEED)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.0015, weight_decay=.02)
    trace = []
    for _ in range(EPOCHS):
        losses = []
        for subset in torch.randperm(len(statuses), generator=rng).split(24):
            gold = statuses[subset]
            logits = head(*arguments(features, view, subset))
            loss = F.cross_entropy(logits.flatten(0, 1), gold.flatten())
            paired = (gold == Status.ACTIVE).sum(dim=1) == 1
            if paired.any():
                target = (gold[paired] == Status.ACTIVE).long().argmax(dim=1)
                loss = loss + .5 * F.cross_entropy(active_scores(logits[paired]), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach().item())
        trace.append(sum(losses) / len(losses))
    return trace


@torch.no_grad()
def evaluate(head, features, view, *, paired):
    inputs = arguments(features, view)
    logits = head(*inputs)
    reversed_logits = head(inputs[0].flip(1), *inputs[1:3], inputs[3].flip(1)).flip(1)
    if not torch.allclose(logits, reversed_logits, atol=1e-5, rtol=1e-5):
        raise ValueError('Changing candidate order changed the status scores')
    roles, gold = logits.argmax(dim=-1), features['roles']
    marked = roles == Status.ACTIVE
    active_count = marked.sum(dim=1)
    scores = active_scores(logits)
    forced = scores.argmax(dim=1)
    selected = torch.where(active_count == 1, marked.long().argmax(dim=1), -1)
    expected = (gold == Status.ACTIVE).long().argmax(dim=1) if paired else torch.full_like(selected, -1)
    correct = forced == expected
    roles_correct = {status.name: [int(((roles == gold) & (gold == status)).sum()), int((gold == status).sum())]
                     for status in Status}
    strict = (selected == expected) if paired else (active_count == 0)
    return {'role_counts': roles_correct, 'strict_correct': int(strict.sum()),
            'forced_correct': int(correct.sum()) if paired else None,
            'complete_pairs': int(correct.view(-1, 2).all(dim=1).sum()) if paired else None,
            'forced': forced.tolist(), 'selected': selected.tolist(), 'roles': roles.tolist(),
            'gold': gold.tolist(), 'scores': scores.tolist(), 'marked_active': marked.tolist(),
            'without_active': int((active_count == 0).sum()), 'conflicting_active': int((active_count > 1).sum())}


def replay(previous, predictions):
    payload = json.loads(previous.read_text(encoding='utf-8'))
    old = summarize_saved_responses(payload)['summary']['trained']
    if len(payload['rows']) != len(predictions):
        raise ValueError('Saved replies do not match the source evaluation')
    for index in range(0, len(predictions), 2):
        pair = payload['rows'][index:index + 2]
        short = tuple(row['outputs']['oracle'] for row in pair)
        for offset, row in enumerate(pair):
            picked = predictions[index + offset]
            row['selected'] = picked
            row['outputs']['selected'] = short[picked]
    current = summarize_saved_responses(payload)
    return {'source_sha256': sha(previous), 'prior_ranker': old,
            'candidate': current['summary']['trained'], 'rows': current['rows']}


def main(args):
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=True)
    destination = target / ('reload.json' if args.verify else 'result.json')
    weights = target / 'source-heads.pt'
    if destination.exists() or (not args.verify and weights.exists()):
        raise FileExistsError('Use a fresh output directory')
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    begin = perf_counter()
    training = training_rows()
    if {item[1] for item in training}.intersection(item[1] for item in (*ARCHIVE, *HELD, *NULL)):
        raise ValueError('An evaluation dialogue appears in training')
    digest = sha256(json.dumps((training, ARCHIVE, HELD, NULL), ensure_ascii=False).encode()).hexdigest()
    chat = load_spora_for_eval(target)
    extractor = FrozenTurnFeatures(chat.decoder, chat.tokenizer)
    datasets = {name: capture(extractor, rows) for name, rows in
                (('archive', ARCHIVE), ('held', HELD), ('no_active', NULL))}
    width = datasets['held']['source_spans'].shape[-1]
    heads, losses = {}, {}
    if args.verify:
        saved = torch.load(weights, map_location='cpu', weights_only=True)
        if (saved['format'], saved['runtime'], saved['data']) != (FORMAT, chat.awake_individual, digest):
            raise ValueError('Source heads were trained with different inputs')
    else:
        features = capture(extractor, training)
        print('captured', len(training), 'training dialogues', flush=True)
    for view in VIEWS:
        torch.manual_seed(SEED)
        head = SourceStatusHead(width)
        if args.verify:
            head.load_state_dict(saved['heads'][view])
        else:
            losses[view] = train(head, features, view)
        heads[view] = head.eval()
    if not args.verify:
        atomic_save({'format': FORMAT, 'runtime': chat.awake_individual, 'data': digest,
                     'heads': {view: head.state_dict() for view, head in heads.items()}}, weights)
    evaluations = {view: {name: evaluate(head, data, view, paired=name != 'no_active')
                          for name, data in datasets.items()} for view, head in heads.items()}
    print({view: {name: (data['forced_correct'], data['complete_pairs'], data['strict_correct'])
                  for name, data in result.items()} for view, result in evaluations.items()}, flush=True)
    if args.verify:
        reference = json.loads((target / 'result.json').read_text(encoding='utf-8'))
        for view, results in evaluations.items():
            if any(result['roles'] != reference['evaluations'][view][name]['roles'] or
                   result['forced'] != reference['evaluations'][view][name]['forced'] for name, result in results.items()):
                raise ValueError('Reload changed the source decisions')
    response_replay = replay(args.replies, evaluations['source_spans']['held']['forced'])
    if args.verify and response_replay['source_sha256'] != reference['response_replay']['source_sha256']:
        raise ValueError('The response replay source changed')
    response_controls = {view: replay(args.replies, evaluations[view]['held']['forced'])['candidate']
                         for view in VIEWS if view != 'source_spans'}
    store_json(destination, {'data_sha256': digest, 'runtime': chat.awake_individual, 'weights_sha256': sha(weights),
                             'train_count': len(training), 'losses': losses, 'evaluations': evaluations,
                             'response_replay': response_replay, 'response_controls': response_controls,
                             'seconds': perf_counter() - begin,
                             'peak_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('replies', type=Path)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
