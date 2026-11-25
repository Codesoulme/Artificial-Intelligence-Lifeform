"""Learn zero-, one- and two-offer decisions from frozen SPORA clause features."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_spora_crosswoz_revisions import CONTACT, validation
from scripts.audit_spora_dialogue_evidence import load_spora_for_eval, store_json
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.deepseek_local_baseline import sha, torch
from scripts.spora_joint_offer_data import REAL, REAL_FRESH, partial_samples, samples
from scripts.train_spora_offer_clauses import capture, make_head
from spora.language_training import atomic_save
from torch import nn
from torch.nn import functional as F


SEED = 20261129
EPOCHS = 38
MODES = ('independent', 'joint')
STATE_NAMES = ('neither', 'first', 'second', 'both')


def natural_scenarios(path, reviewed=REAL, group='real'):
    corpus = validation(path)
    selected = []
    for dialogue, turn, choice in reviewed:
        offer, user_reply = corpus[dialogue]['messages'][turn:turn + 2]
        if (offer['role'], user_reply['role']) != ('sys', 'usr'):
            raise ValueError('Unexpected CrossWOZ speaker order')
        targets = tuple(dict.fromkeys(act[3] for act in offer['dialog_act']
                                     if act[2] == '名称' and isinstance(act[3], str) and len(act[3]) > 1
                                     and act[3] in offer['content']))
        reply = user_reply['content']
        if len(targets) != 2 or CONTACT.search(reply) or (choice >= 0 and targets[choice] not in reply):
            raise ValueError(f'Invalid reviewed offer reference: {dialogue}/{turn}')
        selected.append(dict(key=f'{dialogue}/{turn}', group=group, targets=targets,
                             reply=reply, state=0 if choice < 0 else choice + 1))
    return selected


def observations(rows):
    return [{'reply': row['reply'], 'target': target, 'speaker': 'usr'} for row in rows for target in row['targets']]


@torch.no_grad()
def embeddings(head, features):
    values = head.project(features['clauses'])
    attention = head.relevance(values).squeeze(-1).masked_fill(~features['mask'], float('-inf')).softmax(-1)
    pooled = (attention.unsqueeze(-1) * values).sum(dim=1)
    score = head.decision(pooled)
    return torch.cat((pooled, score), dim=-1).view(-1, 2, pooled.shape[-1] + 1)


class OfferSet(nn.Module):
    def __init__(self, width, mode):
        super().__init__()
        self.option = nn.Sequential(nn.LayerNorm(width, elementwise_affine=False), nn.Linear(width, 32),
                                    nn.GELU(), nn.Linear(32, 1))
        self.interaction = (nn.Sequential(nn.LayerNorm(width * 2, elementwise_affine=False), nn.Linear(width * 2, 36),
                                          nn.GELU(), nn.Linear(36, 2)) if mode != 'independent' else None)

    def forward(self, vectors):
        a, b = self.option(vectors).squeeze(-1).unbind(1)
        zero = torch.zeros_like(a)
        if self.interaction is None:
            one_bias, both_bias = zero, zero
        else:
            left, right = vectors.unbind(1)
            symmetric = torch.cat(((left + right) / 2, (left - right).abs()), dim=-1)
            one_bias, both_bias = self.interaction(symmetric).unbind(1)
        return torch.stack((zero, a + one_bias, b + one_bias, a + b + both_bias), dim=1)


def fit(head, vectors, rows):
    gold = torch.tensor([row['state'] for row in rows])
    optimizer = torch.optim.AdamW(head.parameters(), lr=.0015, weight_decay=.02)
    rng = torch.Generator().manual_seed(SEED)
    trace = []
    for _ in range(EPOCHS):
        losses = []
        for indices in torch.randperm(len(gold), generator=rng).split(16):
            loss = F.cross_entropy(head(vectors[indices]), gold[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.item())
        trace.append(sum(losses) / len(losses))
    return trace


def summarize(rows, selected):
    categories, details = {}, []
    for row, prediction in zip(rows, selected, strict=True):
        expected = row['state']
        counts = categories.setdefault(row['group'], {}).setdefault(STATE_NAMES[expected], Counter())
        counts.update(cases=1, correct=int(prediction == expected), unexpected=(prediction & ~expected).bit_count(),
                      missed=(expected & ~prediction).bit_count())
        details.append({**row, 'prediction': prediction})
    return {'counts': categories, 'rows': details}


@torch.no_grad()
def evaluate(head, vectors, rows):
    logits = head(vectors)
    reverse = head(vectors.flip(1))[:, [0, 2, 1, 3]]
    if not torch.allclose(logits, reverse, atol=1e-5, rtol=1e-5):
        raise ValueError('Changing the input order changed set scores')
    result = summarize(rows, logits.argmax(1).tolist())
    result['logits'] = logits.tolist()
    return result


def mentions(rows):
    return torch.tensor([[target in row['reply'] for target in row['targets']] for row in rows], dtype=torch.float32)


def model_inputs(mode, vectors, occurrences):
    if mode == 'joint_mentions':
        return torch.cat((vectors, occurrences.unsqueeze(-1) * 2), dim=-1)
    return vectors


def explicit_scope(rows, scores, occurrences, threshold):
    mentioned = occurrences.bool()
    approved = mentioned & (scores >= threshold)
    selected = approved[:, 0].long() + 2 * approved[:, 1].long()
    eligible = [index for index, row in enumerate(rows)
                if row['group'] in ('real', 'real_fresh') and int(mentioned[index].sum()) == 1]
    return summarize([rows[index] for index in eligible], selected[eligible].tolist())


def run(args):
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=True)
    destination = target / ('reload.json' if args.verify else 'result.json')
    weights, cache_path = target / 'set-heads.pt', target / 'features.pt'
    if destination.exists() or (not args.verify and (weights.exists() or cache_path.exists())):
        raise FileExistsError('Use a fresh research directory')
    torch.set_num_threads(2)
    if args.verify:
        report = json.loads((target / 'result.json').read_text(encoding='utf-8'))
        if report['weights_sha256'] != sha(weights) or report['features_sha256'] != sha(cache_path):
            raise ValueError('Reloaded offer-set inputs differ from the report')
        data = torch.load(cache_path, map_location='cpu', weights_only=True)
        states = torch.load(weights, map_location='cpu', weights_only=True)
        for mode in states:
            vectors = model_inputs(mode, data['vectors'], data.get('mentions'))
            head = OfferSet(vectors.shape[-1], mode).eval()
            head.load_state_dict(states[mode])
            if evaluate(head, vectors, data['rows']) != report['evaluation'][mode]:
                raise ValueError('Reloaded offer set differs from the report')
        store_json(destination, {'verified': list(states), 'scenarios': len(data['rows'])})
        print('Verified', len(data['rows']), 'scenarios per head', flush=True)
        return

    started = perf_counter()
    train, written = samples()
    held = written + natural_scenarios(args.validation)
    modes = MODES
    if args.mention_contrast:
        more_train, more_held = partial_samples()
        train += more_train
        held += more_held + natural_scenarios(args.validation, REAL_FRESH, 'real_fresh')
        modes = (*modes, 'joint_mentions')
    if {row['reply'] for row in train}.intersection(row['reply'] for row in held):
        raise ValueError('A held reply appears in training')
    print('Scoped offers', len(train), 'training /', len(held), 'held', flush=True)
    torch.cuda.reset_peak_memory_stats()
    previous = json.loads((args.previous.parent / 'result.json').read_text(encoding='utf-8'))
    if previous['sha256'][0] != sha(args.previous):
        raise ValueError('Expected the verified clause-trained weights')
    source = torch.load(args.previous, map_location='cpu', weights_only=True)['supervised']
    chat = load_spora_for_eval(target)
    if {k: v for k, v in chat.awake_individual.items() if k != 'memory_path'} != {
            k: v for k, v in previous['runtime'].items() if k != 'memory_path'}:
        raise ValueError('The trained clause heads use a different backbone')
    chat.tokenizer.padding_side = 'right'
    extractor = FrozenTurnFeatures(chat.decoder, chat.tokenizer)
    train_features, held_features = (capture(extractor, observations(rows)) for rows in (train, held))
    clause = make_head('supervised', train_features).eval()
    clause.load_state_dict(source['weights'])
    fit_vectors, test_vectors = (embeddings(clause, features) for features in (train_features, held_features))
    threshold = source['threshold']
    last = test_vectors[..., -1]
    prior = ((last[:, 0] >= threshold).long() + 2 * (last[:, 1] >= threshold).long()).tolist()
    evaluation, states, losses = {'previous': summarize(held, prior)}, {}, {}
    fit_mentions, test_mentions = mentions(train), mentions(held)
    for mode in modes:
        torch.manual_seed(SEED)
        train_input = model_inputs(mode, fit_vectors, fit_mentions)
        test_input = model_inputs(mode, test_vectors, test_mentions)
        head = OfferSet(train_input.shape[-1], mode)
        losses[mode] = fit(head, train_input, train)
        evaluation[mode] = evaluate(head.eval(), test_input, held)
        states[mode] = head.state_dict()
        print(mode, evaluation[mode]['counts'], flush=True)
    atomic_save(states, weights)
    # Saving rows as JSON-compatible values lets verification compare the same representation.
    atomic_save({'vectors': test_vectors, 'mentions': test_mentions,
                 'rows': json.loads(json.dumps(held, ensure_ascii=False))}, cache_path)
    store_json(destination, {'fit_cases': len(train), 'held_cases': len(held), 'runtime': chat.awake_individual,
                             'evaluation': evaluation, 'loss': losses, 'updates': EPOCHS * ((len(train) + 15) // 16),
                             'explicit_scope_diagnostic': explicit_scope(held, last, test_mentions, threshold),
                             'weights_sha256': sha(weights), 'features_sha256': sha(cache_path),
                             'parent_sha256': sha(args.previous), 'seconds': perf_counter() - started,
                             'peak_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('validation', 'previous', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--mention-contrast', action='store_true')
    parser.add_argument('--verify', action='store_true')
    run(parser.parse_args())
