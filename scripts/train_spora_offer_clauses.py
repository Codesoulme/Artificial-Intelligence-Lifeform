"""Train target-aware confirmation from contrastive clauses and evidence spans."""

import argparse
from collections import Counter
from pathlib import Path
import json
import re
from time import perf_counter

from scripts.audit_spora_dialogue_evidence import load_spora_for_eval, store_json
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.deepseek_local_baseline import sha, torch
from scripts.spora_offer_clause_data import samples
from scripts.train_spora_crosswoz_offer_gate import (FORMAT as PRIOR_FORMAT, assess, calibrate_scores, classifier,
                                                    observation_sets)
from spora.language_training import atomic_save
from torch import nn
from torch.nn import functional as F


SEED = 20261128
EPOCHS = 24
MODES = ('whole', 'clauses', 'supervised')
DECISION = re.compile(r'[^。！？；!?;]+[。！？；!?;]*')


def split_reply(row):
    reply = row['reply']
    target = row.get('target')
    anchor = row.get('anchor')
    prefix = (f'当前要判断是否采用的方案为{target}。' if target else '') + '这是用户对备选方案的回复。用户：'
    parts = [m.span() for m in DECISION.finditer(reply) if m.group().strip()]
    if not parts:
        raise ValueError('An observed user reply has no clause')
    evidence = -1
    if anchor:
        start = reply.index(anchor)
        end = start + len(anchor)
        evidence, = [i for i, (left, right) in enumerate(parts) if left <= start < end <= right]
    return prefix + reply, len(prefix), parts, evidence


def capture(extractor, rows):
    utterances = [row for row in rows if row['speaker'] == 'usr']
    layouts = [split_reply(row) for row in utterances]
    count = max(len(row[2]) for row in layouts)
    whole, clauses, valid, labels = [], [], [], []
    for start in range(0, len(layouts), 12):
        chunk = layouts[start:start + 12]
        features = extractor([x[0] for x in chunk]).float()
        for row, (text, offset, parts, evidence), feature in zip(utterances[start:start + 12], chunk, features, strict=True):
            reply = feature[offset:len(text)]
            whole.append(torch.cat((reply.mean(0), reply[-1])))
            target = row.get('target')
            vectors = []
            for left, right in parts:
                values = reply[left:right]
                relative = row['reply'].find(target, left, right) if target else -1
                mention = (reply[relative:relative + len(target)].mean(0) if relative >= 0 else
                           torch.zeros_like(values[0]))
                vectors.append(torch.cat((values.mean(0), values[-1], mention)))
            clauses.append(F.pad(torch.stack(vectors), (0, 0, 0, count - len(vectors))))
            valid.append(torch.arange(count) < len(vectors))
            labels.append(evidence)
    return dict(whole=torch.stack(whole), clauses=torch.stack(clauses), mask=torch.stack(valid), anchor=torch.tensor(labels))


class ClauseGate(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(width, elementwise_affine=False), nn.Linear(width, 48), nn.GELU())
        self.relevance = nn.Linear(48, 1)
        self.decision = nn.Linear(48, 1)

    def forward(self, clauses, mask):
        vectors = self.project(clauses)
        relevance = self.relevance(vectors).squeeze(-1).masked_fill(~mask, float('-inf'))
        decisions = self.decision(vectors).squeeze(-1)
        return (relevance.softmax(-1) * decisions).sum(-1), relevance


def make_head(mode, features):
    return classifier(features['whole'].shape[-1]) if mode == 'whole' else ClauseGate(features['clauses'].shape[-1])


def forward(head, features, mode, subset=slice(None)):
    if mode == 'whole':
        return head(features['whole'][subset]).squeeze(-1), None
    return head(features['clauses'][subset], features['mask'][subset])


def train(head, features, mode, rows):
    target = torch.tensor([row['expected'] for row in rows], dtype=torch.float32)
    balance = torch.where(target == 1, .5 / target.mean(), .5 / (1 - target.mean()))
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.03)
    rng = torch.Generator().manual_seed(SEED)
    loss_trace = []
    for _ in range(EPOCHS):
        losses = []
        for batch in torch.randperm(len(rows), generator=rng).split(24):
            logits, relevance = forward(head, features, mode, batch)
            loss = F.binary_cross_entropy_with_logits(logits, target[batch], weight=balance[batch])
            if mode == 'supervised':
                anchor = features['anchor'][batch]
                annotated = anchor >= 0
                if annotated.any():
                    loss = loss + .65 * F.cross_entropy(relevance[annotated], anchor[annotated])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach().item())
        loss_trace.append(sum(losses) / len(losses))
    return loss_trace


@torch.no_grad()
def evaluate(head, threshold, features, mode, rows, seen):
    logits, attention = forward(head, features, mode)
    result = assess(nn.Identity(), threshold, logits.unsqueeze(-1), rows, seen)
    pairs, role_counts = {}, {}
    pointer = iter(attention.argmax(-1).tolist()) if attention is not None else None
    targets = iter(features['anchor'].tolist())
    for row in result['rows']:
        if row['speaker'] != 'usr':
            continue
        anchor = next(targets)
        chosen = next(pointer) if pointer is not None else None
        if chosen is not None and anchor >= 0:
            counts = role_counts.setdefault(row['group'], Counter())
            counts.update(cases=1, exact=int(chosen == anchor))
        if row.get('pair'):
            pairs.setdefault((row['group'], row['pair']), []).append(row['learned_accept'] == row['expected'])
    paired = {}
    for (group, _), matches in pairs.items():
        if len(matches) != 2:
            raise ValueError('Every matched scenario needs both sides')
        paired.setdefault(group, Counter()).update(cases=1, correct=int(all(matches)))
    return {**result, 'evidence': role_counts, 'matched_pairs': paired}


def verified_results(target, previous_path, payload, candidates, saved_features):
    report = json.loads((target / 'result.json').read_text(encoding='utf-8'))
    actual = (sha(candidates), sha(saved_features), sha(previous_path))
    if actual != tuple(report['sha256']):
        raise ValueError('Saved evaluation inputs differ from the report')
    states = torch.load(candidates, map_location='cpu', weights_only=True)
    seen = set(payload['known'])
    for mode in MODES:
        teacher = make_head(mode, payload['features']).eval()
        teacher.load_state_dict(states[mode]['weights'])
        result = evaluate(teacher, states[mode]['threshold'], payload['features'], mode, payload['rows'], seen)
        if result != report['evaluation'][mode]:
            raise ValueError('Reloaded candidate produced a different decision')
    store_json(target / 'reload.json', {'cases_per_candidate': len(payload['rows']), 'verified': list(MODES)})
    print('Verified', len(payload['rows']), 'scenarios per candidate', flush=True)


def main(args):
    target = args.output.resolve()
    candidates, features = target / 'heads.pt', target / 'features.pt'
    torch.set_num_threads(2)
    if args.verify:
        payload = torch.load(features, map_location='cpu', weights_only=True)
        verified_results(target, args.previous, payload, candidates, features)
        return
    target.mkdir(parents=True, exist_ok=True)
    if any(path.exists() for path in (candidates, features, target / 'result.json')):
        raise FileExistsError('Use a fresh output directory')
    begin = perf_counter()
    train_rows, calibration, held = observation_sets(args.train, args.validation)
    original_held = list(held)
    additional, additional_cal, additional_held = samples()
    train_rows += additional
    calibration += additional_cal
    held += additional_held
    known = {row['reply'] for row in train_rows + calibration}
    if any(row['reply'] in known for row in additional_held):
        raise ValueError('A new written evaluation reply overlaps training or calibration')
    print('Cases:', len(train_rows), len(calibration), len(held), flush=True)
    torch.cuda.reset_peak_memory_stats()
    chat = load_spora_for_eval(target)
    before = torch.load(args.previous, map_location='cpu', weights_only=True)
    signature = lambda info: {key: value for key, value in info.items() if key != 'memory_path'}
    if before['format'] != PRIOR_FORMAT or signature(before['runtime']) != signature(chat.awake_individual):
        raise ValueError('The original comparison gate uses a different backbone')
    reference_dir = args.previous.parent
    reference_report = json.loads((reference_dir / 'result.json').read_text(encoding='utf-8'))
    reference_features = reference_dir / 'evaluation-features.pt'
    if sha(reference_features) != reference_report['features_sha256'] or sha(args.previous) != reference_report['weights_sha256']:
        raise ValueError('The original comparison inputs do not match their saved report')
    reference = torch.load(reference_features, map_location='cpu', weights_only=True)
    if reference['rows'] != original_held:
        raise ValueError('The comparison set differs from the prior experiment')
    extractor = FrozenTurnFeatures(chat.decoder, chat.tokenizer)
    prior_fresh = capture(extractor, additional_held)['whole']
    probe_replies = [row['reply'] for row in original_held if row['reply']]
    short, long = min(probe_replies, key=len), max(probe_replies, key=len)
    prompt = '这是用户对备选方案的回复。用户：'

    def padding_delta():
        alone = extractor([prompt + short])[0]
        mixed = extractor([prompt + short, prompt + long])[0, :len(alone)]
        return (alone.float() - mixed.float()).abs().max().item()

    previous_delta = padding_delta()
    chat.tokenizer.padding_side = 'right'
    corrected_delta = padding_delta()
    fit, cal, test = (capture(extractor, rows) for rows in (train_rows, calibration, held))
    prior = make_head('whole', test).eval()
    prior.load_state_dict(before['weights'])
    prior_test = {**test, 'whole': torch.cat((reference['values'], prior_fresh))}
    evaluation = {'previous': evaluate(prior, before['threshold'], prior_test, 'whole', held, known)}
    replayed = evaluation['previous']['rows'][:len(original_held)]
    if any(left['learned_accept'] != right['learned_accept']
           for left, right in zip(replayed, reference_report['evaluation']['rows'], strict=True)):
        raise ValueError('The prior baseline did not reproduce its recorded decisions')
    states, calibration_data, losses = {}, {}, {}
    for mode in MODES:
        torch.manual_seed(SEED)
        head = make_head(mode, fit)
        losses[mode] = train(head, fit, mode, train_rows)
        head.eval()
        with torch.no_grad():
            cal_scores, _ = forward(head, cal, mode)
        threshold, calibration_data[mode] = calibrate_scores(cal_scores.tolist(), calibration)
        evaluation[mode] = evaluate(head, threshold, test, mode, held, known)
        states[mode] = {'threshold': threshold, 'weights': head.state_dict()}
        print(mode, evaluation[mode]['metrics']['frozen_features'], evaluation[mode]['matched_pairs'], flush=True)
    atomic_save(states, candidates)
    atomic_save({'features': test, 'rows': held, 'known': sorted(known)}, features)
    store_json(target / 'result.json', {'evaluation': evaluation, 'calibration': calibration_data, 'loss': losses,
                                        'fit_examples': len(train_rows), 'runtime': chat.awake_individual,
                                        'updates': EPOCHS * ((len(train_rows) + 23) // 24),
                                        'sha256': [sha(candidates), sha(features), sha(args.previous)],
                                        'reference_features_sha256': sha(reference_features),
                                        'padding_probe': {'left': previous_delta, 'right': corrected_delta},
                                        'seconds': perf_counter() - begin, 'peak_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('train', 'validation', 'previous', 'output'):
        parser.add_argument(argument, type=Path)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
