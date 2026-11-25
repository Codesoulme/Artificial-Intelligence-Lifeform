"""Learn option reference separately from the decision to approve an option."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_spora_crosswoz_revisions import CONTACT, validation
from scripts.audit_spora_dialogue_evidence import load_spora_for_eval, store_json
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.deepseek_local_baseline import sha, torch
from scripts.spora_offer_reference_data import REAL, samples
from scripts.train_spora_offer_clauses import capture, make_head
from spora.language_training import atomic_save
from torch import nn
from torch.nn import functional as F


SEED = 20261130
EPOCHS = 27


def natural_rows(path):
    corpus = validation(path)
    rows = []
    for dialogue, turn, target in REAL:
        offer, user, answer, reply = corpus[dialogue]['messages'][turn:turn + 4]
        if [message['role'] for message in (offer, user, answer, reply)] != ['sys', 'usr', 'sys', 'usr']:
            raise ValueError('Unexpected CrossWOZ reference order')
        names = tuple(dict.fromkeys(act[3] for act in offer['dialog_act'] if act[2] == '名称'
                                    and isinstance(act[3], str) and len(act[3]) > 1 and act[3] in offer['content']))
        if (len(names) != 2 or any(name in reply['content'] for name in names)
                or any(CONTACT.search(item['content']) for item in (offer, user, answer, reply))):
            raise ValueError(f'Invalid real reference: {dialogue}/{turn}')
        rows.append(dict(key=f'{dialogue}/{turn}', family='real_new_topic' if target == 2 else 'real_continuation',
                         candidates=[(name, None) for name in names],
                         history=[(item['role'], item['content']) for item in (user, answer)],
                         reply=reply['content'], target=target, approval=False))
    return rows


def prepare(row, with_history):
    candidates = row['candidates']
    display = [f'{name}（简称{alias}）' if alias else name for name, alias in candidates]
    earlier = row['history'] if with_history else ()
    prior_user = next((message for actor, message in reversed(earlier) if actor == 'usr'), '')
    last_system = next((message for actor, message in reversed(earlier) if actor == 'sys'), '')
    evidence = ''.join(('用户：' if role == 'usr' else '助手：') + text + '\n' for role, text in earlier)
    prefix = f'备选顺序为：第一项{display[0]}，第二项{display[1]}。\n{evidence}'
    for name, alias in candidates:
        yield (prefix + f'要判断是否指向{name}。用户：', row['reply'],
               (name in row['reply'], bool(alias and alias in row['reply']), name in prior_user, name in last_system))


def feature_rows(extractor, rows, with_history=True):
    records = [record for row in rows for record in prepare(row, with_history)]
    values, hints = [], []
    for start in range(0, len(records), 12):
        group = records[start:start + 12]
        activation = extractor([prefix + reply for prefix, reply, _ in group]).float()
        for feature, (prefix, reply, flags) in zip(activation, group, strict=True):
            segment = feature[len(prefix):len(prefix) + len(reply)]
            values.append(torch.cat((segment.mean(0), segment[-1])))
            hints.append(flags)
    return torch.stack(values).view(len(rows), 2, -1), torch.tensor(hints, dtype=torch.float32).view(len(rows), 2, 4)


class ReferenceHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(width, elementwise_affine=False), nn.Linear(width, 48), nn.GELU())
        self.candidate = nn.Sequential(nn.Linear(52, 32), nn.GELU(), nn.Linear(32, 1))
        self.unbound = nn.Sequential(nn.Linear(52, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, values, flags):
        embedded = torch.cat((self.project(values), flags), dim=-1)
        return torch.cat((self.candidate(embedded).squeeze(-1), self.unbound(embedded.mean(dim=1))), dim=1)


def train(head, features, rows):
    labels = torch.tensor([row['target'] for row in rows])
    counts = labels.bincount().float()
    weights = labels.numel() / (counts * counts.numel())
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.02)
    generator = torch.Generator().manual_seed(SEED)
    trace = []
    for _ in range(EPOCHS):
        total = 0.
        batches = torch.randperm(len(rows), generator=generator).split(24)
        for batch in batches:
            loss = F.cross_entropy(head(features[0][batch], features[1][batch]), labels[batch], weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += loss.item()
        trace.append(total / len(batches))
    return trace


def lexical(flags, follow_history):
    direct = flags[:, 0].bool()
    options = direct
    if follow_history and not direct.any():
        named = flags[:, 1].bool()
        options = named if named.any() else flags[:, 2].bool()
    return int(options.long().argmax()) if options.sum() == 1 else 2


@torch.no_grad()
def score(head, features):
    values, flags = features
    logits = head(values, flags)
    if not torch.allclose(logits, head(values.flip(1), flags.flip(1))[:, [1, 0, 2]], atol=1e-5, rtol=1e-5):
        raise ValueError('Changing candidate processing order changed reference scores')
    return logits.argmax(dim=1).tolist()


def summarize(rows, features, no_history, approvals, head):
    aliases = [lexical(item, True) for item in features[1]]
    literals = [lexical(item, False) for item in features[1]]
    predictions = score(head, features)
    clipped = score(head, no_history)
    results, counts = [], {}
    for row, predicted, without, alias, literal, approval in zip(rows, predictions, clipped, aliases, literals,
                                                                  approvals, strict=True):
        target = row['target']
        expected_accept = target if row['approval'] else 2
        supplied = target if approval else 2
        actual = predicted if approval else 2
        successes = {'learned': predicted == target, 'without_history': without == target,
                     'name_only': literal == target, 'name_prior_focus': alias == target,
                     'intent': approval == row['approval'], 'with_given_scope': supplied == expected_accept,
                     'combined': actual == expected_accept, 'false_approval': actual != 2 and actual != expected_accept}
        counts.setdefault(row['family'], Counter()).update(cases=1, **{k: int(v) for k, v in successes.items()})
        results.append(dict(key=row['key'], family=row['family'], target=target, predicted=predicted,
                            without_history=without, anchor=alias, approval=approval, expected_accept=expected_accept,
                            selected=actual, successes=successes))
    return {'counts': counts, 'rows': results}


def run(args):
    target_path = args.output.resolve()
    target_path.mkdir(parents=True, exist_ok=True)
    destination = target_path / ('reload.json' if args.verify else 'result.json')
    weights, cached = target_path / 'reference.pt', target_path / 'features.pt'
    torch.set_num_threads(2)
    if destination.exists() or (not args.verify and (weights.exists() or cached.exists())):
        raise FileExistsError('Use a fresh output directory')
    if args.verify:
        original = json.loads((target_path / 'result.json').read_text(encoding='utf-8'))
        if original['features_sha256'] != sha(cached) or original['weights_sha256'] != sha(weights):
            raise ValueError('Saved reference inputs differ')
        data = torch.load(cached, map_location='cpu', weights_only=True)
        head = ReferenceHead(data['features'][0].shape[-1]).eval()
        head.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        if summarize(**data, head=head) != original['evaluation']:
            raise ValueError('Reload changed reference predictions')
        store_json(destination, dict(verified=len(data['rows'])))
        print('Verified', len(data['rows']), 'reference and approval outcomes', flush=True)
        return
    torch.manual_seed(SEED)
    begin = perf_counter()
    training, held = samples()
    holding = held + natural_rows(args.validation)
    if {r['reply'] for r in training}.intersection(r['reply'] for r in holding):
        raise ValueError('A held reply appears in training')
    print('Reference training / held:', len(training), len(holding), flush=True)
    torch.cuda.reset_peak_memory_stats()
    statement = json.loads((args.prior.parent / 'result.json').read_text(encoding='utf-8'))
    if sha(args.prior) != statement['sha256'][0]:
        raise ValueError('Expected the verified clause weights')
    previous = torch.load(args.prior, map_location='cpu', weights_only=True)['supervised']
    chat = load_spora_for_eval(target_path)
    if {k: v for k, v in chat.awake_individual.items() if k != 'memory_path'} != {
            k: v for k, v in statement['runtime'].items() if k != 'memory_path'}:
        raise ValueError('Reference and approval use different frozen states')
    chat.tokenizer.padding_side = 'right'
    extractor = FrozenTurnFeatures(chat.decoder, chat.tokenizer)
    fit = feature_rows(extractor, training)
    features = feature_rows(extractor, holding)
    no_history = feature_rows(extractor, holding, with_history=False)
    observation = capture(extractor, [dict(reply=r['reply'], speaker='usr') for r in holding])
    prior = make_head('supervised', observation).eval()
    prior.load_state_dict(previous['weights'])
    with torch.no_grad():
        logits, _ = prior(observation['clauses'], observation['mask'])
    approvals = (logits >= previous['threshold']).tolist()
    head = ReferenceHead(fit[0].shape[-1])
    losses = train(head, fit, training)
    head.eval()
    data = dict(rows=holding, features=features, no_history=no_history, approvals=approvals)
    result = summarize(**data, head=head)
    atomic_save(head.state_dict(), weights)
    atomic_save(data, cached)
    store_json(destination, dict(evaluation=result, loss=losses, training=len(training), runtime=chat.awake_individual,
                                 features_sha256=sha(cached), weights_sha256=sha(weights), parent_sha256=sha(args.prior),
                                 seconds=perf_counter() - begin, peak_bytes=torch.cuda.max_memory_allocated()))
    print(result['counts'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('validation', 'prior', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--verify', action='store_true')
    run(parser.parse_args())
