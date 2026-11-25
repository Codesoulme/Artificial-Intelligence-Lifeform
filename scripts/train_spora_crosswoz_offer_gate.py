"""Train a scoped offer-acceptance gate from reviewed CrossWOZ follow-ups."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter
from zipfile import ZipFile

from scripts.audit_spora_crosswoz_revisions import CONTACT, corpus_rows, validation
from scripts.audit_spora_dialogue_evidence import load_spora_for_eval, store_json
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.deepseek_local_baseline import sha, torch
from scripts.spora_crosswoz_offer_annotations import CHALLENGES, TRAIN_ACCEPT, TRAIN_NO_ACCEPT, VALIDATION_QUESTIONS
from spora.language_training import atomic_save
from torch import nn
from torch.nn import functional as F


TRAIN_ARCHIVE = 'b81f6a0845011bb4593d21dd58e2d76af4dd5d52e300ecc68cb8dee17ccc3cf8'
SEED = 20261127
EPOCHS = 48
FORMAT = 'spora-crosswoz-offer-gate-v1'
ACKNOWLEDGEMENT = re.compile(r'^(?:那|嗯[，, ]*|哦[，, ]*)?(?:好[啊呀吧]|行|可以|也行|也可以|没问题)|就(?:去|选|定)|我可以接受')


def train_archive(path):
    if sha(path) != TRAIN_ARCHIVE:
        raise ValueError('Expected the pinned training archive')
    with ZipFile(path) as archive:
        if archive.namelist() != ['train.json'] or archive.getinfo('train.json').file_size != 389_814_229:
            raise ValueError('Unexpected training archive layout')
        return json.load(archive.open('train.json'))


def redact(followup, context):
    if CONTACT.search(followup):
        raise ValueError('A reviewed follow-up includes a contact value')
    mentions = {act[3] for row in context for act in row['dialog_act'] if len(act) == 4
                and act[2] in ('名称', '地址', '电话', '出发地', '目的地', '推荐菜')
                and isinstance(act[3], str) and len(act[3]) >= 2 and act[3] in followup}
    if mentions:
        alternatives = '|'.join(re.escape(value) for value in sorted(mentions, key=lambda value: (-len(value), value)))
        followup = re.sub(alternatives, '该选项', followup)
    return re.sub(r'\d+(?:[.\-–]\d+)*', '某数', followup).strip()


def extract(corpus, dialogue, index, expected, group):
    turns = corpus[dialogue]['messages'][index:index + 3]
    if ([turn['role'] for turn in turns] != ['usr', 'sys', 'usr'] or
            not any(act[0] == 'NoOffer' for act in turns[1]['dialog_act'])):
        raise ValueError(f'Invalid original request, system offer and user reply: {dialogue}/{index}')
    return {'dialogue': dialogue, 'index': index, 'expected': expected, 'group': group,
            'reply': redact(turns[-1]['content'], turns), 'speaker': 'usr'}


def observation_sets(train_path, validation_path):
    source = train_archive(train_path)
    fit, calibration = [], []
    for label, references in ((True, TRAIN_ACCEPT), (False, TRAIN_NO_ACCEPT)):
        ranked = sorted(references, key=lambda point: sha256(f'{SEED}/{point[0]}'.encode()).hexdigest())
        for offset, (dialogue, index) in enumerate(ranked):
            target = calibration if offset < 8 else fit
            target.append(extract(source, dialogue, index, label, 'reviewed'))
    del source
    crosswoz = validation(validation_path)
    reviewed = corpus_rows(crosswoz)
    held = []
    for row in reviewed:
        if row['group'] in ('pending', 'accepted', 'information_only'):
            if row['group'] == 'pending':
                held.append({'dialogue': row['dialogue'], 'index': row['turns'][0], 'expected': False,
                             'group': 'pending', 'reply': None, 'speaker': 'sys'})
            else:
                held.append(extract(crosswoz, row['dialogue'], row['turns'][0], row['group'] == 'accepted', row['group']))
    held.extend(extract(crosswoz, dialogue, index, False, 'extra_questions') for dialogue, index in VALIDATION_QUESTIONS)
    held.extend({'dialogue': 'challenge', 'index': index, 'expected': expected, 'group': 'written_probes',
                 'reply': text, 'speaker': 'usr'} for index, (text, expected) in enumerate(CHALLENGES))
    if {row['dialogue'] for row in fit + calibration}.intersection(row['dialogue'] for row in held):
        raise ValueError('Training and validation conversations overlap')
    if {row['dialogue'] for row in fit}.intersection(row['dialogue'] for row in calibration):
        raise ValueError('Calibration conversations overlap training')
    return fit, calibration, held


def capture(extractor, rows):
    replies = [row['reply'] for row in rows if row['speaker'] == 'usr']
    prefix = '这是用户对备选方案的回复。用户：'
    vectors = []
    for start in range(0, len(replies), 12):
        batch = replies[start:start + 12]
        encoded = extractor([prefix + reply for reply in batch]).float()
        for reply, text in zip(batch, encoded, strict=True):
            span = text[len(prefix):len(prefix) + len(reply)]
            vectors.append(torch.cat((span.mean(0), span[-1])))
    return torch.stack(vectors)


def classifier(width):
    return nn.Sequential(nn.LayerNorm(width, elementwise_affine=False), nn.Linear(width, 1))


def learn(head, values, rows):
    gold = torch.tensor([row['expected'] for row in rows], dtype=torch.float32)
    balance = torch.where(gold > 0, .5 / gold.mean(), .5 / (1. - gold.mean()))
    optimizer = torch.optim.AdamW(head.parameters(), lr=.0015, weight_decay=.03)
    rng = torch.Generator().manual_seed(SEED)
    trace = []
    for _ in range(EPOCHS):
        losses = []
        for batch in torch.randperm(len(gold), generator=rng).split(16):
            logits = head(values[batch]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, gold[batch], weight=balance[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.item())
        trace.append(sum(losses) / len(losses))
    return trace


@torch.no_grad()
def calibrate(head, values, rows):
    scores = head(values).squeeze(-1).tolist()
    points = sorted(set(scores))
    cuts = [points[0] - 1., *((left + right) / 2 for left, right in zip(points, points[1:])), points[-1] + 1.]

    def agreement(cut):
        pairs = [(value >= cut, row['expected']) for value, row in zip(scores, rows, strict=True)]
        return sum(predicted == gold for predicted, gold in pairs), -sum(predicted and not gold for predicted, gold in pairs)

    best = max(cuts, key=agreement)
    return best, {'correct': agreement(best)[0], 'false_accept': -agreement(best)[1], 'cases': len(rows)}


@torch.no_grad()
def assess(head, threshold, values, rows, known):
    scores = iter(head(values).squeeze(-1).tolist())
    details, metrics = [], {}
    for row in rows:
        # A system turn can create a pending proposal. Only a user reply can activate it.
        reply = row['reply']
        score = next(scores) if row['speaker'] == 'usr' else None
        learned = score is not None and score >= threshold
        regex = reply is not None and bool(ACKNOWLEDGEMENT.search(reply))
        for method, accept in (('frozen_features', learned), ('text_pattern', regex)):
            counters = metrics.setdefault(method, {}).setdefault(row['group'], Counter())
            gold = row['expected']
            counters.update(cases=1, correct=int(accept == gold), false_accept=int(accept and not gold),
                            missed_accept=int(gold and not accept))
        details.append({**row, 'seen_text': reply in known, 'score': score, 'learned_accept': learned,
                        'pattern_accept': regex, 'active_candidate': 1 if learned else 0})
    pairs = {(row['dialogue'], row['index']): row['learned_accept'] for row in details if row['group'] == 'pending'}
    paired = sum(row['learned_accept'] and not pairs[(row['dialogue'], row['index'])]
                 for row in details if row['group'] == 'accepted')
    return {'metrics': metrics, 'complete_pairs': paired, 'rows': details}


def verify(target):
    report = json.loads((target / 'result.json').read_text(encoding='utf-8'))
    weights, features = target / 'gate.pt', target / 'evaluation-features.pt'
    if sha(weights) != report['weights_sha256'] or sha(features) != report['features_sha256']:
        raise ValueError('Saved evaluation inputs differ from the report')
    state = torch.load(weights, map_location='cpu', weights_only=True)
    saved = torch.load(features, map_location='cpu', weights_only=True)
    if state['format'] != FORMAT or state['data'] != report['dataset_sha256']:
        raise ValueError('The saved gate does not match the audited corpus')
    head = classifier(saved['values'].shape[-1]).eval()
    head.load_state_dict(state['weights'])
    result = assess(head, state['threshold'], saved['values'], saved['rows'], set(saved['seen_texts']))
    if result != report['evaluation']:
        raise ValueError('Reloaded gate did not reproduce the scored scenarios')
    store_json(target / 'reload.json', {'matches': True, 'cases': len(result['rows']), 'weights_sha256': sha(weights)})
    print('Reloaded', len(result['rows']), 'scenarios', flush=True)


def main(args):
    target = args.output.resolve()
    torch.set_num_threads(2)
    if args.verify:
        verify(target)
        return
    target.mkdir(parents=True, exist_ok=True)
    weights, features = target / 'gate.pt', target / 'evaluation-features.pt'
    if any(path.exists() for path in (weights, features, target / 'result.json')):
        raise FileExistsError('Use a fresh experiment directory')
    torch.manual_seed(SEED)
    started = perf_counter()
    fit, calibration, held = observation_sets(args.train, args.validation)
    digest = sha256(json.dumps((fit, calibration, held), ensure_ascii=False).encode('utf-8')).hexdigest()
    print('Reviewed splits:', len(fit), len(calibration), len(held), flush=True)
    torch.cuda.reset_peak_memory_stats()
    chat = load_spora_for_eval(target)
    extractor = FrozenTurnFeatures(chat.decoder, chat.tokenizer)
    texts = set(row['reply'] for row in fit + calibration)
    x, cal, test = (capture(extractor, rows) for rows in (fit, calibration, held))
    head = classifier(x.shape[-1])
    loss = learn(head, x, fit)
    head.eval()
    threshold, calibration_result = calibrate(head, cal, calibration)
    result = assess(head, threshold, test, held, texts)
    atomic_save({'format': FORMAT, 'data': digest, 'runtime': chat.awake_individual,
                 'threshold': threshold, 'weights': head.state_dict()}, weights)
    atomic_save({'values': test, 'rows': held, 'seen_texts': sorted(texts)}, features)
    store_json(target / 'result.json', {'dataset_sha256': digest, 'weights_sha256': sha(weights),
                                        'features_sha256': sha(features), 'runtime': chat.awake_individual,
                                        'evaluation': result, 'calibration': calibration_result, 'threshold': threshold,
                                        'fit_count': len(fit), 'fit_positive': sum(row['expected'] for row in fit),
                                        'loss': loss, 'updates': EPOCHS * ((len(fit) + 15) // 16),
                                        'peak_bytes': torch.cuda.max_memory_allocated(), 'seconds': perf_counter() - started})
    print(result['metrics'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', type=Path)
    parser.add_argument('validation', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
