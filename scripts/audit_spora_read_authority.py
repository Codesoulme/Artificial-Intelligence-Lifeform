"""Evaluate whether a learned read decision can protect a frozen field selector."""

import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_fast_weight_lexical_write import base
from scripts.audit_spora_field_evidence import VERSION, field_head
from scripts.audit_spora_fresh_dialogue import digest
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_composition_address_cases import HELD, train_cases
from scripts.spora_field_evidence_cases import FRESH
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save


SEED = 56111
TOPICS = ('那款饮料', '那件要带的东西', '碰头的地方', '要拿的那本书')
PAIRS = (
    ('谈的是{topic}，从已存的记录取这个名字。', '谈的是{topic}，想一个新选项给我。'),
    ('关于{topic}，按原来登记的写。', '关于{topic}，先别看原来登记的，另想一个。'),
    ('帮我把{topic}写进消息，照档案里的安排。', '帮我把{topic}写进消息，我晚一点才会选。'),
    ('我忘了{topic}，请用已有记录帮我核实。', '我提到{topic}只是举例，解释这个说法。'),
    ('请告诉我{topic}，沿用上次已经确认的。', '请告诉我怎么挑{topic}，别沿用上次确认的。'),
    ('请从以前记下的安排填入{topic}，只给成品。', '请改写“从以前记下的安排填入{topic}”这句话。'),
)


def examples():
    rows = [(row.question, bool(row.fields)) for row in train_cases()]
    rows.extend((form.format(topic=topic), label) for topic in TOPICS
                for positive, negative in PAIRS for form, label in ((positive, True), (negative, False)))
    if len({prompt for prompt, _ in rows}) != len(rows):
        raise ValueError('Duplicate training prompt')
    return rows


def intent_features(extract, prompts):
    formatted = [prompt + '\n完成用户实际要求，需要调用此前已经登记的信息吗？' for prompt in prompts]
    vectors = []
    for start in range(0, len(formatted), 24):
        batch = formatted[start:start + 24]
        features = extract(batch)
        vectors.extend(features[i, len(text) - 1].clone() for i, text in enumerate(batch))
    return torch.stack(vectors).float()


def fit(head, training, features):
    labels = torch.tensor([label for _, label in training], dtype=torch.float32)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    random = torch.Generator().manual_seed(SEED)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(12):
        for batch in torch.randperm(len(training), generator=random).split(24):
            loss = loss_fn(head(features[batch]).squeeze(-1), labels[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


@torch.inference_mode()
def evaluate(head, extract, rows, archived):
    probabilities = head(intent_features(extract, [row.question for row in rows])).sigmoid().flatten().tolist()
    summary = {mode: Counter() for mode in ('previous', 'guarded')}
    decisions = []
    for row, old, probability in zip(rows, archived, probabilities, strict=True):
        if row.question != old['request']['question'] or list(row.fields) != old['request']['fields']:
            raise ValueError('Archived decisions do not match the evaluation set')
        route = old['decisions'][-1]['route']
        expected = old['expected']
        explicit = bool(VERSION.search(row.question))
        allowed = explicit or probability >= .5
        for name, chosen in (('previous', route), ('guarded', route if allowed else [])):
            summary[name].update(
                exact_positive=bool(expected) and set(chosen) == set(expected),
                false_read=not expected and bool(chosen),
                wrong_or_partial=bool(expected) and bool(chosen) and set(chosen) != set(expected),
            )
        decisions.append({'question': row.question, 'probability': probability, 'explicit': explicit,
                          'allowed': allowed, 'previous_route': route, 'expected': expected})
    return {'summary': summary, 'decisions': decisions}


def main(args):
    torch.set_num_threads(2)
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / ('replay.json' if args.verify else 'result.json')
    if destination.exists():
        raise FileExistsError(destination)
    baseline_sha = digest(args.baseline)
    baseline = json.loads(args.baseline.read_text(encoding='utf-8'))
    weights = base.MODEL / 'model.safetensors'
    if digest(weights) != base.WEIGHT_SHA or baseline['sources'].get(str(weights)) != base.WEIGHT_SHA:
        raise ValueError('Expected the same frozen model as the archived field decisions')
    training = examples()
    if {prompt for prompt, _ in training} & {row.question for row in (*HELD, *FRESH)}:
        raise ValueError('Training and evaluation text overlap')
    model = load_full_streamed()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer)
    torch.manual_seed(SEED)
    head = field_head(model.config.hidden_size)
    checkpoint = args.output / 'read_intent.pt'
    if args.verify:
        head.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    else:
        fit(head, training, intent_features(extract, [prompt for prompt, _ in training]))
        atomic_save(head.state_dict(), checkpoint)
    head.eval()
    result = {'weights_sha256': digest(checkpoint), 'baseline_sha256': baseline_sha,
              'training_count': len(training), 'splits': {}}
    for name, rows in (('prior', HELD), ('fresh', FRESH)):
        result['splits'][name] = evaluate(head, extract, rows, baseline['rows'][name])
        print(name, result['splits'][name]['summary'], flush=True)
    if digest(args.baseline) != baseline_sha:
        raise ValueError('The archived field decisions changed during evaluation')
    if args.verify and result != json.loads((args.output / 'result.json').read_text(encoding='utf-8')):
        raise ValueError('Independent replay did not reproduce the read decisions')
    store_json(destination, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
