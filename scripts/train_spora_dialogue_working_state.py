"""Learn bounded writes from individual user turns and test neural working-state reads."""

import argparse
import json
from pathlib import Path
import random
from string import Formatter
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from torch.nn import functional as F

from scripts.audit_spora_native_query_key import write
from scripts.deepseek_local_baseline import sha
from spora.language_training import atomic_save
from spora.research_dialogue_working_state import FIELDS, TurnValueWriter, characters


DATA = ROOT / 'data/dialogue-working-state-v1.json'
SEED = 24117
WRITER_WIDTH = 64
MARGIN = 2.


def annotated(template, values):
    parts, offsets, position = [], {}, 0
    for literal, name, _, _ in Formatter().parse(template):
        parts.append(literal)
        position += len(literal)
        if name is None:
            continue
        value = values[name]
        parts.append(value)
        if name in FIELDS:
            offsets[name] = (position + 1, position + len(value))
        position += len(value)
    return ''.join(parts), [offsets.get(field, (0, 0)) for field in FIELDS]


def dialogue_data(payload, *, held, count):
    split = 'held' if held else 'train'
    rng = random.Random(SEED + int(held))
    values = payload['values'][split]
    rows = []
    for kind, patterns in payload['kinds'].items():
        formats = patterns[f'{split}_first']
        updates = patterns[f'{split}_update']
        for index in range(count):
            first = rng.choice(formats)
            fields = [name for _, name, _, _ in Formatter().parse(first) if name is not None]
            initial = {field: rng.choice(values[field]) for field in fields}
            changes = updates[index % len(updates)]
            edit = {field: rng.choice([value for value in values[field] if value != initial[field]])
                    for field in changes[:-1]}
            signals = {f'old_{field}': value for field, value in initial.items()}
            changed = {**initial, **edit}
            questions = patterns[f'{split}_question']
            rows.append({'kind': kind, 'first': annotated(first, initial),
                         'update': annotated(changes[-1], {**signals, **edit}),
                         'question': rng.choice(questions), 'answer': patterns['answer'].format(**changed),
                         'initial_answer': patterns['answer'].format(**initial),
                         'initial': initial, 'changes': edit, 'current': changed})
    return rows


def fit_writer(model, rows, no_write, epochs, device, *, features=None):
    empty = [(0, 0)] * len(FIELDS)
    examples = [example for row in rows for example in (row['first'], row['update'])]
    examples.extend((row['question'], empty) for row in rows[::2])
    examples.extend((text, empty) for text in no_write for _ in range(12))
    texts, boundaries = zip(*examples, strict=True)
    codes = characters(texts)
    targets = torch.tensor(boundaries)
    encoded = None if features is None else features(texts)
    if encoded is not None and encoded.shape[:2] != codes.shape:
        raise ValueError('Semantic features do not align with the training characters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    trace = []
    for epoch in range(epochs):
        losses = []
        for batch in torch.randperm(len(examples), device=device).cpu().split(64):
            context = None if encoded is None else encoded[batch].to(device)
            logits = model(codes[batch].to(device), context).flatten(end_dim=2)
            truth = targets[batch].to(device).flatten()
            loss = F.cross_entropy(logits, truth, reduction='none')
            loss = (loss * torch.where(truth > 0, 2.5, 1.)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        item = {'epoch': epoch + 1, 'loss': sum(losses) / len(losses)}
        trace.append(item)
        print('Working-state writer', item, flush=True)
    return trace


def writer_results(writer, rows, locate=None):
    if locate is None:
        locate = writer.write
    results = []
    for case in rows:
        initial = locate(case['first'][0], MARGIN)
        changes = locate(case['update'][0], MARGIN)
        query = locate(case['question'], MARGIN)
        current = initial | changes | query
        kept = case['initial'].keys() - case['changes'].keys()
        results.append({
            'kind': case['kind'], 'expected': case['current'], 'initial_write': initial,
            'update_write': changes, 'query_write': query, 'current': current,
            'initial_exact': initial == case['initial'], 'update_exact': changes == case['changes'],
            'retained': sum(current.get(field) == case['current'][field] for field in kept),
            'retained_total': len(kept), 'complete': current == case['current'],
        })
    return results


def train_writer(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source = sha(DATA)
    payload = json.loads(DATA.read_text(encoding='utf-8'))
    training = dialogue_data(payload, held=False, count=320)
    held = dialogue_data(payload, held=True, count=8)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TurnValueWriter(WRITER_WIDTH).to(device)
    trace = fit_writer(model, training, payload['no_write'], args.epochs, device)
    model.eval()
    cases = writer_results(model, held)
    if sha(DATA) != source:
        raise ValueError('Working-state training source changed')
    checkpoint = args.output / 'writer.pt'
    atomic_save({'schema': 'spora-working-turn-writer-v1', 'fields': FIELDS, 'width': WRITER_WIDTH,
                 'source_sha256': source, 'model': {key: value.detach().cpu() for key, value in model.state_dict().items()}},
                checkpoint)
    write(args.output / 'writer.json', {'cases': cases, 'trace': trace, 'train_cases': len(training),
                                       'parameters': sum(p.numel() for p in model.parameters()),
                                       'seconds': time.perf_counter() - started, 'writer_sha256': sha(checkpoint),
                                       'source_sha256': source})
    print('Complete held states', sum(row['complete'] for row in cases), '/', len(cases), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    options = parser.parse_args()
    if options.epochs < 1:
        parser.error('--epochs must be positive')
    train_writer(options)
