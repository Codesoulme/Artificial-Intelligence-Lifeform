"""Audit a trigger-gated BIO value copier on independent dialogue holdouts."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from audit_spora_dialogue_working_state import FEATURE_DEPTH, FrozenTurnFeatures
from audit_spora_writer_transfer import write_metrics
from audit_spora_writer_compositional import training_rows
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_dialogue_working_state import DATA as WRITER_DATA, WRITER_WIDTH
from spora.integrated_durable import load_integrated_durable
from spora.language_training import atomic_save, frozen_digest
from spora.research_bio_value_writer import TriggerBioValueWriter
from spora.research_dialogue_working_state import FIELDS, characters


HOLDOUT = ROOT / 'data/dialogue-read-plan-holdout.json'


def labels_for(boundaries, width):
    labels = torch.zeros(len(FIELDS), width, dtype=torch.long)
    for field_index, (start, end) in enumerate(boundaries):
        if start == 0:
            continue
        left, right = start - 1, end - 1
        labels[field_index, left] = 1
        if right > left:
            labels[field_index, left + 1:right + 1] = 2
    return labels


def examples(rows, no_write):
    empty = [(0, 0)] * len(FIELDS)
    result = []
    for row in rows:
        for text, boundaries in (row['first'], row['update']):
            result.append((text, boundaries))
        result.extend((row['question'], empty) for _ in (0, 1))
    result.extend((text, empty) for text in no_write for _ in range(12))
    return result


def fit(model, rows, no_write, feature_reader, epochs, device):
    exampleset = examples(rows, no_write)
    texts, boundaries = zip(*exampleset, strict=True)
    codes = characters(texts)
    target_tags = torch.stack([labels_for(row, codes.shape[1]) for _, row in exampleset])
    target_trigger = torch.tensor(
        [[int(start != 0) for start, _ in row] for _, row in exampleset], dtype=torch.long
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0)
    trace = []
    tag_weights = torch.tensor((.25, 2.5, 1.5), device=device)
    trigger_weights = torch.tensor((.75, 2.0), device=device)
    for epoch in range(epochs):
        losses = []
        for batch in torch.randperm(len(exampleset)).split(64):
            batch_texts = [texts[index] for index in batch.tolist()]
            batch_features = feature_reader(batch_texts).to(device)
            batch_codes = codes[batch].to(device)
            if batch_features.shape[1] < batch_codes.shape[1]:
                batch_features = F.pad(batch_features, (0, 0, 0, batch_codes.shape[1] - batch_features.shape[1]))
            tag_logits, trigger_logits = model(batch_codes, batch_features)
            tag_truth = target_tags[batch].to(device)
            trigger_truth = target_trigger[batch].to(device)
            tag_loss = F.cross_entropy(
                tag_logits.permute(0, 1, 3, 2).reshape(-1, 3),
                tag_truth.reshape(-1), weight=tag_weights,
            )
            trigger_loss = F.cross_entropy(
                trigger_logits.flatten(end_dim=1), trigger_truth.flatten(), weight=trigger_weights
            )
            loss = tag_loss + trigger_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        item = {'epoch': epoch + 1, 'loss': sum(losses) / len(losses)}
        trace.append(item)
        if epoch in (0, epochs - 1) or (epoch + 1) % 5 == 0:
            print('BIO writer', item, flush=True)
    return trace, len(exampleset)


class Locator:
    def __init__(self, writer, features, device):
        self.writer, self.features, self.device = writer, features, device

    def __call__(self, text, margin=2.0):
        encoded = self.features([text]).to(self.device)
        return self.writer.write(text, margin, features=encoded)


def evaluate(writer, features, device, tokenizer, holdout):
    locator = Locator(writer, features, device)
    rows = []
    for row in holdout:
        current = row['fields'] | row['changes']
        case = {
            'turns': [row['first'], row['update'], row['question']],
            'initial': row['fields'], 'changes': row['changes'], 'current': current,
        }
        _, metrics = write_metrics(locator, tokenizer, case)
        rows.append(metrics)
    return rows


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    sources = {path: sha(path) for path in (
        HOLDOUT, WRITER_DATA, args.checkpoint, args.database, args.writer,
    )}
    holdout = json.loads(HOLDOUT.read_text(encoding='utf-8'))
    data = json.loads(WRITER_DATA.read_text(encoding='utf-8'))
    transfer = json.loads((ROOT / 'data/dialogue-writer-transfer.json').read_text(encoding='utf-8'))
    train_rows, augmented = training_rows(data, transfer)
    training_texts = {text for row in train_rows for text in (row['first'][0], row['update'][0], row['question'])}
    holdout_texts = {text for row in holdout for text in (row['first'], row['update'], row['question'])}
    if training_texts & holdout_texts:
        raise ValueError('A holdout turn occurs in BIO training')
    payload = torch.load(args.writer, map_location='cpu', weights_only=True)
    if (payload['format'], payload['parent_sha256'], payload['fields'],
        payload['feature_depth'], payload['writer_width']) != (
            'spora-working-turn-experiment-v1', sources[args.checkpoint], FIELDS,
            FEATURE_DEPTH, WRITER_WIDTH):
        raise ValueError('Writer checkpoint is incompatible with the parent')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(24118)
    random.seed(24118)
    torch.set_num_threads(2)
    with load_integrated_durable(args.database, args.checkpoint, load_full_streamed(), MODEL) as runtime:
        decoder = runtime.decoder.eval().requires_grad_(False)
        device = decoder.base.device
        digest = frozen_digest(runtime.organism, runtime.state)
        features = FrozenTurnFeatures(decoder, tokenizer)
        model = TriggerBioValueWriter(WRITER_WIDTH, feature_width=payload['hidden']).to(device)
        trace, count = fit(model, train_rows, augmented['no_write'], features, args.epochs, device)
        model.eval()
        results = evaluate(model, features, device, tokenizer, holdout)
        checkpoint = args.output / 'writer.pt'
        atomic_save({
            'format': 'spora-trigger-bio-writer-v1', 'parent_sha256': sources[args.checkpoint],
            'fields': FIELDS, 'feature_depth': FEATURE_DEPTH, 'writer_width': WRITER_WIDTH,
            'hidden': payload['hidden'], 'writer': {key: value.cpu() for key, value in model.state_dict().items()},
        }, checkpoint)
        if digest != frozen_digest(runtime.organism, runtime.state):
            raise ValueError('BIO writer changed the durable individual')
    if any(sha(path) != value for path, value in sources.items()):
        raise ValueError('A BIO writer input changed during training')
    result = {
        'format': 'spora-trigger-bio-writer-v1',
        'scores': {
            'state_exact': sum(row['state_exact'] for row in results),
            'missing_cases': sum(bool(row['missing']) for row in results),
            'wrong_cases': sum(bool(row['wrong']) for row in results),
            'stale': sum(row['stale'] for row in results),
        },
        'cases': results, 'trace': trace, 'training_examples': count,
        'digest': digest, 'writer_sha256': sha(checkpoint),
        'inputs': {str(path): value for path, value in sources.items()},
        'seconds': time.perf_counter() - started,
        'peak_bytes': torch.cuda.max_memory_allocated(),
    }
    (args.output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(result['scores'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'database', 'writer', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=8)
    options = parser.parse_args()
    if options.epochs < 1:
        parser.error('--epochs must be positive')
    main(options)
