"""Audit a learned complete-span writer on two independent holdouts."""

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
from spora.research_candidate_span_writer import CandidateSpanWriter, candidate_indices
from spora.research_dialogue_working_state import FIELDS, TurnValueWriter, characters


HOLDOUT = ROOT / 'data/dialogue-read-plan-holdout.json'
TRANSFER = ROOT / 'data/dialogue-writer-transfer.json'
FORMAT = 'spora-working-turn-experiment-v1'


def examples(rows, no_write):
    empty = [(0, 0)] * len(FIELDS)
    result = [example for row in rows for example in (row['first'], row['update'])]
    result.extend((row['question'], empty) for row in rows[::2])
    result.extend((text, empty) for text in no_write for _ in range(12))
    return result


def target_index(boundaries, width, mapping):
    target = []
    for start, end in boundaries:
        if start == 0:
            target.append(0)
            continue
        key = (start - 1, end - 1)
        if key not in mapping:
            raise ValueError('Training span is outside the candidate set')
        target.append(mapping[key] + 1)
    return target


def fit(model, rows, no_write, feature_reader, epochs, device):
    exampleset = examples(rows, no_write)
    texts, boundaries = zip(*exampleset, strict=True)
    codes = characters(texts)
    starts, ends = candidate_indices(codes.shape[1])
    mapping = {(int(start), int(end)): index for index, (start, end) in
               enumerate(zip(starts.tolist(), ends.tolist(), strict=True))}
    targets = torch.tensor([target_index(row, codes.shape[1], mapping) for row in boundaries])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0)
    trace = []
    for epoch in range(epochs):
        losses = []
        for batch in torch.randperm(len(exampleset)).split(64):
            batch_texts = [texts[index] for index in batch.tolist()]
            batch_features = feature_reader(batch_texts).to(device)
            batch_codes = codes[batch].to(device)
            if batch_features.shape[1] < batch_codes.shape[1]:
                batch_features = F.pad(batch_features, (0, 0, 0, batch_codes.shape[1] - batch_features.shape[1]))
            logits = model(batch_codes, batch_features)
            truth = targets[batch].to(device)
            loss = F.cross_entropy(logits.flatten(end_dim=1), truth.flatten(), reduction='none')
            loss = (loss * torch.where(truth.flatten() > 0, 2.5, 1.)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        item = {'epoch': epoch + 1, 'loss': sum(losses) / len(losses)}
        trace.append(item)
        if epoch in (0, epochs - 1) or (epoch + 1) % 5 == 0:
            print('Candidate span writer', item, flush=True)
    return trace, len(exampleset)


class Locator:
    def __init__(self, writer, features, device, margin):
        self.writer, self.features, self.device, self.margin = writer, features, device, margin

    def __call__(self, text, margin=None):
        encoded = self.features([text]).to(self.device)
        return self.writer.write(text, self.margin if margin is None else margin, features=encoded)


def evaluate(locator, tokenizer, cases):
    rows = []
    for row in cases:
        current = row['fields'] | row['changes']
        case = {
            'turns': [row['first'], row['update'], row['question']],
            'initial': row['fields'], 'changes': row['changes'], 'current': current,
        }
        _, measures = write_metrics(locator, tokenizer, case)
        rows.append(measures)
    return rows


def score(rows):
    return {
        'state_exact': sum(row['state_exact'] for row in rows),
        'missing_cases': sum(bool(row['missing']) for row in rows),
        'wrong_cases': sum(bool(row['wrong']) for row in rows),
        'stale': sum(row['stale'] for row in rows),
    }


def initialize_from_base(model, payload, hidden):
    base = TurnValueWriter(WRITER_WIDTH, feature_width=hidden)
    base.load_state_dict(payload['writer'])
    model.embedding.load_state_dict(base.embedding.state_dict())
    model.semantic.load_state_dict(base.semantic.state_dict())
    for target, source in zip(model.context, base.context, strict=True):
        target.load_state_dict(source.state_dict())


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    sources = {path: sha(path) for path in (HOLDOUT, TRANSFER, WRITER_DATA,
                                             args.checkpoint, args.database, args.writer)}
    holdout = json.loads(HOLDOUT.read_text(encoding='utf-8'))
    transfer = json.loads(TRANSFER.read_text(encoding='utf-8'))
    data = json.loads(WRITER_DATA.read_text(encoding='utf-8'))
    transfer_data = json.loads((ROOT / 'data/dialogue-writer-transfer.json').read_text(encoding='utf-8'))
    train_rows, augmented = training_rows(data, transfer_data)
    train_texts = {text for row in train_rows for text in (row['first'][0], row['update'][0], row['question'])}
    held_texts = {text for row in holdout for text in (row['first'], row['update'], row['question'])}
    if train_texts & held_texts:
        raise ValueError('A read-plan holdout turn occurs in candidate-span training')
    payload = torch.load(args.writer, map_location='cpu', weights_only=True)
    if (payload['format'], payload['parent_sha256'], payload['fields'],
        payload['feature_depth'], payload['writer_width']) != (
            FORMAT, sources[args.checkpoint], FIELDS, FEATURE_DEPTH, WRITER_WIDTH):
        raise ValueError('Writer checkpoint is incompatible with the parent')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(24119)
    random.seed(24119)
    torch.set_num_threads(2)
    with load_integrated_durable(args.database, args.checkpoint, load_full_streamed(), MODEL) as runtime:
        decoder = runtime.decoder.eval().requires_grad_(False)
        device = decoder.base.device
        digest = frozen_digest(runtime.organism, runtime.state)
        features = FrozenTurnFeatures(decoder, tokenizer)
        model = CandidateSpanWriter(WRITER_WIDTH, feature_width=payload['hidden']).to(device)
        initialize_from_base(model, payload, payload['hidden'])
        trace, count = fit(model, train_rows, augmented['no_write'], features, args.epochs, device)
        model.eval()
        split_rows = {}
        for name, cases in (('read_plan', holdout), ('writer_transfer', transfer['held'])):
            result_rows = evaluate(Locator(model, features, device, args.margin), tokenizer, cases)
            split_rows[name] = {'score': score(result_rows), 'cases': result_rows}
        checkpoint = args.output / 'writer.pt'
        atomic_save({
            'format': 'spora-candidate-span-writer-v1', 'parent_sha256': sources[args.checkpoint],
            'fields': FIELDS, 'feature_depth': FEATURE_DEPTH, 'writer_width': WRITER_WIDTH,
            'hidden': payload['hidden'], 'writer': {key: value.cpu() for key, value in model.state_dict().items()},
        }, checkpoint)
        if digest != frozen_digest(runtime.organism, runtime.state):
            raise ValueError('Candidate span writer changed the durable individual')
    if any(sha(path) != value for path, value in sources.items()):
        raise ValueError('Candidate span writer input changed during training')
    result = {
        'format': 'spora-candidate-span-writer-v1', 'scores': {name: split['score'] for name, split in split_rows.items()},
        'cases': {name: split['cases'] for name, split in split_rows.items()}, 'trace': trace,
        'training_examples': count, 'digest': digest, 'writer_sha256': sha(checkpoint),
        'inputs': {str(path): value for path, value in sources.items()},
        'margin': args.margin, 'seconds': time.perf_counter() - started,
        'peak_bytes': torch.cuda.max_memory_allocated(),
    }
    (args.output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(result['scores'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'database', 'writer', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--margin', type=float, default=2.0)
    options = parser.parse_args()
    if options.epochs < 1:
        parser.error('--epochs must be positive')
    main(options)
