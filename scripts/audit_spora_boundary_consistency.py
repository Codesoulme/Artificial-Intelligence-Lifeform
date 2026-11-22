"""Train the existing writer with an additional joint span consistency loss."""

import argparse
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

from audit_spora_dialogue_working_state import FEATURE_DEPTH, FORMAT, FrozenTurnFeatures
from audit_spora_writer_transfer import TRANSFER, examples_for, write_metrics
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_dialogue_working_state import DATA as WRITER_DATA, MARGIN, WRITER_WIDTH
from spora.integrated_durable import load_integrated_durable
from spora.language_training import atomic_save, frozen_digest
from spora.research_dialogue_working_state import FIELDS, SPAN_LIMIT, TurnValueWriter, characters


HOLDOUT = ROOT / 'data/dialogue-read-plan-holdout.json'


class FeatureLocator:
    def __init__(self, writer, features, device):
        self.writer, self.features, self.device = writer, features, device

    def __call__(self, text):
        return self.writer.write(text, MARGIN, features=self.features([text]).to(self.device))


def examples(data, transfer):
    rows = examples_for(data, transfer)
    empty = [(0, 0)] * len(FIELDS)
    result = [row['first'] for row in rows] + [row['update'] for row in rows]
    result.extend((row['question'], empty) for row in rows[::2])
    result.extend((text, empty) for text in data['no_write'] for _ in range(12))
    return result


def targets(boundaries, length):
    target = torch.zeros((len(boundaries), len(FIELDS), 2), dtype=torch.long)
    for row, offsets in enumerate(boundaries):
        for field, (start, end) in enumerate(offsets):
            if start:
                target[row, field] = torch.tensor((start, end))
    return target


def consistency_loss(logits, truth):
    length = logits.shape[-1] - 1
    indices = torch.arange(length, device=logits.device)
    span_length = indices[None, :] - indices[:, None]
    valid = (indices[None, :] >= indices[:, None]) & (span_length < SPAN_LIMIT)
    pair = logits[:, :, 0, 1:].unsqueeze(-1) + logits[:, :, 1, 1:].unsqueeze(-2)
    pair = pair.masked_fill(~valid[None, None], -torch.inf)
    log_normalizer = torch.logsumexp(pair.flatten(-2), dim=-1)
    present = truth[:, :, 0].gt(0)
    start = (truth[:, :, 0] - 1).clamp_min(0)
    end = (truth[:, :, 1] - 1).clamp_min(0)
    selected = pair.gather(-2, start.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, length))
    selected = selected.gather(-1, end.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
    losses = (log_normalizer - selected)[present]
    return losses.mean() if losses.numel() else logits.sum() * 0.


def fit(model, rows, feature_reader, epochs, joint_weight):
    texts, boundaries = zip(*rows, strict=True)
    codes = characters(texts)
    features = feature_reader(texts)
    truth = targets(boundaries, codes.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005)
    trace = []
    for epoch in range(epochs):
        losses = []
        for batch in torch.randperm(len(rows)).split(32):
            logits = model(codes[batch], features[batch])
            standard = F.cross_entropy(logits.flatten(0, 2), truth[batch].flatten(), reduction='none')
            flat_truth = truth[batch].flatten()
            standard = (standard * torch.where(flat_truth > 0, 2.5, 1.)).mean()
            joint = consistency_loss(logits, truth[batch])
            loss = standard + joint_weight * joint
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append((float(loss.detach()), float(standard.detach()), float(joint.detach())))
        item = {
            'epoch': epoch + 1,
            'loss': sum(row[0] for row in losses) / len(losses),
            'standard': sum(row[1] for row in losses) / len(losses),
            'joint': sum(row[2] for row in losses) / len(losses),
        }
        trace.append(item)
        if epoch in (0, epochs - 1) or (epoch + 1) % 5 == 0:
            print('Boundary consistency', item, flush=True)
    return trace


def score(writer, features, device, tokenizer, holdout):
    locator = FeatureLocator(writer, features, device)
    rows = []
    for row in holdout:
        current = row['fields'] | row['changes']
        case = {
            'turns': [row['first'], row['update'], row['question']],
            'initial': row['fields'], 'changes': row['changes'], 'current': current,
        }
        _, measures = write_metrics(locator, tokenizer, case)
        rows.append(measures)
    return rows


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    sources = {path: sha(path) for path in (
        HOLDOUT, WRITER_DATA, TRANSFER, args.checkpoint, args.database, args.writer,
    )}
    holdout = json.loads(HOLDOUT.read_text(encoding='utf-8'))
    data = json.loads(WRITER_DATA.read_text(encoding='utf-8'))
    transfer = json.loads(TRANSFER.read_text(encoding='utf-8'))
    payload = torch.load(args.writer, map_location='cpu', weights_only=True)
    if (payload['format'], payload['parent_sha256'], payload['fields'], payload['feature_depth'],
        payload['writer_width']) != (FORMAT, sources[args.checkpoint], FIELDS, FEATURE_DEPTH, WRITER_WIDTH):
        raise ValueError('Writer checkpoint is incompatible with the parent')
    training = examples(data, transfer)
    if {text for text, _ in training} & {row['question'] for row in holdout}:
        raise ValueError('A holdout question occurs in consistency training')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(24117)
    random.seed(24117)
    torch.set_num_threads(2)
    with load_integrated_durable(args.database, args.checkpoint, load_full_streamed(), MODEL) as runtime:
        decoder = runtime.decoder.eval().requires_grad_(False)
        device = decoder.base.device
        digest = frozen_digest(runtime.organism, runtime.state)
        features = FrozenTurnFeatures(decoder, tokenizer)
        model = TurnValueWriter(WRITER_WIDTH, feature_width=payload['hidden'])
        model.load_state_dict(payload['writer'])
        trace = fit(model, training, features, args.epochs, args.joint_weight)
        model.to(device).eval()
        rows = score(model, features, device, tokenizer, holdout)
        checkpoint = args.output / 'writer.pt'
        atomic_save({**payload, 'writer': {key: value.cpu() for key, value in model.state_dict().items()}}, checkpoint)
        if digest != frozen_digest(runtime.organism, runtime.state):
            raise ValueError('Consistency training changed the durable individual')
    if any(sha(path) != value for path, value in sources.items()):
        raise ValueError('Consistency training changed an input')
    result = {
        'format': 'spora-boundary-consistency-v1', 'joint_weight': args.joint_weight,
        'scores': {
            'state_exact': sum(row['state_exact'] for row in rows),
            'missing_cases': sum(bool(row['missing']) for row in rows),
            'wrong_cases': sum(bool(row['wrong']) for row in rows),
            'stale': sum(row['stale'] for row in rows),
        },
        'cases': rows, 'trace': trace, 'training_examples': len(training),
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
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--joint-weight', type=float, default=.25)
    options = parser.parse_args()
    if options.epochs < 1 or options.joint_weight < 0.:
        parser.error('epochs must be positive and joint-weight must be nonnegative')
    main(options)
