"""Train and audit a joint character-span writer without changing the formal writer."""

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

from audit_spora_dialogue_working_state import (
    FEATURE_DEPTH, FORMAT, FrozenTurnFeatures, SemanticLocator, state_from,
    verify_date_isolation,
)
from audit_spora_native_read_plan import (
    DATA as PLAN_DATA, LABELS, ReadPlan, decode_plan, native_generate, output_metrics,
    plan_text,
)
from audit_spora_read_plan import DATA as PLAN_HOLDOUT
from audit_spora_writer_transfer import TRANSFER, examples_for, write_metrics
from scripts.audit_spora_native_query_key import write
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_dialogue_working_state import DATA as WRITER_DATA, WRITER_WIDTH
from spora.integrated_durable import load_integrated_durable
from spora.language_training import atomic_save, frozen_digest
from spora.research_dialogue_working_state import (
    FIELDS, SPAN_LIMIT, TurnValueWriter, WorkingTurnReader, characters,
)
from spora.research_joint_span_writer import JointSpanWriter


JOINT_FORMAT = 'spora-joint-span-writer-v1'


def examples(data, transfer):
    rows = examples_for(data, transfer)
    empty = [(0, 0)] * len(FIELDS)
    result = [row['first'] for row in rows] + [row['update'] for row in rows]
    result.extend((row['question'], empty) for row in rows[::2])
    result.extend((text, empty) for text in data['no_write'] for _ in range(12))
    return result


def span_targets(boundaries, length):
    target = torch.zeros((len(boundaries), len(FIELDS)), dtype=torch.long)
    for row, offsets in enumerate(boundaries):
        for field, (start, end) in enumerate(offsets):
            if start:
                if not 1 <= start <= end <= length or end - start >= SPAN_LIMIT:
                    raise ValueError('Training span is outside the joint writer limits')
                target[row, field] = 1 + (start - 1) * length + (end - 1)
    return target


def fit(model, rows, feature_reader, epochs):
    texts, boundaries = zip(*rows, strict=True)
    codes = characters(texts)
    features = feature_reader(texts)
    targets = span_targets(boundaries, codes.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    trace = []
    for epoch in range(epochs):
        losses = []
        for batch in torch.randperm(len(rows)).split(32):
            logits = model(codes[batch], features[batch])
            truth = targets[batch]
            loss = F.cross_entropy(logits.transpose(1, 2), truth, reduction='none')
            loss = (loss * torch.where(truth > 0, 2.5, 1.)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        item = {'epoch': epoch + 1, 'loss': sum(losses) / len(losses)}
        trace.append(item)
        if epoch in (0, epochs - 1) or (epoch + 1) % 5 == 0:
            print('Joint writer', item, flush=True)
    return trace


def initialize_from_writer(model, path, hidden, parent_sha):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload['format'], payload['parent_sha256'], payload['fields'], payload['hidden']) != (
            FORMAT, parent_sha, FIELDS, hidden):
        raise ValueError('Base writer is incompatible with the parent')
    base = TurnValueWriter(WRITER_WIDTH, feature_width=hidden)
    base.load_state_dict(payload['writer'])
    model.embedding.load_state_dict(base.embedding.state_dict())
    model.semantic.load_state_dict(base.semantic.state_dict())
    for target, source in zip(model.context, base.context, strict=True):
        target.load_state_dict(source.state_dict())
    with torch.no_grad():
        boundary = base.boundary.weight.reshape(len(FIELDS), 2, WRITER_WIDTH)
        boundary_bias = base.boundary.bias.reshape(len(FIELDS), 2)
        model.start_bias.weight.copy_(boundary[:, 0])
        model.start_bias.bias.copy_(boundary_bias[:, 0])
        model.end_bias.weight.copy_(boundary[:, 1])
        model.end_bias.bias.copy_(boundary_bias[:, 1])
        absent = base.absent.weight.reshape(len(FIELDS), 2, WRITER_WIDTH)
        absent_bias = base.absent.bias.reshape(len(FIELDS), 2)
        model.absent.weight.copy_(absent.sum(1))
        model.absent.bias.copy_(absent_bias.sum(1))


class FeatureLocator:
    def __init__(self, writer, features, device):
        self.writer = writer
        self.features = features
        self.device = device

    def __call__(self, text, margin=2.):
        return self.writer.write(text, margin, features=self.features([text]).to(self.device))


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    sources = {path: sha(path) for path in (
        PLAN_HOLDOUT, PLAN_DATA, WRITER_DATA, TRANSFER, args.checkpoint,
        args.database, args.planner, *(path for path in (args.base_writer,) if path is not None),
    )}
    holdout = json.loads(PLAN_HOLDOUT.read_text(encoding='utf-8'))
    data = json.loads(WRITER_DATA.read_text(encoding='utf-8'))
    transfer = json.loads(TRANSFER.read_text(encoding='utf-8'))
    planner_payload = torch.load(args.planner, map_location='cpu', weights_only=True)
    if (planner_payload['format'], planner_payload['parent_sha256']) != (
            'spora-read-plan-v1', sources[args.checkpoint]):
        raise ValueError('Planner checkpoint is incompatible with the parent')
    training = examples(data, transfer)
    if {text for text, _ in training} & {row['question'] for row in holdout}:
        raise ValueError('A holdout question occurs in joint writer training')

    torch.manual_seed(24117)
    random.seed(24117)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    args.output.mkdir(parents=True, exist_ok=False)
    with load_integrated_durable(args.database, args.checkpoint, load_full_streamed(), MODEL) as runtime:
        decoder = runtime.decoder.eval().requires_grad_(False)
        device = decoder.base.device
        digest = frozen_digest(runtime.organism, runtime.state)
        features = FrozenTurnFeatures(decoder, tokenizer)
        hidden = int(decoder.base.config.hidden_size)
        writer = JointSpanWriter(WRITER_WIDTH, feature_width=hidden)
        if args.base_writer is not None:
            initialize_from_writer(writer, args.base_writer, hidden, sources[args.checkpoint])
        train_trace = fit(writer, training, features, args.epochs)
        writer.to(device).eval()
        locator = FeatureLocator(writer, features, device)

        planner_hidden = planner_payload['model']['project.weight'].shape[1]
        planner = ReadPlan(planner_hidden).to(device)
        planner.load_state_dict(planner_payload['model'])
        planner.eval()
        questions = [row['question'] for row in holdout]
        question_features = features(questions).to(device)
        lengths = torch.tensor([len(question) for question in questions], device=device)
        kinds = torch.tensor([tuple(LABELS).index(row['kind']) for row in holdout], device=device)
        with torch.inference_mode():
            heads = planner(question_features, lengths, kinds)

        rows = []
        for index, row in enumerate(holdout):
            current = row['fields'] | row['changes']
            gold = (tuple(row['order']), row['join'], row['style'])
            plan = decode_plan(tuple(head[index] for head in heads), row['kind'])
            case = {
                'turns': [row['first'], row['update'], row['question']],
                'initial': row['fields'], 'changes': row['changes'], 'current': current,
            }
            automatic_state, measures = write_metrics(locator, tokenizer, case)
            native = {}
            for state_name, fields in (('oracle', current), ('automatic', measures['final'])):
                text = plan_text(tokenizer, row['kind'], fields, gold)
                generated = native_generate(decoder, tokenizer, row['question'], text, 32.)
                native[state_name] = {
                    **generated,
                    **output_metrics(generated['answer'], generated['ended'], row['kind'], fields,
                                     gold, row['expected']),
                }
            rows.append({
                'kind': row['kind'], 'question': row['question'], 'expected': row['expected'],
                'plan': plan, 'writes': measures, 'native': native,
            })
            print(index + 1, row['kind'], native['automatic']['answer'], flush=True)

        dates = verify_date_isolation(
            runtime, WorkingTurnReader(decoder.base.config.hidden_size).to(device), tokenizer,
            state_from(holdout[0]['fields'], tokenizer),
        )
        if digest != frozen_digest(runtime.organism, runtime.state):
            raise ValueError('Joint writer audit changed the durable individual')

        checkpoint = args.output / 'writer.pt'
        atomic_save({
            'format': JOINT_FORMAT, 'parent_sha256': sources[args.checkpoint],
            'fields': FIELDS, 'hidden': int(decoder.base.config.hidden_size),
            'writer_width': WRITER_WIDTH, 'span_limit': SPAN_LIMIT,
            'writer': {key: value.cpu() for key, value in writer.state_dict().items()},
        }, checkpoint)
    if any(sha(path) != value for path, value in sources.items()):
        raise ValueError('Joint writer audit changed an input')

    scores = {
        name: {
            'state_exact': sum(row['writes']['state_exact'] for row in rows)
            if name == 'automatic' else None,
            'renderable': sum(row['native'][name]['renderable'] for row in rows),
            'fields': sum(row['native'][name]['fields'] for row in rows),
            'exact': sum(row['native'][name]['exact'] for row in rows),
        }
        for name in ('oracle', 'automatic')
    }
    write(args.output / 'result.json', {
        'format': JOINT_FORMAT, 'scores': scores, 'cases': rows, 'date': dates,
        'trace': train_trace, 'training_examples': len(training), 'digest': digest,
        'writer_sha256': sha(checkpoint), 'inputs': {str(path): value for path, value in sources.items()},
        'seconds': time.perf_counter() - started,
        'peak_bytes': torch.cuda.max_memory_allocated(),
    })
    print(scores, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'database', 'planner', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--base-writer', type=Path)
    parser.add_argument('--epochs', type=int, default=10)
    options = parser.parse_args()
    if options.epochs < 1:
        parser.error('--epochs must be positive')
    main(options)
