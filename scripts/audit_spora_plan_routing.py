"""Audit category-free routing before the bounded native read plan."""

import argparse
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer

from audit_spora_dialogue_working_state import (
    FEATURE_DEPTH, FORMAT, FrozenTurnFeatures, SemanticLocator, state_from,
)
from audit_spora_native_read_plan import (
    DATA, LABELS, ReadPlan, decode_plan, native_generate, output_metrics, plan_text,
)
from audit_spora_read_plan import training_rows
from audit_spora_writer_transfer import write_metrics
from scripts.audit_spora_native_query_key import write
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_dialogue_working_state import SEED, WRITER_WIDTH
from spora.integrated_durable import load_integrated_durable
from spora.language_training import atomic_save, frozen_digest
from spora.research_dialogue_working_state import FIELDS, TurnValueWriter


class KindRouter(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.project = nn.Linear(hidden, 64)
        self.encoder = nn.GRU(64, 64, batch_first=True)
        self.output = nn.Linear(64, len(LABELS))

    def forward(self, features, lengths):
        projected = self.project(F.layer_norm(features.float(), (features.shape[-1],)))
        hidden, _ = self.encoder(projected)
        return self.output(hidden[torch.arange(len(features), device=features.device), lengths - 1])


def fit(router, features, lengths, targets, epochs, device):
    optimizer = torch.optim.AdamW(router.parameters(), lr=.0025)
    trace = []
    for epoch in range(epochs):
        losses = []
        for batch in torch.randperm(len(targets)).split(32):
            logits = router(features[batch].to(device), lengths[batch].to(device))
            loss = F.cross_entropy(logits, targets[batch].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        trace.append(sum(losses) / len(losses))
        if epoch in (0, epochs - 1) or (epoch + 1) % 5 == 0:
            print('Kind route', epoch + 1, trace[-1], flush=True)
    return trace


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    sources = {path: sha(path) for path in (
        DATA, args.checkpoint, args.database, args.writer, args.planner,
    )}
    held = json.loads(DATA.read_text(encoding='utf-8'))
    planner_payload = torch.load(args.planner, map_location='cpu', weights_only=True)
    writer_payload = torch.load(args.writer, map_location='cpu', weights_only=True)
    if (planner_payload['format'], planner_payload['parent_sha256']) != (
            'spora-read-plan-v1', sources[args.checkpoint]):
        raise ValueError('Planner checkpoint is incompatible with the parent')
    if (writer_payload['format'], writer_payload['parent_sha256'], writer_payload['fields'],
        writer_payload['feature_depth'], writer_payload['writer_width']) != (
            'spora-working-turn-experiment-v1', sources[args.checkpoint], FIELDS,
            FEATURE_DEPTH, WRITER_WIDTH):
        raise ValueError('Writer checkpoint is incompatible with the parent')
    training = training_rows()
    if {row[1] for row in training} & {row['question'] for row in held}:
        raise ValueError('A holdout question occurs in routing training')

    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    args.output.mkdir(parents=True, exist_ok=False)
    with load_integrated_durable(args.database, args.checkpoint, load_full_streamed(), MODEL) as runtime:
        decoder = runtime.decoder.eval().requires_grad_(False)
        device = decoder.base.device
        digest = frozen_digest(runtime.organism, runtime.state)
        features = FrozenTurnFeatures(decoder, tokenizer)
        hidden = int(decoder.base.config.hidden_size)
        train_questions = [row[1] for row in training]
        train_features = features(train_questions)
        train_lengths = torch.tensor([len(text) for text in train_questions])
        train_targets = torch.tensor([tuple(LABELS).index(row[0]) for row in training])
        # Keep the small router and its Adam state on CPU; the frozen decoder
        # already occupies most of the 6 GiB device budget.
        router = KindRouter(hidden)
        trace = fit(router, train_features, train_lengths, train_targets, args.epochs, 'cpu')
        router.to(device)
        router.eval()

        planner_hidden = planner_payload['model']['project.weight'].shape[1]
        planner = ReadPlan(planner_hidden).to(device)
        planner.load_state_dict(planner_payload['model'])
        planner.eval()
        writer = TurnValueWriter(WRITER_WIDTH, feature_width=writer_payload['hidden']).to(device)
        writer.load_state_dict(writer_payload['writer'])
        locator = SemanticLocator(writer.eval(), features)

        questions = [row['question'] for row in held]
        encoded = features(questions)
        lengths = torch.tensor([len(text) for text in questions])
        with torch.inference_mode():
            route_logits = router(encoded.to(device), lengths.to(device))
            routed_kinds = route_logits.argmax(-1)
            heads = planner(encoded.to(device), lengths.to(device), routed_kinds)
        rows = []
        for index, row in enumerate(held):
            kind = tuple(LABELS)[int(routed_kinds[index])]
            guessed = decode_plan(tuple(head[index] for head in heads), kind)
            current = row['fields'] | row['changes']
            case = {
                'turns': [row['first'], row['update'], row['question']],
                'initial': row['fields'], 'changes': row['changes'], 'current': current,
            }
            automatic_state, writes = write_metrics(locator, tokenizer, case)
            state_map = {'oracle': current, 'automatic': writes['final']}
            native = {}
            for state_name, fields in state_map.items():
                text = plan_text(tokenizer, kind, fields, guessed)
                generated = native_generate(decoder, tokenizer, row['question'], text, args.weight)
                native[state_name] = {
                    **generated,
                    **output_metrics(generated['answer'], generated['ended'], kind, fields, guessed,
                                     row['expected']),
                }
            rows.append({
                'expected_kind': row['kind'], 'routed_kind': kind,
                'route_correct': kind == row['kind'], 'question': row['question'],
                'expected': row['expected'], 'plan': guessed, 'writes': writes,
                'native': native,
            })
            print(index + 1, row['kind'], '=>', kind, native['oracle']['answer'], flush=True)

        if digest != frozen_digest(runtime.organism, runtime.state):
            raise ValueError('Routing audit changed the durable individual')
    if any(sha(path) != value for path, value in sources.items()):
        raise ValueError('Routing audit changed an input')

    scores = {
        'route_correct': sum(row['route_correct'] for row in rows),
        'count': len(rows),
        'oracle_renderable': sum(row['native']['oracle']['renderable'] for row in rows),
        'oracle_exact': sum(row['native']['oracle']['exact'] for row in rows),
        'automatic_renderable': sum(row['native']['automatic']['renderable'] for row in rows),
        'automatic_exact': sum(row['native']['automatic']['exact'] for row in rows),
    }
    route_file = args.output / 'router.pt'
    atomic_save({'format': 'spora-kind-router-v1', 'parent_sha256': sources[args.checkpoint],
                 'model': {key: value.cpu() for key, value in router.state_dict().items()}}, route_file)
    write(args.output / 'result.json', {
        'format': 'spora-native-read-plan-routing-v1', 'scores': scores, 'cases': rows,
        'trace': trace, 'training_rows': len(training), 'digest': digest,
        'router_sha256': sha(route_file), 'inputs': {str(path): value for path, value in sources.items()},
        'seconds': time.perf_counter() - started,
        'peak_bytes': torch.cuda.max_memory_allocated(),
    })
    print(scores, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'database', 'writer', 'planner', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--weight', type=float, default=32.)
    options = parser.parse_args()
    if options.epochs < 1:
        parser.error('--epochs must be positive')
    if not 0. <= options.weight <= 64.:
        parser.error('--weight must be in [0, 64]')
    main(options)
