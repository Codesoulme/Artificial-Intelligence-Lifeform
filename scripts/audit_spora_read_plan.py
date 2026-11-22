"""Audit a supervised decomposition of bounded dialogue readout."""

import argparse
from itertools import permutations
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

from audit_spora_dialogue_working_state import (FEATURE_DEPTH, FORMAT, FrozenTurnFeatures, SemanticLocator,
                                                state_from, verify_date_isolation)
from audit_spora_question_read import EXTRA_STYLE, FIELD_WORDS, LABELS, STYLE
from audit_spora_writer_transfer import write_metrics
from scripts.audit_spora_native_query_key import write
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_dialogue_working_state import SEED, WRITER_WIDTH
from spora.integrated_durable import load_integrated_durable
from spora.language_training import atomic_save, frozen_digest
from spora.research_dialogue_working_state import FIELDS, TurnValueWriter, WorkingTurnReader


DATA = ROOT / 'data/dialogue-read-plan-holdout.json'
STYLES = ('only', 'slash', 'comma', 'lines')
ORDERS = tuple(dict.fromkeys(order for fields in LABELS.values() for size in (1, 2, 3)
                            for order in permutations(fields, size)))


class ReadPlan(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.project = nn.Linear(hidden, 64)
        self.encoder = nn.GRU(64, 64, batch_first=True)
        self.order = nn.Linear(64 + len(LABELS), len(ORDERS))
        self.group = nn.Linear(64 + len(LABELS), 2)
        self.style = nn.Linear(64 + len(LABELS), len(STYLES))

    def forward(self, features, lengths, kinds):
        projected = self.project(F.layer_norm(features.float(), (features.shape[-1],)))
        hidden, _ = self.encoder(projected)
        last = hidden[torch.arange(len(features), device=features.device), lengths - 1]
        x = torch.cat((last, F.one_hot(kinds, num_classes=len(LABELS))), dim=-1)
        return self.order(x), self.group(x), self.style(x)


def training_rows():
    rng = random.Random(SEED)
    rows = []
    explicit = {
        'only': ('仅报告{fields}这一项即可。', '不用复述全部，给我{fields}便可。'),
        'slash': ('请列{fields}，每两项之间加斜线。', '斜杠连接，输出顺序为{fields}。'),
        'comma': ('逗号连接这些内容：{fields}。', '先后次序是{fields}，项目之间写逗号。'),
        'lines': ('每项各用一行，按{fields}写。', '换行列出各项，依次是{fields}。'),
    }
    for kind, allowed in LABELS.items():
        for order in ORDERS:
            if not set(order) <= allowed.keys():
                continue
            styles = ('only',) if len(order) == 1 else STYLES[1:]
            for style in styles:
                for phrase in STYLE[style] + EXTRA_STYLE[style] + explicit[style]:
                    for trial in range(4):
                        words = [rng.choice(FIELD_WORDS[name]) for name in order]
                        joiner = ('、', '和', '，再给', '，然后报')[trial]
                        question = phrase.format(fields=joiner.join(words))
                        rows.append((kind, question, order, False, style))
                if style in ('comma', 'slash') and 'day' in order and 'time' in order:
                    index = order.index('day')
                    if index + 1 == order.index('time') and len(order) == 3:
                        for phrase in STYLE[style] + EXTRA_STYLE[style] + explicit[style]:
                            for part in ('日期和时刻并作一项', '日子跟时间写在一起', '日期和钟点合成一项'):
                                place = rng.choice(FIELD_WORDS['place'])
                                fields = f'{place}、{part}' if index else f'{part}、{place}'
                                rows.append((kind, phrase.format(fields=fields), order, True, style))
    return list(dict.fromkeys(rows))


def vectors(rows, features, device):
    texts = [row[1] for row in rows]
    kinds = [tuple(LABELS).index(row[0]) for row in rows]
    return (features(texts).to(device), torch.tensor(list(map(len, texts)), device=device),
            torch.tensor(kinds, device=device))


def fit(model, rows, features, device, epochs):
    encoded, lengths, kinds = vectors(rows, features, 'cpu')
    targets = torch.tensor([(ORDERS.index(order), int(group), STYLES.index(style))
                            for _, _, order, group, style in rows], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0025)
    for epoch in range(epochs):
        for batch in torch.randperm(len(rows), device=device).cpu().split(24):
            heads = model(encoded[batch].to(device), lengths[batch].to(device), kinds[batch].to(device))
            expected = targets.index_select(0, batch.to(device))
            loss = sum(F.cross_entropy(head, expected[:, index]) for index, head in enumerate(heads))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch in (0, epochs - 1) or (epoch + 1) % 5 == 0:
            print('Read plan', epoch + 1, float(loss.detach()), flush=True)


def decode_plan(heads, kind):
    order_scores, group_scores, style_scores = heads
    allowed = LABELS[kind].keys()
    choices = [index for index, order in enumerate(ORDERS) if set(order) <= allowed]
    order = ORDERS[max(choices, key=lambda index: float(order_scores[index]))]
    return order, bool(group_scores.argmax()), STYLES[int(style_scores.argmax())]


def render(kind, fields, plan):
    order, joined, style = plan
    if (not order or any(field not in fields for field in order)
            or (style == 'only') != (len(order) == 1)):
        return None
    values = [fields[field] for field in order]
    if joined:
        if 'day' not in order or 'time' not in order or style not in ('comma', 'slash'):
            return None
        position = order.index('day')
        if position + 1 != order.index('time'):
            return None
        values[position:position + 2] = [''.join(values[position:position + 2])]
    if style == 'lines':
        return '\n'.join(LABELS[kind][name] + '：' + value for name, value in zip(order, values, strict=True))
    return {'only': '', 'comma': '，', 'slash': ' / '}[style].join(values)


@torch.inference_mode()
def evaluate(rows, model, features, locator, tokenizer, device):
    encoded, lengths, kinds = vectors([(row['kind'], row['question']) for row in rows], features, device)
    results = []
    for row, heads in zip(rows, zip(*model(encoded, lengths, kinds), strict=True), strict=True):
        kind = row['kind']
        current = row['fields'] | row['changes']
        gold = tuple(row['order']), row['join'], row['style']
        guessed = decode_plan(heads, kind)
        if render(kind, current, gold) != row['expected']:
            raise ValueError('Holdout answer disagrees with its declared plan')
        case = {'turns': [row['first'], row['update'], row['question']], 'initial': row['fields'],
                'current': current, 'changes': row['changes']}
        _, metrics = write_metrics(locator, tokenizer, case)
        hypotheses = {'oracle_state': render(kind, current, guessed),
                      'automatic': render(kind, metrics['final'], guessed),
                      'oracle_order': render(kind, current, (gold[0], guessed[1], guessed[2])),
                      'oracle_group': render(kind, current, (guessed[0], gold[1], guessed[2])),
                      'oracle_style': render(kind, current, (guessed[0], guessed[1], gold[2])),
                      'oracle_plan': render(kind, metrics['final'], gold)}
        results.append({'question': row['question'], 'kind': kind, 'expected': row['expected'], 'gold': gold,
                        'guessed': guessed, 'writes': metrics, 'answers': hypotheses})
    return results


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    sources = {path: sha(path) for path in (DATA, args.checkpoint, args.database, args.writer)}
    held = json.loads(DATA.read_text(encoding='utf-8'))
    train = training_rows()
    if {row[1] for row in train} & {row['question'] for row in held}:
        raise ValueError('A holdout question occurs in training')
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    payload = torch.load(args.writer, map_location='cpu', weights_only=True)
    binding = (FORMAT, sources[args.checkpoint], FIELDS, FEATURE_DEPTH, WRITER_WIDTH)
    if tuple(payload[key] for key in ('format', 'parent_sha256', 'fields', 'feature_depth', 'writer_width')) != binding:
        raise ValueError('Incompatible writer checkpoint')
    args.output.mkdir(parents=True, exist_ok=False)
    with load_integrated_durable(args.database, args.checkpoint, load_full_streamed(), MODEL) as runtime:
        decoder = runtime.decoder.eval().requires_grad_(False)
        digest = frozen_digest(runtime.organism, runtime.state)
        device = decoder.base.device
        features = FrozenTurnFeatures(decoder, tokenizer)
        writer = TurnValueWriter(WRITER_WIDTH, feature_width=payload['hidden'])
        writer.load_state_dict(payload['writer'])
        locator = SemanticLocator(writer.to(device).eval(), features)
        model = ReadPlan(payload['hidden']).to(device)
        fit(model, train, features, device, args.epochs)
        results = evaluate(held, model.eval(), features, locator, tokenizer, device)
        dates = verify_date_isolation(runtime, WorkingTurnReader(payload['hidden']).to(device), tokenizer,
                                      state_from(held[0]['fields'], tokenizer))
        if frozen_digest(runtime.organism, runtime.state) != digest:
            raise ValueError('Read plan changed the individual')
    if any(sha(path) != value for path, value in sources.items()):
        raise ValueError('A probe input changed')
    file = args.output / 'planner.pt'
    atomic_save({'format': 'spora-read-plan-v1', 'parent_sha256': sources[args.checkpoint],
                 'model': {name: value.cpu() for name, value in model.state_dict().items()}}, file)
    scores = {arm: sum(row['answers'][arm] == row['expected'] for row in results)
              for arm in results[0]['answers']}
    components = {name: sum(row['guessed'][index] == row['gold'][index] for row in results)
                  for index, name in enumerate(('order', 'group', 'style'))}
    write(args.output / 'result.json', {'scores': scores, 'components': components, 'cases': results,
                                        'date': dates, 'digest': digest, 'train_rows': len(train),
                                        'planner_sha256': sha(file),
                                        'inputs': {str(path): value for path, value in sources.items()},
                                        'seconds': time.perf_counter() - started,
                                        'peak_bytes': torch.cuda.max_memory_allocated()})
    print(scores, components, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'database', 'writer', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=22)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    main(args)
