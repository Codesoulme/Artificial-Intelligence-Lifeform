"""Expand compositional Chinese write examples and retrain the existing writer."""

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

from audit_spora_dialogue_working_state import FEATURE_DEPTH, FrozenTurnFeatures, FORMAT
from audit_spora_writer_transfer import TRANSFER, examples_for, write_metrics
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_dialogue_working_state import DATA as WRITER_DATA, MARGIN, WRITER_WIDTH
from spora.integrated_durable import load_integrated_durable
from spora.language_training import atomic_save, frozen_digest
from spora.research_dialogue_working_state import FIELDS, TurnValueWriter, characters


HOLDOUT = ROOT / 'data/dialogue-read-plan-holdout.json'


def expanded_transfer(data, transfer):
    result = deepcopy(transfer)
    extra = result['augmentation']
    extra['appointment']['first'].extend((
        '提醒写作：{day}{time}在{place}见面。',
        '预定为{day}{time}，地点填{place}。',
    ))
    extra['appointment']['update'].extend((
        ['day', 'time', '日期换成{day}{time}，地点继续使用。'],
        ['place', '碰面地点调整为{place}，日期和钟点不变。'],
    ))
    extra['parcel']['first'].extend((
        '货物数量标作{quantity}，收件人{person}，发送至{place}。',
        '数量是{quantity}，寄给{person}，地址为{place}。',
    ))
    extra['parcel']['update'].extend((
        ['quantity', '数量重新核对为{quantity}，姓名和地址保留。'],
        ['person', '请让{person}签收，件数和地址不改。'],
    ))
    extra['route']['first'].extend((
        '请在{place}集合，准备{gear}，之后{direction}。',
        '到{place}碰头，携带{gear}并沿{direction}前进。',
    ))
    extra['route']['update'].extend((
        ['gear', '准备品换成{gear}，位置和方向不动。'],
        ['gear', '携带物更新为{gear}，其余安排照旧。'],
    ))
    values = result.setdefault('compositional_values', {})
    values['quantity'] = ('十二套', '十三包', '十四袋', '十五卷', '十六册', '十七盒',
                          '十八份', '二十袋', '二十一套', '二十三卷')
    values['gear'] = ('短绳', '长绳', '防潮布', '急救包', '测距表', '备用灯')
    values['place'] = ('青栎厅', '月桂巷', '石榴台', '云杉廊')
    return result


def training_rows(data, transfer):
    augmented = deepcopy(data)
    expanded = expanded_transfer(data, transfer)
    for field, values in expanded.get('compositional_values', {}).items():
        augmented['values']['train'][field].extend(value for value in values
                                                   if value not in augmented['values']['train'][field])
    augmented_transfer = deepcopy(expanded)
    return examples_for(augmented, augmented_transfer), augmented


def fit(model, rows, feature_reader, no_write, epochs):
    empty = [(0, 0)] * len(FIELDS)
    examples = [example for row in rows for example in (row['first'], row['update'])]
    examples.extend((row['question'], empty) for row in rows[::2])
    examples.extend((text, empty) for text in no_write for _ in range(12))
    texts, boundaries = zip(*examples, strict=True)
    codes = characters(texts)
    features = feature_reader(texts)
    targets = torch.tensor(boundaries, dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005)
    trace = []
    for epoch in range(epochs):
        losses = []
        for batch in torch.randperm(len(examples)).split(64):
            logits = model(codes[batch], features[batch]).flatten(end_dim=2)
            truth = targets[batch].flatten()
            loss = F.cross_entropy(logits, truth, reduction='none')
            loss = (loss * torch.where(truth > 0, 2.5, 1.)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        item = {'epoch': epoch + 1, 'loss': sum(losses) / len(losses)}
        trace.append(item)
        if epoch in (0, epochs - 1) or (epoch + 1) % 5 == 0:
            print('Compositional writer', item, flush=True)
    return trace, len(examples)


class FeatureLocator:
    def __init__(self, writer, features, device):
        self.writer, self.features, self.device = writer, features, device

    def __call__(self, text):
        return self.writer.write(text, MARGIN, features=self.features([text]).to(self.device))


def evaluate(writer, features, device, tokenizer, holdout):
    locator = FeatureLocator(writer, features, device)
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
        HOLDOUT, WRITER_DATA, TRANSFER, args.checkpoint, args.database, args.writer,
    )}
    holdout = json.loads(HOLDOUT.read_text(encoding='utf-8'))
    data = json.loads(WRITER_DATA.read_text(encoding='utf-8'))
    transfer = json.loads(TRANSFER.read_text(encoding='utf-8'))
    rows, augmented = training_rows(data, transfer)
    training_texts = {text for row in rows for text in (row['first'][0], row['update'][0], row['question'])}
    holdout_texts = {text for row in holdout for text in (row['first'], row['update'], row['question'])}
    if training_texts & holdout_texts:
        raise ValueError('A holdout question occurs in compositional training')
    payload = torch.load(args.writer, map_location='cpu', weights_only=True)
    if (payload['format'], payload['parent_sha256'], payload['fields'], payload['feature_depth'],
        payload['writer_width']) != ('spora-working-turn-experiment-v1', sources[args.checkpoint],
                                     FIELDS, FEATURE_DEPTH, WRITER_WIDTH):
        raise ValueError('Writer checkpoint is incompatible with the parent')
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
        writer = TurnValueWriter(WRITER_WIDTH, feature_width=payload['hidden'])
        writer.load_state_dict(payload['writer'])
        trace, count = fit(writer, rows, features, augmented['no_write'], args.epochs)
        writer.to(device).eval()
        results = evaluate(writer, features, device, tokenizer, holdout)
        checkpoint = args.output / 'writer.pt'
        atomic_save({**payload, 'writer': {key: value.cpu() for key, value in writer.state_dict().items()}}, checkpoint)
        if digest != frozen_digest(runtime.organism, runtime.state):
            raise ValueError('Compositional writer changed the durable individual')
    if any(sha(path) != value for path, value in sources.items()):
        raise ValueError('Compositional writer changed an input')
    result = {
        'format': 'spora-compositional-writer-v1', 'scores': {
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
    parser.add_argument('--epochs', type=int, default=5)
    options = parser.parse_args()
    if options.epochs < 1:
        parser.error('--epochs must be positive')
    main(options)
