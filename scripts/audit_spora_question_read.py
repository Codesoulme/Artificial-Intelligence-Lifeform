"""Condition bounded working-state reads on the natural-language question."""

import argparse
from itertools import permutations
import json
from pathlib import Path
import random
from string import Formatter
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer

from audit_spora_dialogue_working_state import (FEATURE_DEPTH, FORMAT, FrozenTurnFeatures, SemanticLocator,
                                                rollout_states, state_from, verify_date_isolation)
from audit_spora_ordered_working_state import POINTERS, WIDTH, answer_actions, generate, state_tokens
from audit_spora_writer_transfer import TRANSFER, new_cases, original_cases
from scripts.audit_spora_native_query_key import write
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_dialogue_working_state import DATA, SEED, WRITER_WIDTH, dialogue_data
from spora.integrated_durable import load_integrated_durable
from spora.language_training import atomic_save, frozen_digest
from spora.research_dialogue_working_state import FIELDS, TurnValueWriter, WorkingTurnReader


FORMAT_ID = 'spora-question-conditioned-read-v1'
FIELD_WORDS = {
    'day': ('日期', '星期', '日子'), 'time': ('时间', '钟点', '时刻'),
    'place': ('地点', '场地', '地址', '集合点'), 'gear': ('装备', '物品', '携带物'),
    'person': ('收货人', '收件人', '姓名'), 'quantity': ('件数', '数量', '套数'),
    'direction': ('方向', '行进方向', '路线'),
}
LABELS = {
    'appointment': {'day': '日期', 'time': '时间', 'place': '地点'},
    'parcel': {'person': '收货人', 'quantity': '件数', 'place': '地址'},
    'route': {'place': '集合点', 'gear': '装备', 'direction': '方向'},
}
STYLE = {
    'slash': ('按顺序写出{fields}，用斜杠隔开。', '请用 / 分隔，顺次给出{fields}。'),
    'lines': ('将{fields}各写一行，带字段标签。', '请按{fields}逐行列出。'),
    'comma': ('回答{fields}，用逗号分隔。', '把{fields}按序写好，用逗号隔开。'),
    'only': ('请只给{fields}。', '只写{fields}这一项。'),
}
EXTRA_STYLE = {
    'slash': ('请依次列出{fields}，每项之间加斜线。', '汇报顺序是{fields}，请使用斜杠。'),
    'lines': ('分行给出{fields}，每行先写项目名。', '按{fields}的顺序，每项独占一行。'),
    'comma': ('依序报{fields}，以逗号隔开各项。', '按{fields}这个次序回复，中间加逗号。'),
    'only': ('这一轮只回复{fields}就行。', '单独报一下{fields}，不用其它项目。'),
}


class QuestionRead(nn.Module):
    def __init__(self, hidden, glue_size, *, ordered=False):
        super().__init__()
        self.start = POINTERS + glue_size
        self.actions = nn.Embedding(self.start + 1, WIDTH)
        self.context = nn.Linear(POINTERS, WIDTH)
        self.question = nn.Linear(hidden, WIDTH, bias=False)
        self.question_order = nn.GRU(WIDTH, WIDTH, batch_first=True) if ordered else None
        self.recurrent = nn.GRU(3 * WIDTH, WIDTH, batch_first=True)
        self.output = nn.Linear(WIDTH, self.start)

    def forward(self, previous, valid, question, mask):
        query = self.question(F.layer_norm(question.float(), (question.shape[-1],)))
        lengths = mask.sum(1).clamp_min(1)
        if self.question_order is not None:
            encoded, _ = self.question_order(query)
            query = encoded[torch.arange(query.shape[0], device=query.device), lengths - 1]
        else:
            query = (query * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(-1)
        slot = self.context(valid[:, :POINTERS].float())
        context = torch.cat((slot, query), dim=-1).unsqueeze(1)
        inputs = torch.cat((self.actions(previous), context.expand(-1, previous.shape[1], -1)), dim=-1)
        hidden, _ = self.recurrent(inputs)
        return self.output(hidden).masked_fill(~valid[:, None, :], -torch.inf)


class BoundQuestion:
    def __init__(self, module, question, mask):
        self.module = module
        self.start = module.start
        self.question = question
        self.mask = mask

    def __call__(self, previous, valid):
        return self.module(previous, valid, self.question, self.mask)


def patterns_for(kind, rng, ordered=False):
    fields = tuple(LABELS[kind])
    styles = {name: values + EXTRA_STYLE[name] for name, values in STYLE.items()} if ordered else STYLE
    for field in fields:
        for phrasing in styles['only']:
            yield phrasing.format(fields=rng.choice(FIELD_WORDS[field])), '{' + field + '}'
    for count in (2, 3):
        for order in permutations(fields, count):
            names = '、'.join(rng.choice(FIELD_WORDS[field]) for field in order)
            args = ['{' + field + '}' for field in order]
            lines = [LABELS[kind][field] + '：' + arg for field, arg in zip(order, args, strict=True)]
            for style, out in (('slash', ' / '.join(args)), ('comma', '，'.join(args)), ('lines', '\n'.join(lines))):
                for phrasing in styles[style]:
                    yield phrasing.format(fields=names), out
    if ordered and kind == 'appointment':
        yield '把日期和时刻合在一项，与场地用斜杠分隔，地点放在前面。', '{place} / {day}{time}'
        yield '场地写在最前面，后面接上日期时间，用逗号隔开。', '{place}，{day}{time}'


def training_rows(data, ordered=False):
    rng = random.Random(SEED)
    states = dialogue_data(data, held=False, count=32)
    rows = []
    for kind in LABELS:
        pool = [row for row in states if row['kind'] == kind]
        for _ in range(3):
            for question, template in patterns_for(kind, rng, ordered):
                for values in rng.sample(pool, 4):
                    rows.append((question, template, values['current']))
        canonical = data['kinds'][kind]['answer']
        for question in data['kinds'][kind]['train_question']:
            rows.extend((question, canonical, example['current']) for example in pool * (4 if ordered else 1))
    return rows


def encoded_questions(features, questions, device):
    encoded = features(questions).to(device)
    lengths = torch.tensor([len(question) for question in questions], device=device)
    mask = torch.arange(encoded.shape[1], device=device)[None, :] < lengths[:, None]
    return encoded, mask


def fit(model, rows, tokenizer, glue, features, epochs, device):
    questions = sorted({question for question, _, _ in rows})
    index = {question: i for i, question in enumerate(questions)}
    query, mask = encoded_questions(features, questions, device)
    actions, valid, groups = [], [], []
    for question, template, fields in rows:
        actions.append(answer_actions(template, fields, tokenizer, glue))
        valid.append(state_tokens(state_from(fields, tokenizer), glue, 'cpu').ge(0))
        groups.append(index[question])
    groups = torch.tensor(groups, device=device)
    valid = torch.stack(valid).to(device)
    targets = torch.full((len(rows), max(map(len, actions))), -100, device=device, dtype=torch.long)
    for i, sequence in enumerate(actions):
        targets[i, :len(sequence)] = torch.tensor(sequence, device=device)
    previous = F.pad(targets[:, :-1], (1, 0), value=model.start)
    previous = previous.masked_fill(previous.lt(0), model.start)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    trace = []
    for epoch in range(epochs):
        losses = []
        for batch in torch.randperm(len(rows), device=device).split(64):
            positions = groups[batch]
            logits = model(previous[batch], valid[batch], query[positions], mask[positions])
            loss = F.cross_entropy(logits.flatten(0, 1), targets[batch].flatten())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        trace.append(sum(losses) / len(losses))
        if epoch in (0, epochs - 1) or (epoch + 1) % 5 == 0:
            print('Question reader', epoch + 1, trace[-1], flush=True)
    return trace


def audit(cases, model, features, decoder, tokenizer, locator, glue):
    questions = [case['turns'][-1] for case in cases]
    query, mask = encoded_questions(features, questions, decoder.base.device)
    rows = []
    for index, case in enumerate(cases):
        reader = BoundQuestion(model, query[index:index + 1], mask[index:index + 1])
        predicted, writes = rollout_states(locator, tokenizer, case['turns'])
        states = {'oracle': state_from(case['current'], tokenizer), 'automatic': predicted}
        row = {'kind': case['kind'], 'question': questions[index], 'writes': writes, 'expected': case['requested']}
        for name, state in states.items():
            answer = generate(decoder, tokenizer, questions[index], state, reader, glue, 16., .95)
            row[name] = {**answer, 'exact': answer['ended'] and answer['answer'] == case['requested']}
        rows.append(row)
        print(case['kind'], {name: row[name]['exact'] for name in states}, flush=True)
    return rows


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    sources = {path: sha(path) for path in (DATA, TRANSFER, args.checkpoint, args.database, args.writer)}
    data = json.loads(DATA.read_text(encoding='utf-8'))
    transfer = json.loads(TRANSFER.read_text(encoding='utf-8'))
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    payload = torch.load(args.writer, map_location='cpu', weights_only=True)
    binding = (FORMAT, sources[args.checkpoint], FIELDS, FEATURE_DEPTH, WRITER_WIDTH)
    if tuple(payload[key] for key in ('format', 'parent_sha256', 'fields', 'feature_depth', 'writer_width')) != binding:
        raise ValueError('Incompatible writer checkpoint')
    training = training_rows(data, args.ordered)
    test = {'novel': new_cases(transfer, data), 'legacy': original_cases(data)}
    if {question for question, _, _ in training} & {case['turns'][-1] for rows in test.values() for case in rows}:
        raise ValueError('A test question occurs in training')
    glue = sorted({tokenizer.eos_token_id} | {
        token for _, template, _ in training for literal, _, _, _ in Formatter().parse(template)
        for token in tokenizer.encode(literal, add_special_tokens=False)})
    args.output.mkdir(parents=True, exist_ok=False)
    with load_integrated_durable(args.database, args.checkpoint, load_full_streamed(), MODEL) as runtime:
        decoder = runtime.decoder.eval().requires_grad_(False)
        digest = frozen_digest(runtime.organism, runtime.state)
        features = FrozenTurnFeatures(decoder, tokenizer)
        device = decoder.base.device
        writer = TurnValueWriter(WRITER_WIDTH, feature_width=payload['hidden'])
        writer.load_state_dict(payload['writer'])
        locator = SemanticLocator(writer.to(device).eval(), features)
        model = QuestionRead(payload['hidden'], len(glue), ordered=args.ordered).to(device)
        trace = fit(model, training, tokenizer, glue, features, args.epochs, device)
        model.eval()
        file = args.output / 'reader.pt'
        atomic_save({'format': FORMAT_ID, 'parent_sha256': sources[args.checkpoint], 'glue': glue,
                     'ordered': args.ordered,
                     'model': {key: value.cpu() for key, value in model.state_dict().items()}}, file)
        torch.cuda.empty_cache()
        results = {name: audit(rows, model, features, decoder, tokenizer, locator, glue) for name, rows in test.items()}
        dates = verify_date_isolation(runtime, WorkingTurnReader(payload['hidden']).to(device), tokenizer,
                                      state_from(test['legacy'][0]['current'], tokenizer))
        if frozen_digest(runtime.organism, runtime.state) != digest:
            raise ValueError('Question read changed the individual')
    if any(sha(path) != value for path, value in sources.items()):
        raise ValueError('A probe input changed')
    scores = {group: {mode: sum(row[mode]['exact'] for row in rows) for mode in ('oracle', 'automatic')}
              for group, rows in results.items()}
    write(args.output / 'result.json', {'scores': scores, 'cases': results, 'date': dates, 'digest': digest,
                                        'train_rows': len(training), 'trace': trace, 'reader_sha256': sha(file),
                                        'inputs': {str(path): value for path, value in sources.items()},
                                        'seconds': time.perf_counter() - started,
                                        'peak_bytes': torch.cuda.max_memory_allocated()})
    print(scores, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'database', 'writer', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--ordered', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    main(args)
