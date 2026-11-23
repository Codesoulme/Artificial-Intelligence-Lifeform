"""Train character BIO tags on frozen SPORA states, with separate read intent."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

from scripts.audit_qwen_compatible_natural_memory_20261121 import MODEL, NAMES, extract, sha
from scripts.audit_registered_name_spans import CALIBRATION_NAMES, TRAIN_NAMES, metrics
from scripts.audit_registered_name_teacher import bio_transitions, person_spans, viterbi
from scripts.deepseek_cluener_locator_train import TRAIN, TRAIN_SHA, VALIDATION, VALIDATION_SHA, remove_validation_texts
from scripts.deepseek_cluener_span_audit import labeled_names
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_qwen_character_name_span_20261122 import INDICES, SINGLE, character_inputs
from spora.integrated_checkpoint import load_integrated
from spora.language_training import atomic_save
from transformers import AutoTokenizer


SEED = 20261124
LABELS = ('O', 'B-PER', 'I-PER')
PREFIX = '找出文字中提及的每个人名，包括没有登记的人。\n文字：'
EXPANDED_TEMPLATES = (
    '盒子里放着几份材料，麻烦帮我核对{name}报上来的{field}。',
    '你读一读{name}填的东西。我只想知道{field}那一行。',
    '名单上有这么些人，{name}交的那张写了什么{field}？',
    '上两天转来的登记材料提到，{name}当时挑了什么{field}？',
    '帮我对一下归在{name}名下的{field}。原先那份先不用。',
    '照这张表，请为{name}安排合适的{field}。可以告诉我填的是什么吗？',
    '{name}那一栏现在是什么？我指的是关于{field}的格子。',
    '我一时记不起来。改动前那份文书写的是{name}订的什么{field}？',
    '别再沿用我昨天说的那个了。请查看{name}点了什么{field}。',
    '请问{name}给自己留的{field}是什么样的？',
    '保留登记原样，只抄出{name}所选{field}的字样。',
    '左边那张单子标着{name}，当时记在表上的{field}是哪项？',
    '翻一下落款为{name}的表格，想知道填的{field}是什么。',
    '能说说刚收到的材料吗？我问的是{name}用到的{field}。',
)


def make_example(template, names, *, field='饮品'):
    question = template.format(name=names[0], other=names[-1], field=field)
    spans = []
    for name in names:
        if question.count(name) != 1:
            raise ValueError('Expected one occurrence of each generated name')
        start = question.index(name)
        spans.append((start, start + len(name)))
    return question, sorted(spans)


def synthetic(names):
    rows = []
    for index, name in enumerate(names):
        other = names[(index + 1) % len(names)]
        for j, template in enumerate(SINGLE):
            field = ('饮品', '茶水', '饮料')[j % 3]
            rows.append(make_example(template, (name,), field=field))
            variant = name + ('峥', '珩', '洵')[index % 3] if j % 2 else ('方', '贺', '江')[j % 3] + name
            rows.append(make_example(template, (variant,), field=field))
        rows.append(make_example('把{name}和{other}登记的饮料分别列出来。', (name, other)))
        rows.append(make_example('不要碰{name}的记录，陪我聊一会儿。', (name,)))
    return rows


def expanded_examples(names):
    fields = ('热饮', '寄件地址', '清单颜色', '会议位置', '取件方式', '配套餐食')
    for index, name in enumerate(names):
        other = names[(index + 3) % len(names)]
        for j, template in enumerate(EXPANDED_TEMPLATES):
            field = fields[(index + j) % len(fields)]
            yield make_example(template, (name,), field=field)
            variant = (name + ('卓', '荞', '岑', '玥', '勉', '倩')[j % 6]
                       if (j + index) % 2 else ('袁', '廖', '王', '温', '吕', '罗')[j % 6] + name)
            yield make_example(template, (variant,), field=field)
        yield make_example('我已经听过{other}的了，{name}当时定的{field}是什么？', (name, other), field='饮料')
        yield make_example('请分别念一遍{name}与{other}的登记内容。', (name, other))
        yield make_example('帮我写封信给{name}，另外别改{other}填的资料。', (name, other))


def news(excluded, *, expanded=False):
    from pyarrow import parquet

    if sha(TRAIN) != TRAIN_SHA or sha(VALIDATION) != VALIDATION_SHA:
        raise ValueError('CLUENER source digest mismatch')
    rows = remove_validation_texts(parquet.read_table(TRAIN).to_pylist(), parquet.read_table(VALIDATION).to_pylist())
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        text = row['text']
        if not 8 <= len(text) <= 54 or text in seen or any(name in text for name in excluded):
            continue
        spans = labeled_names(row)
        if any(first[1] > second[0] for first, second in zip(spans, spans[1:])):
            continue
        seen.add(text)
        groups[min(len(spans), 2)].append((text, spans))
    limits = (80, 320, 130) if expanded else (48, 160, 72)
    for group, limit in enumerate(limits):
        ordered = sorted(groups[group], key=lambda row: hashlib.sha256(row[0].encode()).digest())
        yield from ordered[:limit]


class CharacterNameTagger(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.project = nn.Linear(width, 64)
        self.context = nn.Conv1d(64, 64, 5, padding=2)
        self.sequence = nn.GRU(64, 32, batch_first=True, bidirectional=True)
        self.classify = nn.Linear(64, len(LABELS))

    def forward(self, vectors, mask):
        hidden = F.gelu(self.project(vectors.float())).masked_fill(~mask[..., None], 0.)
        local = F.gelu(self.context(hidden.transpose(1, 2)).transpose(1, 2))
        features = (hidden + local).masked_fill(~mask[..., None], 0.)
        packed = nn.utils.rnn.pack_padded_sequence(features, mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
        recurrent, _ = self.sequence(packed)
        recurrent, _ = nn.utils.rnn.pad_packed_sequence(recurrent, batch_first=True, total_length=features.shape[1])
        return self.classify(features + recurrent)


def encode(decoder, tokenizer, rows):
    batches = []
    for offset in range(0, len(rows), 24):
        chunk = rows[offset:offset + 24]
        questions = [question for question, _ in chunk]
        hidden, _ = extract(decoder.base, tokenizer, questions, batch_size=2, hidden_indices=INDICES,
                            user_prefix=PREFIX, last_logit_only=True)
        vectors, mask = character_inputs(tokenizer, questions, hidden, PREFIX)
        labels = torch.zeros(mask.shape, dtype=torch.long).masked_fill(~mask, -100)
        for index, (_, spans) in enumerate(chunk):
            for start, end in spans:
                labels[index, start] = 1
                labels[index, start + 1:end] = 2
        batches.append((vectors, mask, labels))
        if (offset // 24 + 1) % 4 == 0:
            print('features', min(offset + 24, len(rows)), '/', len(rows), flush=True)
    return batches


def train(batches, *, name_weight):
    head = CharacterNameTagger(batches[0][0].shape[-1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.02)
    class_weights = torch.tensor([1., name_weight, name_weight])
    generator = random.Random(SEED)
    order = list(range(len(batches)))
    for epoch in range(20):
        generator.shuffle(order)
        for index in order:
            vectors, mask, labels = batches[index]
            optimizer.zero_grad(set_to_none=True)
            logits = head(vectors, mask)
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), weight=class_weights)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite name-tag loss')
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            print('epoch', epoch + 1, 'last loss', float(loss), flush=True)
    return head.eval()


@torch.inference_mode()
def predict(head, batches):
    transitions, first = bio_transitions(LABELS)
    for vectors, mask, _ in batches:
        for scores, allowed in zip(head(vectors, mask), mask, strict=True):
            scores = scores[allowed]
            path = viterbi(scores, transitions, first)
            yield person_spans([(i, i + 1) for i in range(len(path))], path, LABELS)


def main(checkpoint, baseline, output, expanded=False):
    if output.exists():
        raise FileExistsError(output)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    checkpoint_sha, baseline_sha = sha(checkpoint), sha(baseline)
    source = json.loads(baseline.read_text(encoding='utf-8'))
    if source['contextual']['checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Evaluation must use the audited decoder')
    excluded = (*NAMES, '小芮', '阿奕', *CALIBRATION_NAMES)
    training = [*synthetic(TRAIN_NAMES), *news(excluded, expanded=expanded)]
    if expanded:
        # Small names and three-character names appear with interchangeable grammatical context.
        extra = ('秦维恩', '崔岚宁', '魏晗', '小恬', '阿郑', '朱少昂', '苏米', '阿琤')
        training.extend(expanded_examples(TRAIN_NAMES + extra))
    if any(name in text[start:end] for text, spans in training for start, end in spans for name in excluded):
        raise ValueError('Training spans contain a reserved test name')
    random.Random(SEED).shuffle(training)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    decoder, _, _ = load_integrated(checkpoint, load_full_streamed(), MODEL)
    name_weight = 2.25 if expanded else 1.
    head = train(encode(decoder, tokenizer, training), name_weight=name_weight)
    groups = source['groups']
    rows = [row for data in groups.values() for row in data['rows']]
    batches = encode(decoder, tokenizer, [(row['question'], ()) for row in rows])
    for row, spans in zip(rows, predict(head, batches), strict=True):
        row['tagged_spans'] = spans
        candidate = row['candidates']
        accepted = (len(spans) == len(candidate) == 1 and spans[0] == candidate[0][:2]
                    and row.get('temporal_mode', 3) != 3)
        row['bio_read'] = row['unique'] if accepted else None
    summary = {group: metrics(data['rows'], 'bio_read') for group, data in groups.items()}
    for group, result in summary.items():
        print(group, result, flush=True)
    if sha(checkpoint) != checkpoint_sha or sha(baseline) != baseline_sha:
        raise ValueError('Experiment inputs changed')
    output.mkdir(parents=True, exist_ok=False)
    weights = output / 'tagger.pt'
    atomic_save({'format': 'spora-research-name-bio-v1', 'checkpoint_sha256': checkpoint_sha,
                 'prefix': PREFIX, 'hidden_indices': list(INDICES), 'state': head.state_dict()}, weights)
    report = {'checkpoint_sha256': checkpoint_sha, 'baseline_sha256': baseline_sha,
              'tagger_sha256': sha(weights), 'seed': SEED, 'training_questions': len(training),
              'expanded': expanded, 'person_class_weight': name_weight,
              'cluener_train_sha256': TRAIN_SHA, 'cluener_validation_sha256': VALIDATION_SHA,
              'summary': summary, 'groups': groups}
    with (output / 'result.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'baseline', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expanded', action='store_true')
    main(**vars(parser.parse_args()))
