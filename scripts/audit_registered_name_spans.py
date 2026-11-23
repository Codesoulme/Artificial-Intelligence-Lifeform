"""Measure fingerprint candidate recall and a name independent boundary verifier."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.audit_qwen_compatible_natural_memory_20261121 import NAMES, fresh_cases, fresh_version_cases
from scripts.audit_qwen_durable_entity_reads_20261122 import PEOPLE, QUESTIONS
from scripts.train_qwen_character_name_span_20261122 import SINGLE
from spora.identity_fingerprints import IdentityFingerprints


TRAIN_NAMES = ('小雯', '阿秋', '小宸', '阿鹭', '任青禾', '江若烟', '叶佳敏', '刘平', '霍焱', '宋季夏', '蓉蓉', '周简宁')
CALIBRATION_NAMES = ('杨沂', '阿鹏飞', '小蓓', '严岑萱')
CALIBRATION_TEMPLATES = (
    '劳驾复核{name}先前挑的那款冷饮。', '依{name}现行的那页，该端上什么饮料？',
    '请把临时更正前，给{name}那杯饮品的名字报一下。',
)
HELD_NEGATIVES = (
    '打开大{known}现行的登记，看他点了哪种饮品。',
    '登记在{known}珂名下的饮料是什么？',
    '{known}和沈香各登记了什么饮品？',
    '给令嘉和{known}各准备一杯吧，他们分别挑的是什么？',
    '我只问别人，别动{known}那份饮品记录。',
    '引号里的例句是“{known}选了什么饮品”，这句话怎么译？',
    '贺子言现在登记的是哪种饮品？',
)


def registry(names):
    names = tuple(names)
    result = IdentityFingerprints(len(names))
    for key, name in enumerate(names):
        result.bind(key, name)
    return result


def learning_rows(names, templates):
    rows = []
    for key, name in enumerate(names):
        other = names[(key + 1) % len(names)]
        for index, template in enumerate(templates):
            field = ('饮品', '饮料', '热饮')[index % 3]
            rows.append((template.format(name=name, field=field), key))
            for variant in (name + ('珉', '宣', '雁')[index % 3], '魏' + name, name + '和顾卓'):
                rows.append((template.format(name=variant, field=field), None))
        rows.extend(((f'说出{name}和{other}各自最新的饮料。', None),
                     (f'不用读{name}的饮料记录，写一句邀请。', None),
                     (f'沈瑾登记的饮品是什么？给{name}也问声好。', None),
                     (f'把{name}当天登记的饮品这一句译成英文。', None)))
    return rows


def candidates(index, rows):
    return [index.matching_spans(question, max_name_chars=12) for question, _ in rows]


def features(question, hit):
    start, end, _ = hit
    left, right = '^' + question[:start], question[end:] + '$'
    masked = left + '□' + right
    counts = Counter('L:' + left[-n:] for n in range(1, min(6, len(left)) + 1))
    counts.update('R:' + right[:n] for n in range(1, min(6, len(right)) + 1))
    counts.update(f'J:{left[-n:]}|{right[:m]}' for n in range(1, min(3, len(left)) + 1)
                  for m in range(1, min(3, len(right)) + 1))
    counts.update('Q:' + masked[i:i + n] for n in (2, 3, 4) for i in range(len(masked) - n + 1))
    return counts


def examples(rows, hits):
    for (question, target), spans in zip(rows, hits, strict=True):
        if len(spans) == 1:
            yield features(question, spans[0]), float(target == spans[0][2])


def vectorize(rows, vocabulary):
    vectors = torch.zeros(len(rows), len(vocabulary))
    for index, counts in enumerate(rows):
        for token, value in counts.items():
            if token in vocabulary:
                vectors[index, vocabulary[token]] = value
    return vectors


def fit(rows, hits):
    encoded, labels = zip(*examples(rows, hits), strict=True)
    vocabulary = {token: column for column, token in enumerate(sorted(set().union(*encoded)))}
    matrix = vectorize(encoded, vocabulary)
    targets = torch.tensor(labels)
    weight = torch.zeros(matrix.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS((weight, bias), max_iter=100, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(matrix @ weight + bias, targets) + .005 * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return vocabulary, weight.detach(), bias.detach()


def score(rows, hits, model):
    vocabulary, weight, bias = model
    scores = iter((vectorize([vector for vector, _ in examples(rows, hits)], vocabulary) @ weight + bias).tolist())
    return [next(scores) if len(spans) == 1 else None for spans in hits]


def evaluate(index, rows, model, threshold):
    retrieved = candidates(index, rows)
    scores = score(rows, retrieved, model)
    details = []
    for (question, target), spans, logit in zip(rows, retrieved, scores, strict=True):
        first = spans[0][2] if len(spans) == 1 else None
        predicted = first if logit is not None and logit > threshold else None
        details.append({'question': question, 'expected': target, 'candidates': spans,
                        'unique': first, 'verified': predicted, 'score': logit})
    return {'rows': details, 'positive': sum(row['expected'] is not None for row in details),
            'positive_recalled': sum(row['expected'] is not None and any(
                key == row['expected'] for _, _, key in row['candidates']) for row in details),
            **{method: metrics(details, method) for method in ('unique', 'verified')}}


def metrics(rows, method):
    return {'correct': sum(row[method] == row['expected'] for row in rows),
            'false_links': sum(row[method] is not None and row[method] != row['expected'] for row in rows),
            'positive_correct': sum(row[method] is not None and row[method] == row['expected'] for row in rows)}


@torch.inference_mode()
def contextual_scores(reports, checkpoint, router_dir, temporal_dir):
    from transformers import AutoTokenizer
    from scripts.audit_qwen_compatible_natural_memory_20261121 import MODEL, extract, load_head, sha
    from scripts.deepseek_spora_full_stream_loader import load_full_streamed
    from scripts.train_qwen_character_name_span_20261122 import character_inputs
    from spora.integrated_checkpoint import load_integrated
    from spora.research_name_span import CharacterNameSpan
    from spora.research_temporal_intent import temporal_logits, time_selection

    digest = sha(checkpoint)
    router, router_sha = load_head(router_dir, 'router.pt', 'router_sha256', digest)
    temporal, temporal_sha = load_head(temporal_dir, 'temporal.pt', 'temporal_sha256', digest)
    if router['format'] != 'spora-character-name-span-v1':
        raise ValueError('Expected a character span checkpoint')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    decoder, _, _ = load_integrated(checkpoint, load_full_streamed(), MODEL)
    model = CharacterNameSpan(router['feature_width']).eval()
    model.load_state_dict(router['state'], strict=True)
    rows = [row for group in reports.values() for row in group['rows'] if row['unique'] is not None]
    texts = [row['question'] for row in rows]
    if not texts:
        raise ValueError('Expected at least one candidate')
    temporal_hidden, mask = extract(decoder.base, tokenizer, texts, batch_size=2,
                                    hidden_indices=tuple(temporal['hidden_indices']), last_logit_only=True)
    logits = temporal_logits(temporal, temporal_hidden, mask, texts=texts)
    modes = time_selection(logits.softmax(-1), temporal['threshold'],
                           factorized=temporal.get('architecture') == 'factorized').tolist()
    del temporal_hidden, mask
    surrogate = '小雯'
    replaced = []
    for text, row, mode in zip(texts, rows, modes, strict=True):
        start, end, _ = row['candidates'][0]
        replaced.append(text[:start] + surrogate + text[end:])
        row['temporal_mode'] = mode
    for field, questions in (('original_margin', texts), ('surrogate_margin', replaced)):
        for offset in range(0, len(rows), 16):
            batch = questions[offset:offset + 16]
            hidden, _ = extract(decoder.base, tokenizer, batch, batch_size=2,
                                hidden_indices=tuple(router['hidden_indices']), user_prefix=router['user_prefix'],
                                last_logit_only=True)
            vectors, mask = character_inputs(tokenizer, batch, hidden, router['user_prefix'])
            raw = model(vectors, mask)
            width = mask.shape[1]
            for row, scores in zip(rows[offset:offset + 16], raw, strict=True):
                start, end, _ = row['candidates'][0]
                if field == 'surrogate_margin':
                    end = start + len(surrogate)
                position = start * width + end - 1
                score = float(scores[position])
                if not torch.isfinite(scores[position]):
                    row[field] = None
                    continue
                scores[position] = -torch.inf
                row[field] = score - float(scores.max())
    calibration = reports['calibration']['rows']
    thresholds = {}
    for field in ('original_margin', 'surrogate_margin'):
        thresholds[field] = max([-float('inf')] + [row[field] for row in calibration
                                if row['expected'] is None and row.get('temporal_mode', 3) != 3
                                and row.get(field) is not None])
    for group in reports.values():
        for row in group['rows']:
            read_allowed = row.get('temporal_mode', 3) != 3
            row['intent_only'] = row['unique'] if read_allowed else None
            for field, threshold in thresholds.items():
                score = row.get(field)
                row[field + '_read'] = row['unique'] if read_allowed and score is not None and score > threshold else None
        for method in ('intent_only', 'original_margin_read', 'surrogate_margin_read'):
            group[method] = metrics(group['rows'], method)
    if sha(checkpoint) != digest:
        raise ValueError('Checkpoint changed during the audit')
    return {'checkpoint_sha256': digest, 'router_sha256': router_sha, 'temporal_sha256': temporal_sha,
            'thresholds': {key: None if value == -float('inf') else value for key, value in thresholds.items()}}


def main(output, checkpoint=None, router=None, temporal=None):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    training = learning_rows(TRAIN_NAMES, SINGLE)
    model = fit(training, candidates(registry(TRAIN_NAMES), training))
    calibration = learning_rows(CALIBRATION_NAMES, CALIBRATION_TEMPLATES)
    hits = candidates(registry(CALIBRATION_NAMES), calibration)
    logits = score(calibration, hits, model)
    threshold = max(logit for (_, target), logit in zip(calibration, logits, strict=True)
                    if logit is not None and target is None)
    new_names = tuple(name for name, _ in PEOPLE)
    new_keys = {key: i for i, (_, key) in enumerate(PEOPLE)}
    new_rows = [(question, new_keys.get(key)) for question, key, _ in QUESTIONS]
    new_rows += [(template.format(known=new_names[i % len(new_names)]), None)
                 for i, template in enumerate(HELD_NEGATIVES)]
    groups = {'calibration': (registry(CALIBRATION_NAMES), calibration),
              'new_names': (registry(new_names), new_rows),
              'new_questions': (registry(NAMES), [(question, key) for question, key, _ in fresh_cases()]),
              'version_questions': (registry(NAMES), [(question, key) for question, key, _ in fresh_version_cases()])}
    report = {'training_rows': len(training), 'calibration_rows': len(calibration), 'threshold': threshold,
              'groups': {name: evaluate(index, rows, model, threshold) for name, (index, rows) in groups.items()}}
    if checkpoint is not None:
        report['contextual'] = contextual_scores(report['groups'], checkpoint, router, temporal)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    for group, data in report['groups'].items():
        print(group, 'recall', data['positive_recalled'], '/', data['positive'],
              'unique', data['unique'], 'verified', data['verified'], flush=True)
        if checkpoint is not None:
            print('context', {key: data[key] for key in ('intent_only', 'original_margin_read', 'surrogate_margin_read')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--router', type=Path)
    parser.add_argument('--temporal', type=Path)
    args = parser.parse_args()
    supplied = [item is not None for item in (args.checkpoint, args.router, args.temporal)]
    if any(supplied) and not all(supplied):
        parser.error('Contextual audit requires --checkpoint, --router, and --temporal together')
    main(**vars(args))
