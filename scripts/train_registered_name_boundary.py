"""Train explicit left and right boundaries for fingerprint candidates."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from scripts.audit_qwen_compatible_natural_memory_20261121 import MODEL, extract, sha
from scripts.audit_registered_name_spans import CALIBRATION_NAMES, CALIBRATION_TEMPLATES, TRAIN_NAMES, metrics
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_qwen_character_name_span_20261122 import INDICES, SINGLE, USER_PREFIX, character_inputs, news_replay
from spora.integrated_checkpoint import load_integrated
from spora.language_training import atomic_save
from transformers import AutoTokenizer


SEED = 20261123


def synthetic(names, templates):
    for index, name in enumerate(names):
        for j, template in enumerate(templates):
            yield template.format(name=name, field=('饮品', '饮料', '热饮')[j % 3]), name
            variant = (name + ('珉', '宣', '雁')[index % 3] if j % 2 else ('蒋', '成', '谭')[index % 3] + name)
            yield template.format(name=variant, field=('饮品', '饮料', '热饮')[j % 3]), variant


def labeled(rows):
    items = []
    for question, name in rows:
        start = question.index(name)
        end = start + len(name)
        spans = [(start, end)]
        spans.extend((a, b) for a, b in ((start - 1, end), (start, end + 1), (start + 1, end), (start, end - 1))
                     if 0 <= a < b <= len(question))
        items.append((question, spans, [(float(a == start), float(b == end)) for a, b in spans]))
    return items


@torch.no_grad()
def encode(decoder, tokenizer, items):
    features = []
    for offset in range(0, len(items), 24):
        batch = items[offset:offset + 24]
        texts = [text for text, _, _ in batch]
        hidden, _ = extract(decoder.base, tokenizer, texts, batch_size=2, hidden_indices=INDICES,
                            user_prefix=USER_PREFIX, last_logit_only=True)
        vectors, _ = character_inputs(tokenizer, texts, hidden, USER_PREFIX)
        vectors = F.pad(vectors.float(), (0, 0, 1, 1))
        for i, (text, spans, _) in enumerate(batch):
            for start, end in spans:
                if not 0 <= start < end <= len(text):
                    raise ValueError('Candidate outside the original question')
                features.append(vectors[i, [start, start + 1, end, end + 1]])
        if (offset // 24 + 1) % 4 == 0:
            print('boundary features', min(offset + 24, len(items)), '/', len(items), flush=True)
    return torch.stack(features)


class BoundaryHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.project = nn.Linear(width, 48)
        self.classifiers = nn.ModuleList(nn.Sequential(nn.Linear(48 * 3, 32), nn.GELU(), nn.Linear(32, 1))
                                         for _ in range(2))

    def forward(self, features):
        vectors = F.gelu(self.project(features))
        scores = []
        for side, classifier in enumerate(self.classifiers):
            left, right = vectors[:, side * 2], vectors[:, side * 2 + 1]
            scores.append(classifier(torch.cat((left, right, right - left), dim=-1)).squeeze(-1))
        return torch.stack(scores, dim=-1)


def fit(vectors, labels):
    torch.manual_seed(SEED)
    model = BoundaryHead(vectors.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.02)
    positive_weight = (1 - labels).sum(0) / labels.sum(0)
    for _ in range(16):
        for indices in torch.randperm(len(labels)).split(48):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(vectors[indices]), labels[indices],
                                                      pos_weight=positive_weight)
            loss.backward()
            optimizer.step()
    return model.eval()


def main(checkpoint, baseline, output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    baseline_sha = sha(baseline)
    source = json.loads(baseline.read_text(encoding='utf-8'))
    checkpoint_sha = sha(checkpoint)
    if source['contextual']['checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Evaluation must use the audited base and time selector')
    replay, sources = news_replay()
    training = labeled([*synthetic(TRAIN_NAMES, SINGLE), *((text, name) for text, name in replay if name)])
    calibration = labeled(list(synthetic(CALIBRATION_NAMES, CALIBRATION_TEMPLATES)))
    held = {group: data for group, data in source['groups'].items() if group != 'calibration'}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    decoder, _, _ = load_integrated(checkpoint, load_full_streamed(), MODEL)
    vectors = encode(decoder, tokenizer, training)
    labels = torch.tensor([label for _, _, labels in training for label in labels])
    head = fit(vectors, labels)
    del vectors
    with torch.no_grad():
        calibrate = head(encode(decoder, tokenizer, calibration)).amin(1)
        calibration_labels = torch.tensor([all(label) for _, _, labels in calibration for label in labels])
        threshold = float(calibrate[~calibration_labels].max())
        selection = [row for data in held.values() for row in data['rows'] if row['unique'] is not None]
        inputs = [(row['question'], [row['candidates'][0][:2]], None) for row in selection]
        logits = head(encode(decoder, tokenizer, inputs))
    for row, pair in zip(selection, logits.tolist(), strict=True):
        row['boundary_logits'] = pair
    summary = {}
    for group, data in held.items():
        for row in data['rows']:
            pair = row.get('boundary_logits')
            allow = pair is not None and min(pair) > threshold and row.get('temporal_mode', 3) != 3
            row['boundary_read'] = row['unique'] if allow else None
        summary[group] = metrics(data['rows'], 'boundary_read')
        print(group, summary[group], flush=True)
    if sha(baseline) != baseline_sha or sha(checkpoint) != checkpoint_sha or any(
            sha(path) != digest for path, digest in sources.items()):
        raise ValueError('Experiment inputs changed')
    output.mkdir(parents=True, exist_ok=False)
    weights = output / 'boundary.pt'
    atomic_save({'format': 'spora-research-name-boundary-v1', 'checkpoint_sha256': checkpoint_sha,
                 'threshold': threshold, 'state': head.state_dict(), 'hidden_indices': list(INDICES),
                 'user_prefix': USER_PREFIX}, weights)
    report = {'checkpoint_sha256': checkpoint_sha, 'baseline_sha256': baseline_sha,
              'boundary_sha256': sha(weights), 'news_replay': {str(path): digest for path, digest in sources.items()},
              'train_questions': len(training), 'calibration_questions': len(calibration), 'threshold': threshold,
              'calibration_correct_positives': int(((calibrate > threshold) & calibration_labels).sum()),
              'calibration_positives': int(calibration_labels.sum()), 'summary': summary, 'groups': held}
    with (output / 'result.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('checkpoint', 'baseline', 'output'):
        parser.add_argument('--' + argument, required=True, type=Path)
    main(**vars(parser.parse_args()))
