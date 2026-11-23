"""Train the SPORA name-span head with an independent Chinese NER teacher."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from scripts.audit_qwen_compatible_natural_memory_20261121 import (
    MODEL, NAMES, extract, fresh_cases, fresh_version_cases, load_head, sha,
)
from scripts.audit_qwen_durable_entity_reads_20261122 import PEOPLE, QUESTIONS
from scripts.audit_registered_name_spans import (
    CALIBRATION_NAMES, CALIBRATION_TEMPLATES, TRAIN_NAMES, learning_rows, registry,
)
from scripts.audit_registered_name_teacher import (
    REVISION, TEACHER, WEIGHTS_SHA, annotate,
)
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_qwen_character_name_span_20261122 import (
    INDICES, SINGLE, USER_PREFIX, character_inputs, fit, news_replay, select,
)
from spora.integrated_checkpoint import load_integrated
from spora.language_training import atomic_save
from spora.named_address import feature_files_sha256


SEED = 20261124


def synthetic_rows(names):
    rows = []
    for question, key in learning_rows(names, SINGLE):
        rows.append((question, None if key is None else names[key]))
    return rows


def teacher_labels(model, tokenizer, rows):
    predictions = list(annotate(model, tokenizer, [question for question, _ in rows]))
    labeled = []
    for (question, intended), prediction in zip(rows, predictions, strict=True):
        spans = {question[start:end] for start, end in prediction['bio']}
        if intended is not None and question.count(intended) == 1 and intended in spans:
            labeled.append((question, intended))
        else:
            labeled.append((question, None))
    return labeled


def evaluate(model, tokenizer, decoder, rows, margin):
    questions = [question for question, _ in rows]
    features, _ = extract(decoder.base, tokenizer, questions, batch_size=2,
                          hidden_indices=INDICES, user_prefix=USER_PREFIX, last_logit_only=True)
    return predictions(rows, select(model, *character_inputs(tokenizer, questions, features, USER_PREFIX), margin))


def predictions(rows, selected):
    result = []
    for (question, expected), (span, confidence) in zip(rows, selected, strict=True):
        actual = None if span is None else question[slice(*span)]
        result.append({'question': question, 'expected': expected, 'selected': actual,
                       'span': span, 'margin': confidence})
    return result


def exact_counts(rows):
    return {'total': len(rows), 'positive': sum(row['expected'] is not None for row in rows),
            'exact': sum(row['selected'] == row['expected'] for row in rows),
            'false_links': sum(row['selected'] is not None and row['selected'] != row['expected']
                               for row in rows)}


def main(checkpoint, temporal, output, teacher):
    if output.exists():
        raise FileExistsError(output)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    checkpoint_sha = sha(checkpoint)
    temporal_payload, temporal_sha = load_head(temporal, 'temporal.pt', 'temporal_sha256', checkpoint_sha)
    if temporal_payload['hidden_indices'] != list(INDICES):
        raise ValueError('Temporal and name heads must use matching hidden indices')
    teacher_weights_sha = sha(teacher / 'model.safetensors')
    if teacher_weights_sha != WEIGHTS_SHA:
        raise ValueError('Pinned teacher weights are incomplete or changed')

    tokenizer_teacher = AutoTokenizer.from_pretrained(teacher, local_files_only=True, trust_remote_code=False,
                                                       use_fast=True)
    teacher_model = AutoModelForTokenClassification.from_pretrained(
        teacher, local_files_only=True, use_safetensors=True, trust_remote_code=False).eval()
    replay, sources = news_replay()
    train_source = synthetic_rows(TRAIN_NAMES)
    teacher_rows = teacher_labels(teacher_model, tokenizer_teacher, train_source)
    # A missed gold span is ambiguous supervision. Keep only teacher-agreed synthetic examples.
    train_rows = [gold for gold, predicted in zip(train_source, teacher_rows, strict=True) if gold == predicted]
    train_rows.extend(replay)
    calibration_rows = [(question, None if key is None else CALIBRATION_NAMES[key])
                        for question, key in learning_rows(CALIBRATION_NAMES, CALIBRATION_TEMPLATES)]
    print('supervised questions', len(train_rows), 'calibration', len(calibration_rows), flush=True)
    del teacher_model, tokenizer_teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not any(name is not None for _, name in train_rows):
        raise ValueError('Teacher produced no positive training labels')

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    decoder, _, _ = load_integrated(checkpoint, load_full_streamed(), MODEL)
    decoder.eval().requires_grad_(False)
    train_questions = [question for question, _ in train_rows]
    train_features, _ = extract(decoder.base, tokenizer, train_questions, batch_size=1,
                                hidden_indices=INDICES, user_prefix=USER_PREFIX, last_logit_only=True)
    vectors, mask = character_inputs(tokenizer, train_questions, train_features, USER_PREFIX)
    feature_width = vectors.shape[-1] - 4
    del train_features
    head = fit(vectors, mask, train_rows).eval()
    calibration_questions = [question for question, _ in calibration_rows]
    calibration_features, _ = extract(decoder.base, tokenizer, calibration_questions, batch_size=2,
                                      hidden_indices=INDICES, user_prefix=USER_PREFIX, last_logit_only=True)
    calibration_scores = select(head, *character_inputs(tokenizer, calibration_questions,
                                                        calibration_features, USER_PREFIX), 0.)
    negative_scores = [score for (span, score), (_, expected) in zip(calibration_scores, calibration_rows, strict=True)
                       if expected is None and span is not None]
    margin = max([0., *negative_scores])
    data = {}
    data['original'] = [(question, None if index is None else NAMES[index])
                        for question, index, _ in fresh_cases()]
    data['fresh_v2'] = [(question, None if index is None else NAMES[index])
                        for question, index, _ in fresh_version_cases()]
    names_by_key = {key: name for name, key in PEOPLE}
    data['durable_fixture'] = [(question, names_by_key.get(key)) for question, key, _ in QUESTIONS]
    results = {'train': predictions(train_rows, select(head, vectors, mask, margin)),
               'calibration': predictions(calibration_rows, [(span if score > margin else None, score)
                                                             for span, score in calibration_scores])}
    del vectors, mask
    for name, rows in data.items():
        results[name] = evaluate(head, tokenizer, decoder, rows, margin)
    for name, rows in results.items():
        print(name, exact_counts(rows), flush=True)
    if sha(checkpoint) != checkpoint_sha or sha(temporal / 'temporal.pt') != temporal_sha:
        raise ValueError('Source changed during teacher-guided training')
    output.mkdir(parents=True, exist_ok=False)
    weights = output / 'router.pt'
    atomic_save({'format': 'spora-character-name-span-v1', 'checkpoint_sha256': checkpoint_sha,
                 'base_sha256': sha(MODEL / 'model.safetensors'),
                 'feature_config_sha256': feature_files_sha256(MODEL),
                 'hidden_indices': list(INDICES), 'user_prefix': USER_PREFIX,
                 'feature_width': feature_width, 'margin': margin,
                 'state': head.state_dict()}, weights)
    report = {'router_sha256': sha(weights), 'checkpoint_sha256': checkpoint_sha,
              'temporal_sha256': temporal_sha, 'teacher': TEACHER, 'teacher_revision': REVISION,
              'teacher_weights_sha256': teacher_weights_sha, 'seed': SEED, 'margin': margin,
              'train_source_count': len(train_source), 'train_positive_count': sum(name is not None for _, name in train_rows),
              'train_count': len(train_rows), 'calibration_source_count': len(calibration_rows),
              'sources': {str(path): digest for path, digest in sources.items()},
              'results': {name: {'summary': exact_counts(rows), 'rows': rows} for name, rows in results.items()}}
    (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'temporal', 'output', 'teacher'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(**vars(parser.parse_args()))
