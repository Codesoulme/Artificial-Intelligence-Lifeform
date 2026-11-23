"""Measure a shallow Chinese name encoder on the persistent routing audit."""

import argparse
import copy
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from scripts.audit_qwen_compatible_natural_memory_20261121 import sha
from scripts.audit_registered_name_teacher import REVISION, TEACHER, WEIGHTS_SHA, annotate, compare
from scripts.train_registered_name_bio import (CALIBRATION_NAMES, NAMES, SEED, TRAIN_NAMES,
                                                expanded_examples, make_example, news, synthetic)
from transformers import AutoModelForTokenClassification, AutoTokenizer


TEACHER_LAYERS = (0, 3, 7, 11)


def paired_boundaries(names):
    templates = (
        '去看看那页由{name}填写的登记单。', '资料上写着替{name}选了哪个品种？',
        '柜台收到了{name}的东西，请按单子取。', '交接本里{name}订了哪一份？',
        '对照签字人{name}保留的那次记录。', '帮{name}留出一份，登记的规格是哪种？',
        '请把{name}配好的材料和另一包一起放过来。', '请告诉我{name}选用的款式。',
        '旧表是不是给{name}安排在了早班？', '我想查{name}管的那件东西。',
    )
    suffixes = '越远念文可欢那明青南先怀飞正时又则成来新定采秋方泠'
    for i, name in enumerate(names):
        for j, template in enumerate(templates):
            ending = suffixes[(i * 7 + j * 3) % len(suffixes)]
            yield make_example(template, (name,))
            yield make_example(template, (name + ending,))


def shallow_copy(teacher):
    config = copy.deepcopy(teacher.config)
    config.num_hidden_layers = len(TEACHER_LAYERS)
    student = AutoModelForTokenClassification.from_config(config)
    student.bert.embeddings.load_state_dict(teacher.bert.embeddings.state_dict())
    for layer, original in zip(student.bert.encoder.layer, TEACHER_LAYERS, strict=True):
        layer.load_state_dict(teacher.bert.encoder.layer[original].state_dict())
    student.classifier.load_state_dict(teacher.classifier.state_dict())
    student.bert.embeddings.requires_grad_(False)
    return student


def prepare_training(teacher, tokenizer, rows, *, match_layers):
    teacher.eval()
    permutation = random.Random(SEED)
    permutation.shuffle(rows)
    labels = teacher.config.label2id
    for start in range(0, len(rows), 16):
        batch = rows[start:start + 16]
        inputs = tokenizer([question for question, _ in batch], padding=True, return_offsets_mapping=True,
                           return_tensors='pt')
        offsets = inputs.pop('offset_mapping')
        first, last = offsets.unbind(-1)
        valid = last > first
        targets = torch.full_like(first, labels['O']).masked_fill(~valid, -100)
        for i, (_, spans) in enumerate(batch):
            for left, right in spans:
                inside = valid[i] & (first[i] >= left) & (last[i] <= right)
                indices = inside.nonzero(as_tuple=True)[0]
                if len(indices) == 0 or first[i, indices[0]] != left or last[i, indices[-1]] != right:
                    targets[i].fill_(-100)
                    break
                targets[i, indices] = labels['I-PER']
                targets[i, indices[0]] = labels['B-PER']
        # Compute labels on CPU so the teacher can be released before training.
        with torch.no_grad():
            result = teacher(**inputs, output_hidden_states=match_layers)
            states = (tuple(result.hidden_states[layer + 1].half() for layer in TEACHER_LAYERS)
                      if match_layers else ())
        yield inputs, valid, targets, result.logits.detach(), states


def train(student, batches, *, epochs, match_layers, gold_first):
    student.to('cuda').train()
    optimizer = torch.optim.AdamW((param for param in student.parameters() if param.requires_grad),
                                  lr=6e-5, weight_decay=.01)
    generator = random.Random(SEED)
    order = list(range(len(batches)))
    labels = student.config.label2id
    weights = torch.ones(student.config.num_labels, device='cuda')
    weights[[labels['B-PER'], labels['I-PER']]] = 2.5
    temperature = 2.
    for epoch in range(epochs):
        generator.shuffle(order)
        total = 0.
        for index in order:
            inputs, valid, targets, reference, states = batches[index]
            inputs = {name: tensor.cuda() for name, tensor in inputs.items()}
            valid, targets, reference = valid.cuda(), targets.cuda(), reference.cuda()
            optimizer.zero_grad(set_to_none=True)
            result = student(**inputs, output_hidden_states=match_layers)
            logits = result.logits.float()
            supervised = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), weight=weights,
                                         reduction='sum') / targets.ne(-100).sum().clamp_min(1)
            distill_mask = valid
            if gold_first:
                expected_name = (targets == labels['B-PER']) | (targets == labels['I-PER'])
                predicted = reference.argmax(-1)
                predicted_name = (predicted == labels['B-PER']) | (predicted == labels['I-PER'])
                # The teacher can keep its non-person labels but cannot overwrite a known boundary.
                disagreement = (expected_name != predicted_name) | (expected_name & (targets != predicted))
                distill_mask = valid & ((targets == -100) | ~disagreement)
            distill_weights = distill_mask[valid].float()
            divergence = F.kl_div((logits[valid] / temperature).log_softmax(-1),
                                  (reference[valid] / temperature).softmax(-1), reduction='none').sum(-1)
            distilled = (divergence * distill_weights).sum() / distill_weights.sum().clamp_min(1)
            loss = supervised + .5 * temperature ** 2 * distilled
            if match_layers:
                losses = [1 - F.cosine_similarity(hidden[valid].float(), reference_state.cuda()[valid].float()).mean()
                          for hidden, reference_state in zip(result.hidden_states[1:], states, strict=True)]
                loss = loss + torch.stack(losses).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite distillation loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach())
        print('epoch', epoch + 1, 'mean loss', total / len(order), flush=True)
    return student.cpu().eval()


def evaluate(model, tokenizer, groups):
    groups = copy.deepcopy(groups)
    rows = [row for group in groups.values() for row in group['rows']]
    for row, tags in zip(rows, annotate(model.eval(), tokenizer, [row['question'] for row in rows]), strict=True):
        row['teacher'] = tags
    summary = {name: compare(group['rows']) for name, group in groups.items()}
    return {'summary': summary, 'groups': groups}


def main(teacher, baseline, output, epochs, match_layers=False, boundary_pairs=False, gold_first=False):
    if output.exists():
        raise FileExistsError(output)
    if not 1 <= epochs <= 12:
        raise ValueError('Epochs must be within [1, 12]')
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    started = time.perf_counter()
    weight_path = teacher / 'model.safetensors'
    if sha(weight_path) != WEIGHTS_SHA:
        raise ValueError('Expected the pinned Chinese NER model')
    baseline_sha = sha(baseline)
    groups = json.loads(baseline.read_text(encoding='utf-8'))['groups']
    tokenizer = AutoTokenizer.from_pretrained(teacher, local_files_only=True, trust_remote_code=False, use_fast=True)
    teacher_model = AutoModelForTokenClassification.from_pretrained(
        teacher, local_files_only=True, use_safetensors=True, trust_remote_code=False).eval()
    student = shallow_copy(teacher_model).eval()
    initial = evaluate(student, tokenizer, groups)
    print('shallow initialization', {group: scores['bio_single'] for group, scores in initial['summary'].items()},
          flush=True)
    held_names = (*NAMES, '小芮', '阿奕', *CALIBRATION_NAMES)
    names = (*TRAIN_NAMES, '秦维恩', '崔岚宁', '魏晗', '小恬', '阿郑', '朱少昂', '苏米', '阿琤')
    training = [*synthetic(TRAIN_NAMES), *expanded_examples(names), *news(held_names, expanded=True)]
    if boundary_pairs:
        training.extend(paired_boundaries(names))
    if any(name in text[start:end] for text, spans in training for start, end in spans for name in held_names):
        raise ValueError('Training data overlaps held names')
    batches = list(prepare_training(teacher_model, tokenizer, training, match_layers=match_layers))
    del teacher_model
    student = train(student, batches, epochs=epochs, match_layers=match_layers, gold_first=gold_first)
    del batches
    distilled = evaluate(student, tokenizer, groups)
    print('distilled', {group: scores['bio_single'] for group, scores in distilled['summary'].items()}, flush=True)
    if sha(weight_path) != WEIGHTS_SHA or sha(baseline) != baseline_sha:
        raise ValueError('Experiment inputs changed')
    output.mkdir(parents=True, exist_ok=False)
    student.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    report = {'teacher': TEACHER, 'teacher_revision': REVISION, 'teacher_sha256': WEIGHTS_SHA,
              'copied_layers': list(TEACHER_LAYERS), 'training_rows': len(training), 'epochs': epochs,
              'match_intermediate_layers': match_layers,
              'paired_boundaries': boundary_pairs,
              'prefer_annotated_boundaries': gold_first,
              'seed': SEED, 'baseline_sha256': baseline_sha, 'seconds': time.perf_counter() - started,
              'student_sha256': sha(output / 'model.safetensors'),
              'initial': initial, 'distilled': distilled}
    with (output / 'result.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('teacher', 'baseline', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--match-layers', action='store_true')
    parser.add_argument('--boundary-pairs', action='store_true')
    parser.add_argument('--gold-first', action='store_true')
    main(**vars(parser.parse_args()))
