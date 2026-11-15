"""Evaluate a sentence-level abstention gate before an adapted span head.

Labels are weak or agent-reviewed. This is an offline probe, not a deployed
guard, identity verifier, or guarantee that a single detected span is correct.
"""
import json
import sys
import time
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer
from scripts import deepseek_spora_duconv_dev_identity_audit as dev
from scripts import deepseek_spora_natural_span_adaptation as adaptation
from scripts import deepseek_spora_natural_span_test_audit as held
from scripts import deepseek_spora_name_extraction_audit as synthetic
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.audit_p236_duconv import ARCHIVE, ARCHIVE_SHA
from scripts.deepseek_spora_joint_span_pilot import JointSpan, selected_text, token_span
from scripts.deepseek_spora_prefix_features import load_prefix
from scripts.deepseek_spora_prefix_parity_audit import OUTPUT as PARITY
from spora.identity_fingerprints import IdentityFingerprints

OUT = ROOT / 'runs/deepseek-spora-sentence-gate-audit-20261116'
SEED = 61813


def synthetic_name_groups(names):
    if len(names) != 24 or len(set(names)) != 24:
        raise ValueError('Expected 24 distinct synthetic names')
    fit, calibration = [], []
    for target, group_names in ((fit, names[:16]), (calibration, names[16:])):
        for name in group_names:
            for template in ('请问{name}的饮品偏好？', '记得{name}住在哪个城市吗？'):
                target.append({'question': template.format(name=name), 'subject': name,
                               'group': 'synthetic_single', 'doc': None})
        for i in range(0, len(group_names), 2):
            a, b = group_names[i:i + 2]
            for question in (f'比较{a}和{b}喜欢什么？', f'{a}与{b}各住在哪里？'):
                target.append({'question': question, 'subject': None,
                               'group': 'synthetic_multiple', 'doc': None})
    return fit, calibration


def split_training(rows):
    groups = ('weak_single', 'weak_multiple', 'synthetic_none')
    fit, calibration = [], []
    for group in groups:
        indices = sorted((i for i, row in enumerate(rows) if row['group'] == group),
                         key=lambda i: dev.digest(rows[i]['question']))
        if len(indices) < 4:
            raise ValueError('Insufficient group size for calibration')
        calibration.extend(indices[::4])
        fit.extend(index for position, index in enumerate(indices) if position % 4)
    if set(fit) & set(calibration) or len(fit) + len(calibration) != len(rows):
        raise ValueError('Training split invalid')
    return fit, calibration


def calibrated_cutoff(scores, labels):
    if len(scores) != len(labels) or not scores or any(type(x) not in (int, float) or not 0 <= x <= 1 for x in scores):
        raise ValueError('Finite probability scores and labels required')
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not negatives or any(label not in (0, 1) for label in labels) or 1 not in labels:
        raise ValueError('Calibration requires both classes')
    return max(negatives)


def main(include_synthetic=False):
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    reviewed = dev.reviewed_sample(dev.select_sample())
    test = held.reviewed_sample(held.select_sample())
    if runtime.sha(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError('Archive checksum mismatch')
    with ZipFile(ARCHIVE) as source:
        training = adaptation.build_training(list(source.open('DuConv/train.txt')), reviewed)
    fit_indices, calibration_indices = split_training(training)
    synthetic_name_digests = {}
    if include_synthetic:
        import pyarrow.parquet as parquet
        from scripts.deepseek_cluener_locator_train import TRAIN, TRAIN_SHA
        from scripts.deepseek_cluener_span_audit import SOURCE, SOURCE_SHA
        from scripts.deepseek_spora_boundary_train_pilot import extract_names

        if runtime.sha(TRAIN) != TRAIN_SHA or runtime.sha(SOURCE) != SOURCE_SHA:
            raise ValueError('Name-source checksum mismatch')
        source_names = extract_names(parquet.read_table(TRAIN).to_pylist())
        held_source_names = extract_names(parquet.read_table(SOURCE).to_pylist())
        forbidden = ({name for row in reviewed + test for name in row['names']}
                     | {row['subject'] for row in reviewed + test if row['subject']}
                     | {name for _, _, name in synthetic.CASES if name})
        names = sorted((name for name in source_names - held_source_names
                        if not any(name in other or other in name for other in forbidden)
                        and not any(name in row['question'] for row in training)), key=dev.digest)[:24]
        if len(names) != 24:
            raise ValueError('Insufficient disjoint names for synthetic training')
        extra_fit, extra_calibration = synthetic_name_groups(names)
        synthetic_name_digests = {'fit': [dev.digest(name) for name in names[:16]],
                                  'calibration': [dev.digest(name) for name in names[16:]],
                                  'train_source_sha256': TRAIN_SHA, 'held_source_sha256': SOURCE_SHA}
        fit_indices.extend(range(len(training), len(training) + len(extra_fit)))
        training += extra_fit
        calibration_indices.extend(range(len(training), len(training) + len(extra_calibration)))
        training += extra_calibration
    evaluation = [{'question': row['text'], 'subject': row['subject'],
                   'group': 'dev_' + row['reviewed_group']} for row in reviewed]
    evaluation += [{'question': row['text'], 'subject': row['subject'],
                    'group': 'test_' + row['reviewed_group']} for row in test]
    evaluation += [{'question': question, 'subject': expected, 'group': 'synthetic_' + group}
                   for group, question, expected in synthetic.CASES]
    questions = [row['question'] for row in training + evaluation]
    if len(set(questions)) != len(questions):
        raise ValueError('Train, calibration and evaluation questions overlap')
    source_sha = runtime.sha(runtime.CHECKPOINT)
    parity = json.loads(PARITY.read_text(encoding='utf-8'))
    if (parity['source_checkpoint_sha256'] != source_sha or parity['examples'] != 10
            or parity['span_matches'] != 10 or parity['max_entity_score_difference'] > .001):
        raise ValueError('Prefix parity evidence missing')
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    model, _, variant = load_prefix()
    if variant != parity['feature_variant_sha256']:
        raise ValueError('Feature variant mismatch')
    tokenizer = AutoTokenizer.from_pretrained(runtime.MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': question}], tokenize=False,
                                           add_generation_prompt=True) for question in questions]
    encoded = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False,
                        return_offsets_mapping=True)
    features, masks = runtime.extract(model, tokenizer, questions, batch_size=1)
    vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],))
                         for i in range(2)], dim=-1).cpu()
    masks = masks.cpu()
    pooled = []
    for vector, mask in zip(vectors, masks):
        positions = mask.nonzero(as_tuple=True)[0]
        if not len(positions):
            raise ValueError('Question contains no eligible tokens')
        pooled.append(torch.cat((vector[positions].mean(dim=0), vector[positions[-1]])))
    pooled = torch.stack(pooled)
    del model, features
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mean = pooled[fit_indices].mean(dim=0)
    scale = pooled[fit_indices].std(dim=0).clamp_min(.01)
    x = (pooled - mean) / scale
    labels = torch.tensor([int(row['group'] in ('weak_single', 'synthetic_single')) for row in training])
    gate = nn.Linear(x.shape[1], 2)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=.001, weight_decay=.1)
    weights = torch.tensor([1 / int((labels[fit_indices] == i).sum()) for i in range(2)])
    for _ in range(500):
        loss = F.cross_entropy(gate(x[fit_indices]), labels[fit_indices], weight=weights)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite gate loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        probabilities = F.softmax(gate(x), dim=-1)[:, 1]
    cutoff = calibrated_cutoff([float(probabilities[i]) for i in calibration_indices],
                               [int(labels[i]) for i in calibration_indices])
    calibration = {'examples': len(calibration_indices), 'cutoff_strictly_greater': cutoff,
                   'positive': sum(int(labels[i]) for i in calibration_indices),
                   'positive_accepted': sum(int(labels[i]) == 1 and float(probabilities[i]) > cutoff
                                            for i in calibration_indices),
                   'negative': sum(int(labels[i]) == 0 for i in calibration_indices),
                   'negative_accepted': sum(int(labels[i]) == 0 and float(probabilities[i]) > cutoff
                                            for i in calibration_indices)}
    directories = {'baseline': ROOT / 'runs/deepseek-spora-joint-span-crossed-pilot-20261113',
                   'candidate': adaptation.OUT}
    heads, head_hashes = {}, {}
    for label, directory in directories.items():
        info = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        path = directory / 'span.pt'
        head_hashes[label] = runtime.sha(path)
        expected = info['weights_sha256'] if label == 'baseline' else info['candidate_sha256']
        if head_hashes[label] != expected or info['feature_variant_sha256'] != variant:
            raise ValueError('Span candidate mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload['feature_variant_sha256'] != variant:
            raise ValueError('Embedded feature mismatch')
        head = JointSpan(vectors.shape[-1], width=payload['width']).eval()
        head.load_state_dict(payload['state'], strict=True)
        heads[label] = head
    fingerprints = IdentityFingerprints(len(synthetic.ENROLLED))
    for key, name in enumerate(synthetic.ENROLLED):
        fingerprints.bind(key, name)
    rows = []
    with torch.no_grad():
        for index, item in enumerate(evaluation, start=len(training)):
            start = texts[index].find(item['question'])
            target = token_span(encoded['offset_mapping'][index], masks[index], start,
                                item['question'], item['subject'])
            selected = {}
            for label, head in heads.items():
                choice = int(head(vectors[index], masks[index]).argmax())
                selected[label] = selected_text(choice, encoded['offset_mapping'][index], start, item['question'])
            accepted = float(probabilities[index]) > cutoff
            selected['gated'] = selected['candidate'] if accepted else None
            row = {'group': item['group'], 'question_sha256': dev.digest(item['question']),
                   'representable': target is not None, 'gate_probability': float(probabilities[index]),
                   'gate_accepted': accepted}
            for label, name in selected.items():
                result = {'correct': name == item['subject'], 'selected': name is not None}
                if item['group'].startswith('synthetic_'):
                    key = fingerprints.lookup(name) if name else None
                    expected = synthetic.ENROLLED.index(item['subject']) if item['subject'] in synthetic.ENROLLED else None
                    result['route_correct'] = key == expected
                    result['false_accept'] = key is not None and expected is None
                row[label] = result
            rows.append(row)
    if runtime.sha(ARCHIVE) != ARCHIVE_SHA or runtime.sha(runtime.CHECKPOINT) != source_sha:
        raise ValueError('Protected input changed')
    torch.save({'state': gate.state_dict(), 'mean': mean, 'scale': scale, 'cutoff': cutoff,
                'feature_variant_sha256': variant, 'source_checkpoint_sha256': source_sha}, OUT / 'gate.pt')
    summary = {}
    for group in sorted({row['group'] for row in rows}):
        items = [row for row in rows if row['group'] == group]
        summary[group] = {'total': len(items), 'gate_accepted': sum(row['gate_accepted'] for row in items)}
        for label in ('baseline', 'candidate', 'gated'):
            summary[group][label] = {metric: sum(row[label].get(metric, False) for row in items)
                                     for metric in ('correct', 'selected', 'route_correct', 'false_accept')}
    report = {'status': 'offline_weak_label_development_only', 'summary': summary, 'rows': rows,
              'fit_examples': len(fit_indices), 'calibration': calibration, 'gate_sha256': runtime.sha(OUT / 'gate.pt'),
              'synthetic_name_groups': synthetic_name_digests,
              'candidate_sha256': head_hashes, 'feature_variant_sha256': variant,
              'source_checkpoint_sha256': source_sha, 'archive_sha256': ARCHIVE_SHA,
              'seconds': time.perf_counter() - started,
              'limitations': ['Training and calibration labels are weak knowledge-derived or synthetic.',
                              'Calibration set is small and zero observed false acceptance is not a guarantee.',
                              'Both reviewed dialogue sets and the synthetic cases were observed in development.',
                              'No-person and multi-person abstention is a conservative test policy.',
                              'Gate and span head share frozen features; errors are not independent.',
                              'No runtime integration, automatic enrollment, actual memory read or multi-turn test.',
                              'Older SPORA-graft prefix, not the current low-rank edited language model.']}
    with (OUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'calibration': calibration, 'summary': summary}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--include-synthetic']:
        OUT = ROOT / 'runs/deepseek-spora-sentence-gate-synthetic-audit-20261116'
        main(include_synthetic=True)
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit('Usage: script [--include-synthetic]')
