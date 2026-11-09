"""Compare full-question pooling with oracle name-span pooling for bindings."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_named_binding_migrate import OUTPUT as SOURCE
from scripts.deepseek_named_binding_probe import HELD_TEMPLATES, KEYS, NAMES, TRAIN_TEMPLATES, UNKNOWN, fit_binding, pooled, predict
from scripts.deepseek_named_incremental_probe import NEW_KEYS
from scripts.deepseek_pooled_address import extract, question_mask
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-name-span-20261110'


def oracle_span_mask(offsets, attention_mask, texts, questions, names):
    question_tokens = question_mask(offsets, attention_mask, texts, questions)
    if len(names) != len(questions):
        raise ValueError('Expected one name per question')
    result = torch.zeros_like(attention_mask, dtype=torch.bool)
    crossings = []
    for i, (text, question, name) in enumerate(zip(texts, questions, names)):
        if not name or question.count(name) != 1:
            raise ValueError('Exactly one explicitly provided name required')
        start = text.find(question) + question.find(name)
        end = start + len(name)
        left, right = offsets[i, :, 0], offsets[i, :, 1]
        result[i] = (left < end) & (right > start) & question_tokens[i]
        if not bool(result[i].any()):
            raise ValueError('Name has no selectable tokens')
        crossings.append(bool((((left < start) | (right > end)) & result[i]).any()))
    return result, crossings


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    individual, report_path = SOURCE / 'individual.pt', SOURCE / 'result.json'
    protected = {p: sha(p) for p in (MODEL / 'model.safetensors', INDIVIDUAL, individual, report_path)}
    if protected[MODEL / 'model.safetensors'] != WEIGHT_SHA or protected[INDIVIDUAL] != INDIVIDUAL_SHA:
        raise ValueError('Protected model mismatch')
    with report_path.open(encoding='utf-8') as stream:
        previous = json.load(stream)
    config_sha = feature_files_sha256(MODEL)
    if previous['saved_individual_sha256'] != protected[individual] or previous['feature_config_sha256'] != config_sha:
        raise ValueError('Individual or encoder configuration mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, state, _ = load(individual, 'cpu')
    state_digest = frozen_digest(model, state)
    bases = torch.stack([F.normalize(site.current_basis.detach().float(), dim=-1) for site in model.memories])
    training = [(name, key, template.format(name=name)) for name, key in zip(NAMES, KEYS) for template in TRAIN_TEMPLATES]
    training += [(name, key, template.format(name=name)) for name, key in zip(UNKNOWN[:2], NEW_KEYS)
                 for template in TRAIN_TEMPLATES[:2]]
    evaluation = [(name, key, template.format(name=name)) for name, key in zip(NAMES + UNKNOWN[:2], KEYS + NEW_KEYS)
                  for template in HELD_TEMPLATES]
    evaluation += [(name, None, template.format(name=name)) for name in UNKNOWN[2:] for template in HELD_TEMPLATES]
    if set(q for _, _, q in training) & set(q for _, _, q in evaluation):
        raise ValueError('Training and evaluation overlap')
    examples = training + evaluation
    questions = [q for _, _, q in examples]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False,
                                           add_generation_prompt=True) for q in questions]
    encoding = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False, return_offsets_mapping=True)
    span_mask, crossings = oracle_span_mask(encoding['offset_mapping'], encoding['attention_mask'], texts, questions,
                                           [name for name, _, _ in examples])
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    features, question_masks = extract(base, tokenizer, questions, batch_size=1)
    if span_mask.shape != question_masks.shape or bool((span_mask & ~question_masks).any()):
        raise ValueError('Tokenization or mask mismatch')
    n_train = len(training)
    targets = torch.stack([bases[:, key] for _, key, _ in training])
    results = {}
    snapshots = {}
    for method, masks in (('whole_question', question_masks), ('oracle_name_span', span_mask)):
        vectors = pooled(features, masks)
        weights = fit_binding(vectors[:n_train], targets)
        train_indices, train_scores = predict(vectors[:n_train], weights, bases)
        indices, scores = predict(vectors[n_train:], weights, bases)
        labels = torch.tensor([key for _, key, _ in training])
        threshold = float(train_scores.min()) - 1e-6
        accepted = (scores >= threshold).all(-1) & (indices[:, 0] == indices[:, 1])
        rows = []
        for i, (name, key, question) in enumerate(evaluation):
            predicted = indices[i].tolist()
            rows.append({'name': name, 'question': question, 'expected_key': key, 'indices': predicted,
                         'scores': scores[i].tolist(), 'accepted': bool(accepted[i]),
                         'correct': key is not None and bool(accepted[i]) and predicted == [key, key],
                         'token_crosses_name_boundary': crossings[n_train + i]})
        results[method] = {'threshold': threshold, 'train_correct': int((train_indices == labels[:, None]).all(-1).sum()),
                           'old_correct': sum(r['correct'] for r in rows[:16]),
                           'new_correct': sum(r['correct'] for r in rows[16:20]),
                           'unknown_false_accept': sum(r['accepted'] for r in rows[20:]), 'rows': rows}
        snapshots[method] = weights
    if (frozen_digest(model, state) != state_digest or feature_files_sha256(MODEL) != config_sha
            or any(sha(p) != d for p, d in protected.items())):
        raise ValueError('Protected input changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT / 'projections.pt'
    torch.save({'format': 'oracle-name-span-development-v1', 'weights': snapshots,
                'weight_sha256': WEIGHT_SHA, 'feature_config_sha256': config_sha,
                'individual_sha256': INDIVIDUAL_SHA}, checkpoint)
    report = {'status': 'oracle_span_development_only', 'weight_sha256': WEIGHT_SHA,
              'individual_sha256': INDIVIDUAL_SHA, 'projection_sha256': sha(checkpoint),
              'train_examples': len(training), 'old_held': 16, 'new_held': 4, 'unknown_held': 4,
              'crossing_token_examples': sum(crossings), 'results': results,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Name character span is supplied by the evaluation labels, not inferred by the model.',
                              'Exactly one explicit name per question; no pronouns, ambiguity or multi-entity requests.',
                              'The same small template families and names were used in previous development.',
                              'Thresholds are minimum training similarities and are not calibrated confidence.',
                              'Only two unbound names and four questions test rejection.',
                              'Re-fits a batch as a diagnostic, not an incremental lifecycle or live chat.',
                              'No memory readout, write, fact values, persistence or end-to-end generation tested.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: {name: v for name, v in item.items() if name != 'rows'} for k, item in results.items()}), flush=True)


if __name__ == '__main__':
    run()
