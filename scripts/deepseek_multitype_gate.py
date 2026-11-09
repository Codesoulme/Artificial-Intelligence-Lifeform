"""Development-only multi-type routing from number-hidden DeepSeek features."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_guard_adversarial import SECOND_CASES
from scripts.deepseek_local_baseline import MODEL, REVISION, WEIGHT_SHA
from scripts.deepseek_memory_intent import IntentGate, build_cases
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_semantic_gate import hide_numbers
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.finite_question_guard import valid_finite_question_form
from spora.growing import load
from spora.language_training import frozen_digest

CLASSES = ('current', 'history', 'write', 'multiple', 'derived', 'source', 'negated', 'unrelated')
THRESHOLD = 0.8


class MultiIntentGate(IntentGate):
    def __init__(self, hidden_width):
        super().__init__(hidden_width)
        self.classifier = nn.Linear(2 * hidden_width, len(CLASSES))


def checkpoint_payload(gate, hidden_indices):
    if (not isinstance(hidden_indices, (tuple, list)) or len(hidden_indices) != 2
            or any(type(index) is not int or index < 1 for index in hidden_indices)
            or hidden_indices[0] == hidden_indices[1]):
        raise ValueError('Invalid hidden-state metadata')
    return {'format': 'deepseek-spora-multitype-intent-v1',
            'state': {k: v.detach().cpu() for k, v in gate.state_dict().items()},
            'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA,
            'hidden_state_indices': tuple(hidden_indices), 'threshold': THRESHOLD, 'classes': CLASSES}


def unique_cases(rows):
    unique = {}
    for row in rows:
        text = hide_numbers(row['question'])
        category = row['category']
        if category not in CLASSES:
            raise ValueError('Unsupported training category')
        if text in unique and unique[text]['category'] != category:
            raise ValueError('Conflicting categories after hiding numbers')
        unique.setdefault(text, {'question': text, 'category': category})
    return list(unique.values())


@torch.no_grad()
def infer(gate, features, masks, batch_size=64):
    values = []
    for offset in range(0, len(features), batch_size):
        logits = gate(features[offset:offset + batch_size].to('cuda'), masks[offset:offset + batch_size].to('cuda'))
        if not bool(torch.isfinite(logits).all()):
            raise ValueError('Nonfinite gate outputs')
        values.extend(logits.softmax(-1).cpu().tolist())
    return values


def summarize(rows):
    groups = {}
    for category in sorted({row['category'] for row in rows}):
        group = [row for row in rows if row['category'] == category]
        groups[category] = {'examples': len(group), 'unique_sanitized': len({r['sanitized'] for r in group}),
                            'accepted': sum(r['accepted'] for r in group),
                            'correct': sum(r['accepted'] == (r['category'] == 'current') for r in group)}
    return {'examples': len(rows), 'correct': sum(r['accepted'] == (r['category'] == 'current') for r in rows),
            'categories': groups, 'rows': rows}


def run(output, steps, deep_features=False):
    output = Path(output).resolve()
    runs = (ROOT / 'runs').resolve()
    if not output.is_relative_to(runs) or output == runs or output.exists() or not 1 <= steps <= 5000:
        raise ValueError('Expected a new output under runs and valid steps')
    def verify():
        return sha(MODEL / 'model.safetensors') == WEIGHT_SHA and sha(INDIVIDUAL) == INDIVIDUAL_SHA
    if not verify():
        raise ValueError('Protected source mismatch')
    torch.manual_seed(4614)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(organism, original)
    training = [row for row in build_cases(augmented=True, malformed=True, contrastive=True)
                if row['category'] in CLASSES]
    held = build_cases(True, augmented=True, malformed=True, contrastive=True)
    train_unique = unique_cases(training)
    held_valid = [row for row in held if valid_finite_question_form(row['question'])]
    held_texts = sorted({hide_numbers(row['question']) for row in held_valid})
    stress = [{'category': ('current' if label else category), 'question': question}
              for label, category, question in SECOND_CASES]
    stress_valid = [row for row in stress if valid_finite_question_form(row['question'])]
    stress_texts = sorted({hide_numbers(row['question']) for row in stress_valid})
    train_texts = [row['question'] for row in train_unique]
    if set(train_texts) & (set(held_texts) | set(stress_texts)):
        raise ValueError('Train and evaluation normalized questions overlap')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    hidden_indices = (24, 28) if deep_features else (6, 18)
    features, masks = extract(model, tokenizer, train_texts, hidden_indices=hidden_indices)
    held_features, held_masks = extract(model, tokenizer, held_texts, hidden_indices=hidden_indices)
    stress_features, stress_masks = extract(model, tokenizer, stress_texts, hidden_indices=hidden_indices)
    labels = torch.tensor([CLASSES.index(row['category']) for row in train_unique], dtype=torch.long)
    groups = [torch.nonzero(labels == i).flatten() for i in range(len(CLASSES))]
    if any(len(group) == 0 for group in groups):
        raise ValueError('Missing training class')
    generator = torch.Generator().manual_seed(4614)
    gate = MultiIntentGate(model.config.hidden_size).to('cuda')
    optimizer = torch.optim.AdamW(gate.parameters(), lr=0.001, weight_decay=0.)
    log = []
    for step in range(steps):
        batch_indices = torch.cat([group[torch.randint(len(group), (8,), generator=generator)] for group in groups])
        logits = gate(features[batch_indices].to('cuda'), masks[batch_indices].to('cuda'))
        loss = F.cross_entropy(logits, labels[batch_indices].to('cuda'))
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite gate loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(gate.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 250 == 0 or step + 1 == steps:
            entry = {'step': step + 1, 'loss': float(loss.detach()), 'gradient_norm': float(norm)}
            log.append(entry)
            print(json.dumps(entry), flush=True)
    gate.eval()
    if any(p.grad is not None for p in model.parameters()) or frozen_digest(organism, original) != before or not verify():
        raise ValueError('Protected input changed')
    output.mkdir(parents=False, exist_ok=False)
    snapshot = output / 'gate.pt'
    torch.save(checkpoint_payload(gate, hidden_indices), snapshot)
    print('multitype checkpoint saved; evaluation starting', flush=True)
    held_lookup = dict(zip(held_texts, infer(gate, held_features, held_masks)))
    stress_lookup = dict(zip(stress_texts, infer(gate, stress_features, stress_masks)))
    results = {}
    for name, cases, lookup in (('held', held, held_lookup), ('stress_development', stress, stress_lookup)):
        rows = []
        for case in cases:
            text = hide_numbers(case['question'])
            form_ok = valid_finite_question_form(case['question'])
            probs = lookup[text] if form_ok else None
            predicted = CLASSES[max(range(len(CLASSES)), key=lambda i: probs[i])] if probs is not None else None
            accepted = bool(form_ok and predicted == 'current' and probs[0] >= THRESHOLD)
            rows.append(dict(case, sanitized=text, form_ok=form_ok, predicted_type=predicted,
                             current_probability=probs[0] if probs is not None else None, accepted=accepted))
        results[name] = summarize(rows)
    if frozen_digest(organism, original) != before or not verify():
        raise ValueError('Protected input changed during evaluation')
    report = {'status': 'multitype_intent_development_only', 'revision': REVISION,
              'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA, 'gate_sha256': sha(snapshot),
              'steps': steps, 'classes': CLASSES, 'threshold': THRESHOLD, 'train_unique': len(train_unique),
              'hidden_state_indices': hidden_indices,
              'results': results, 'log': log, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Eight synthetic request types; other intents are not modeled.',
                              'Numbers are hidden for intent and separately checked by a symbolic finite-format guard.',
                              'Few independent templates; held/stress examples are development, not final evaluation.',
                              'Threshold is fixed, not calibrated for open-domain queries.',
                              'Unknown in-range memory values need a separate state-aware check.',
                              'No integration into a user-facing runtime.']}
    with (output / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({name: {k: v for k, v in result.items() if k != 'rows'} for name, result in results.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deep-features', action='store_true')
    args = parser.parse_args()
    run(args.output, args.steps, args.deep_features)
