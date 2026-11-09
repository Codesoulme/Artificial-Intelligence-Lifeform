"""Train a number-invariant current-value intent gate with separate format validation."""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_local_baseline import MODEL, REVISION, WEIGHT_SHA
from scripts.deepseek_memory_intent import IntentGate, build_cases
from scripts.deepseek_pooled_address import extract
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.finite_question_guard import valid_finite_question_form
from spora.growing import load
from spora.language_training import frozen_digest


def hide_numbers(question):
    if not isinstance(question, str):
        raise ValueError('Expected a question string')
    return re.sub(r'[0-9]+', '[编号]', question)


def unique_semantic_cases(rows, enforce_labels=True):
    unique = {}
    for row in rows:
        question = hide_numbers(row['question'])
        if enforce_labels and question in unique and unique[question]['label'] != row['label']:
            raise ValueError('Conflicting labels after hiding numbers')
        unique.setdefault(question, {'question': question, 'category': row['category'], 'label': row['label']})
    return list(unique.values())


@torch.no_grad()
def evaluate(gate, features, masks, unique, rows, batch_size=64):
    probabilities = []
    for offset in range(0, len(unique), batch_size):
        logits = gate(features[offset:offset + batch_size].to('cuda'), masks[offset:offset + batch_size].to('cuda'))
        if not bool(torch.isfinite(logits).all()):
            raise ValueError('Nonfinite gate outputs')
        probabilities.extend(logits.sigmoid().cpu().tolist())
    lookup = {row['question']: float(score) for row, score in zip(unique, probabilities)}
    result = []
    for row in rows:
        hidden = hide_numbers(row['question'])
        form_ok = valid_finite_question_form(row['question'])
        score = lookup[hidden] if form_ok else None
        accepted = form_ok and score >= 0.5
        result.append(dict(row, sanitized=hidden, probability=score, form_accepted=form_ok, accepted=accepted))
    categories = {}
    for category in sorted({row['category'] for row in rows}):
        group = [row for row in result if row['category'] == category]
        categories[category] = {'examples': len(group), 'unique_sanitized': len({row['sanitized'] for row in group}),
                                'accepted': sum(row['accepted'] for row in group),
                                'correct': sum(row['accepted'] == bool(row['label']) for row in group)}
    return {'examples': len(rows), 'unique_sanitized': len(unique),
            'correct': sum(row['accepted'] == bool(row['label']) for row in result),
            'categories': categories, 'rows': result}


def run(output, steps):
    output = Path(output).resolve()
    runs = (ROOT / 'runs').resolve()
    if not output.is_relative_to(runs) or output == runs or output.exists() or not 1 <= steps <= 5000:
        raise ValueError('Expected a new output under runs and valid steps')
    def verify():
        return sha(MODEL / 'model.safetensors') == WEIGHT_SHA and sha(INDIVIDUAL) == INDIVIDUAL_SHA
    if not verify():
        raise ValueError('Protected source mismatch')
    torch.manual_seed(4611)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(organism, original)
    training = [row for row in build_cases(augmented=True, malformed=True, contrastive=True)
                if row['category'] not in ('out_of_range', 'malformed')]
    held = build_cases(True, augmented=True, malformed=True, contrastive=True)
    train_unique = unique_semantic_cases(training)
    held_unique = unique_semantic_cases([row for row in held if valid_finite_question_form(row['question'])],
                                        enforce_labels=False)
    overlap = {row['question'] for row in train_unique} & {row['question'] for row in held_unique}
    if overlap:
        raise ValueError(f'Train/held normalized questions overlap: {sorted(overlap)}')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    features, masks = extract(model, tokenizer, [r['question'] for r in train_unique])
    held_features, held_masks = extract(model, tokenizer, [r['question'] for r in held_unique])
    labels = torch.tensor([row['label'] for row in train_unique], dtype=torch.float32)
    positive = torch.nonzero(labels == 1).flatten()
    negative = torch.nonzero(labels == 0).flatten()
    generator = torch.Generator().manual_seed(4611)
    gate = IntentGate(model.config.hidden_size).to('cuda')
    optimizer = torch.optim.AdamW(gate.parameters(), lr=0.001, weight_decay=0.)
    log = []
    for step in range(steps):
        indices = torch.cat((positive[torch.randint(len(positive), (32,), generator=generator)],
                             negative[torch.randint(len(negative), (32,), generator=generator)]))
        logits = gate(features[indices].to('cuda'), masks[indices].to('cuda'))
        loss = F.binary_cross_entropy_with_logits(logits, labels[indices].to('cuda'))
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
        raise ValueError('Protected inputs changed')
    output.mkdir(parents=False, exist_ok=False)
    snapshot = output / 'gate.pt'
    torch.save({'format': 'deepseek-spora-number-invariant-intent-v1',
                'state': {k: v.detach().cpu() for k, v in gate.state_dict().items()},
                'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA,
                'layers': [5, 17], 'threshold': 0.5}, snapshot)
    print('semantic gate checkpoint saved; evaluation starting', flush=True)
    results = {'train': evaluate(gate, features, masks, train_unique, training),
               'held': evaluate(gate, held_features, held_masks, held_unique, held)}
    if frozen_digest(organism, original) != before or not verify():
        raise ValueError('Protected input changed during evaluation')
    report = {'status': 'number_invariant_semantic_gate_development_only', 'revision': REVISION,
              'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA, 'gate_sha256': sha(snapshot),
              'steps': steps, 'threshold': 0.5, 'results': results, 'log': log,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Intent gate receives numerals replaced with a placeholder; a separate symbolic format guard is required.',
                              'The guard is restricted to one canonical in-range Arabic identifier.',
                              'All examples are synthetic and held questions are few template families.',
                              'Unknown in-range memory values require a separate state-aware check.',
                              'Gate is not integrated into a user-facing runtime.',
                              'Fixed threshold was selected before evaluation and is not calibrated.']}
    with (output / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({name: {k: v for k, v in value.items() if k != 'rows'} for name, value in results.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.steps)
