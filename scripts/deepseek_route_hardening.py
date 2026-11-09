"""Continue a frozen-base intent gate on known failures and compare held questions."""
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

from scripts.deepseek_guard_adversarial import SECOND_CASES
from scripts.deepseek_local_baseline import MODEL, REVISION, WEIGHT_SHA
from scripts.deepseek_memory_intent import IntentGate, build_cases
from scripts.deepseek_oasst_route_probe import load_questions, REVIEW, SOURCE
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_route_calibration import NEW_CASES
from scripts.deepseek_route_comparison import BINARY, CASES
from scripts.deepseek_semantic_gate import hide_numbers, unique_semantic_cases
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.finite_question_guard import valid_finite_question_form

OUTPUT = ROOT / 'runs/deepseek-route-hardening-20261110'
STEPS = 500


@torch.no_grad()
def scores(gate, features, masks):
    result = []
    for start in range(0, len(features), 4):
        x = features[start:start + 4].to('cuda')
        m = masks[start:start + 4].to('cuda')
        value = gate(x, m).sigmoid()
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite gate outputs')
        result.extend(value.cpu().tolist())
    return result


def evaluate(rows, old_scores, new_scores):
    if len(rows) != len(old_scores) or len(rows) != len(new_scores):
        raise ValueError('Scores must match evaluation rows')
    result = []
    for row, old, new in zip(rows, old_scores, new_scores):
        form = valid_finite_question_form(row['question'])
        result.append(dict(row, form_ok=form, old_score=old, new_score=new,
                           old_accept=form and old >= 0.5, new_accept=form and new >= 0.5))
    summary = {}
    for which in ('old', 'new'):
        accepted = which + '_accept'
        summary[which] = {'correct': sum(r[accepted] == bool(r['label']) for r in result),
                          'false_accept': sum(r[accepted] and not r['label'] for r in result),
                          'false_defer': sum(not r[accepted] and r['label'] for r in result)}
    return {'examples': len(result), 'summary': summary, 'rows': result}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    protected = {p: sha(p) for p in (MODEL / 'model.safetensors', INDIVIDUAL, BINARY, REVIEW, SOURCE)}
    if protected[MODEL / 'model.safetensors'] != WEIGHT_SHA or protected[INDIVIDUAL] != INDIVIDUAL_SHA:
        raise ValueError('Protected model mismatch')
    old_data = torch.load(BINARY, map_location='cpu', weights_only=True)
    if (old_data.get('format') != 'deepseek-spora-number-invariant-intent-v1'
            or old_data.get('weight_sha256') != WEIGHT_SHA or old_data.get('individual_sha256') != INDIVIDUAL_SHA
            or old_data.get('layers') != [5, 17] or old_data.get('threshold') != 0.5):
        raise ValueError('Unexpected initial gate')
    train = [r for r in build_cases(augmented=True, malformed=True, contrastive=True)
             if r['category'] not in ('out_of_range', 'malformed')]
    train.extend({'question': question, 'category': category, 'label': int(label)}
                 for label, category, question in CASES)
    unique = unique_semantic_cases(train)
    eval_sets = {
        'new_handwritten': [{'question': q, 'category': c, 'label': label} for label, c, q in NEW_CASES],
        'previous_stress': [{'question': q, 'category': c, 'label': label} for label, c, q in SECOND_CASES],
        'oasst_prefixed': [{'question': '对象8：' + r['question'], 'category': 'unrelated', 'label': 0}
                           for r in load_questions()],
    }
    training_texts = {r['question'] for r in unique}
    for name, rows in eval_sets.items():
        overlap = training_texts & {hide_numbers(r['question']) for r in rows}
        if overlap:
            raise ValueError(f'Train/evaluation overlap in {name}: {overlap}')
    torch.manual_seed(4624)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    print('model loaded', flush=True)
    features, masks = extract(model, tokenizer, [r['question'] for r in unique], batch_size=1)
    labels = torch.tensor([r['label'] for r in unique], dtype=torch.float32)
    positive, negative = (torch.nonzero(labels == value).flatten() for value in (1, 0))
    if not len(positive) or not len(negative):
        raise ValueError('Missing class')
    old = IntentGate(model.config.hidden_size).to('cuda').eval()
    old.load_state_dict(old_data['state'], strict=True)
    gate = IntentGate(model.config.hidden_size).to('cuda')
    gate.load_state_dict(old_data['state'], strict=True)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=0.0001, weight_decay=0.)
    generator = torch.Generator().manual_seed(4624)
    log = []
    for step in range(STEPS):
        batch = torch.cat((positive[torch.randint(len(positive), (16,), generator=generator)],
                           negative[torch.randint(len(negative), (16,), generator=generator)]))
        logits = gate(features[batch].to('cuda'), masks[batch].to('cuda'))
        loss = F.binary_cross_entropy_with_logits(logits, labels[batch].to('cuda'))
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(gate.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0:
            log.append({'step': step + 1, 'loss': float(loss.detach()), 'gradient_norm': float(norm)})
            print(json.dumps(log[-1]), flush=True)
    gate.eval()
    if any(p.grad is not None for p in model.parameters()) or any(sha(p) != d for p, d in protected.items()):
        raise ValueError('Protected input changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT / 'gate.pt'
    torch.save({'format': 'deepseek-spora-number-invariant-intent-v1',
                'state': {k: v.detach().cpu() for k, v in gate.state_dict().items()},
                'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA,
                'layers': [5, 17], 'threshold': 0.5, 'parent_sha256': protected[BINARY]}, checkpoint)
    print('gate saved', flush=True)
    results = {}
    for name, rows in eval_sets.items():
        x, m = extract(model, tokenizer, [hide_numbers(r['question']) for r in rows], batch_size=1)
        results[name] = evaluate(rows, scores(old, x, m), scores(gate, x, m))
        print(f'{name} complete', flush=True)
    if any(sha(p) != d for p, d in protected.items()):
        raise ValueError('Protected input changed')
    report = {'status': 'hardening_development_only', 'revision': REVISION, 'weight_sha256': WEIGHT_SHA,
              'individual_sha256': INDIVIDUAL_SHA, 'parent_gate_sha256': protected[BINARY],
              'gate_sha256': sha(checkpoint), 'steps': STEPS, 'train_unique': len(unique), 'log': log,
              'results': results, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['DeepSeek and original SPORA individual are frozen; only a small intent gate is trained.',
                              'Hard examples are hand-authored and repeated; held sets were used in prior development.',
                              'OASST prefix is synthetic; the source was not used for training.',
                              'Finite symbolic format guard and 0.5 threshold remain unchanged.',
                              'No final independent evaluation, actual memory generation, or open-domain readiness.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: v['summary'] for k, v in results.items()}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
