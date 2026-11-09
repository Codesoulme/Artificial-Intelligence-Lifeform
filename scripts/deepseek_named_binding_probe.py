"""Experimental neural binding of Chinese entity names to SPORA address vectors."""
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

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_chat_interference import SELECTOR
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_read_strength_probe import ScaledReadout
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.open_base_trajectory_comparison import numeric_correct
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import load
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest
from spora.open_base_memory import memory_hooks

OUTPUT = ROOT / 'runs/deepseek-named-binding-20261110'
NAMES = ('小林', '阿宁', '小雨', '阿哲', '小梅', '阿远', '小岚', '阿明')
UNKNOWN = ('阿松', '小雪', '阿禾', '小青')
KEYS = (6, 22, 35, 51, 16, 27, 42, 10)
VALUES = (3, 13, 7, 9, 11, 4, 2, 15)
TRAIN_TEMPLATES = ('{name}当前的数字是多少？', '请查询{name}现在的数值。',
                   '告诉我{name}记录的数字。', '{name}目前存的数值是几？')
HELD_TEMPLATES = ('{name}眼下记着什么数？', '我想知道{name}的最新数字。')


def pooled(features, masks):
    if features.ndim != 4 or masks.shape != (features.shape[0], features.shape[2]):
        raise ValueError('Bad feature shapes')
    mask = masks[:, None, :, None].to(features.dtype)
    vectors = (features.float() * mask).sum(2) / mask.sum(2).clamp_min(1)
    return F.normalize(vectors, dim=-1)


def fit_binding(features, targets, ridge=0.1):
    if features.ndim != 3 or targets.ndim != 3 or features.shape[:2] != targets.shape[:2] or ridge <= 0:
        raise ValueError('Expected matching [examples, sites, width] tensors and positive ridge')
    weights = []
    for site in range(features.shape[1]):
        x, y = features[:, site], targets[:, site]
        gram = x @ x.T + ridge * torch.eye(len(x), dtype=x.dtype, device=x.device)
        weights.append(x.T @ torch.linalg.solve(gram, y))
    return torch.stack(weights)


def predict(features, weights, bases):
    queries = F.normalize(torch.einsum('bsh,shd->bsd', features, weights), dim=-1)
    similarities = torch.einsum('bsd,skd->bsk', queries, bases)
    scores, indices = similarities.max(-1)
    return indices, scores


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    protected = {p: sha(p) for p in (MODEL / 'model.safetensors', INDIVIDUAL, SELECTOR)}
    if protected[MODEL / 'model.safetensors'] != WEIGHT_SHA or protected[INDIVIDUAL] != INDIVIDUAL_SHA:
        raise ValueError('Protected model mismatch')
    selector = torch.load(SELECTOR, map_location='cpu', weights_only=True)
    if (selector.get('format') != 'deepseek-spora-question-pooled-finite-address-v1'
            or selector.get('weight_sha256') != WEIGHT_SHA or selector.get('individual_sha256') != INDIVIDUAL_SHA
            or selector.get('layers') != [5, 17]):
        raise ValueError('Adapter provenance mismatch')
    if len(set(NAMES + UNKNOWN)) != len(NAMES) + len(UNKNOWN) or len(set(KEYS)) != len(KEYS):
        raise ValueError('Duplicate names or keys')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(organism, original)
    state = fresh_controlled_state(organism)
    events = []
    for key, value in zip(KEYS, VALUES):
        state, decision = observe_sourced(organism, state, key, value, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('Direct observation was not written')
        events.append({'key': key, 'value': value, 'action': decision['action']})
    state_before = frozen_digest(organism, state)
    train = [(name, key, t.format(name=name)) for name, key in zip(NAMES, KEYS) for t in TRAIN_TEMPLATES]
    held = [(name, key, t.format(name=name)) for name, key in zip(NAMES, KEYS) for t in HELD_TEMPLATES]
    unknown = [(name, None, t.format(name=name)) for name in UNKNOWN for t in HELD_TEMPLATES]
    if set(q for _, _, q in train) & set(q for _, _, q in held + unknown):
        raise ValueError('Training and evaluation text overlap')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    bases = torch.stack([F.normalize(site.current_basis.detach().float(), dim=-1) for site in organism.memories])
    train_features, train_masks = extract(model, tokenizer, [r[2] for r in train], batch_size=1)
    held_features, held_masks = extract(model, tokenizer, [r[2] for r in held + unknown], batch_size=1)
    x_train, x_test = pooled(train_features, train_masks), pooled(held_features, held_masks)
    targets = torch.stack([bases[:, key] for _, key, _ in train])
    weights = fit_binding(x_train, targets)
    train_predicted, train_scores = predict(x_train, weights, bases)
    test_predicted, test_scores = predict(x_test, weights, bases)
    train_labels = torch.tensor([key for _, key, _ in train])
    train_correct = (train_predicted == train_labels[:, None]).all(-1)
    if not bool(train_correct.all()):
        print('Training bindings are not all recovered; reporting without hiding failures.', flush=True)
    threshold = float(train_scores.min()) - 1e-6
    adapter = ScaledReadout(model.config.hidden_size, organism.config.memory_width, 0.5).to('cuda').eval()
    adapter.load_state_dict(selector['readout'], strict=True)
    rows = []
    with torch.inference_mode():
        for i, (name, key, question) in enumerate(held + unknown):
            indices, scores = test_predicted[i].tolist(), test_scores[i].tolist()
            accepted = bool(min(scores) >= threshold and indices[0] == indices[1])
            row = {'name': name, 'question': question, 'expected_key': key, 'predicted': indices,
                   'similarities': scores, 'accepted': accepted,
                   'address_correct': key is not None and accepted and indices == [key, key]}
            if key is not None and accepted:
                chosen = indices[0]
                addresses = [bases[j, chosen:chosen+1].to('cuda') for j in range(2)]
                inputs = closed_think_inputs(tokenizer, question, 'cuda')
                with memory_hooks([model.model.layers[5], model.model.layers[17]], adapter, state,
                                  current_only=True, addresses=addresses):
                    generated = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                               pad_token_id=tokenizer.eos_token_id)
                tokens = generated[0, inputs['input_ids'].shape[1]:].tolist()
                text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                expected_value = VALUES[KEYS.index(key)]
                row['answer'] = text
                row['answer_correct'] = row['address_correct'] and numeric_correct(text, expected_value)
                row['ended_with_eos'] = tokenizer.eos_token_id in tokens
            rows.append(row)
            if (i + 1) % 8 == 0:
                print(f'checked {i + 1}/{len(held) + len(unknown)}', flush=True)
    if (frozen_digest(organism, original) != before or frozen_digest(organism, state) != state_before
            or any(sha(p) != d for p, d in protected.items())):
        raise ValueError('Protected input changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT / 'binding.pt'
    torch.save({'format': 'deepseek-spora-named-binding-experiment-v1', 'weights': weights,
                'threshold': threshold, 'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA,
                'sites_hidden_indices': [6, 18], 'ridge': 0.1}, checkpoint)
    summary = {'train_examples': len(train), 'train_correct_both': int(train_correct.sum()),
               'held_examples': len(held), 'held_address_correct': sum(r['address_correct'] for r in rows if r['expected_key'] is not None),
               'held_answer_correct': sum(r.get('answer_correct', False) for r in rows),
               'unknown_examples': len(unknown), 'unknown_false_accept': sum(r['accepted'] for r in rows if r['expected_key'] is None)}
    report = {'status': 'named_binding_development_only', 'weight_sha256': WEIGHT_SHA,
              'individual_sha256': INDIVIDUAL_SHA, 'selector_sha256': protected[SELECTOR],
              'binding_sha256': sha(checkpoint), 'threshold': threshold, 'events': events,
              'summary': summary, 'rows': rows, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Eight named entities and four unknown names, two held templates per name.',
                              'All names and address bindings are supervised; no unsupervised name creation.',
                              'Fixed arbitrary number values, not natural-language fact storage.',
                              'Predicted nearest address always belongs to the full finite basis; threshold estimated only on training samples.',
                              'Binding matrices contain learned associations, not a separate text lookup table; names and labels are retained here only for evaluation.',
                              'Adapter trained on numbered questions; ordinary language may not transfer.',
                              'Closed thinking and greedy eight-token answers; no real conversation or safe write routing.',
                              'No separate individuals, persistent binding reload, multi-turn, or crash recovery tested.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
