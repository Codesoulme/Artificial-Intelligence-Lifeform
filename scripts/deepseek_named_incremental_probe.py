"""Research-only online ridge updates for supervised named-address bindings."""
import copy
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
from scripts.deepseek_named_binding_migrate import OUTPUT as SOURCE
from scripts.deepseek_named_binding_probe import HELD_TEMPLATES, KEYS, NAMES, TRAIN_TEMPLATES, UNKNOWN, pooled
from scripts.deepseek_named_individual_roundtrip import OUTPUT as AUDIT_SOURCE
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_read_strength_probe import ScaledReadout
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.open_base_trajectory_comparison import numeric_correct
from spora.current_memory_status import current_memory_status
from spora.growing import load, query
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256
from spora.open_base_memory import memory_hooks
from spora.state_codec import flatten, restore

OUTPUT = ROOT / 'runs/deepseek-named-incremental-20261110'
NEW_KEYS = (0, 1)
NEW_VALUES = (8, 12)
RIDGE = 0.1


def ridge_statistics(features, targets, ridge=RIDGE):
    if (features.ndim != 3 or targets.ndim != 3 or features.shape[:2] != targets.shape[:2]
            or features.shape[1] != 2 or not features.shape[0] or ridge <= 0 or features.device != targets.device
            or not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(targets).all())):
        raise ValueError('Invalid ridge samples')
    weights, precision = [], []
    for site in range(2):
        x, y = features[:, site].double(), targets[:, site].double()
        gram = x @ x.T + ridge * torch.eye(len(x), dtype=torch.float64, device=x.device)
        solved = torch.linalg.solve(gram, x)
        precision.append((torch.eye(x.shape[1], dtype=torch.float64, device=x.device) - x.T @ solved) / ridge)
        weights.append(x.T @ torch.linalg.solve(gram, y))
    return torch.stack(weights), torch.stack(precision)


def ridge_update(weights, precision, features, targets):
    if (weights.ndim != 3 or precision.shape != (2, weights.shape[1], weights.shape[1])
            or features.shape != (2, weights.shape[1]) or targets.shape != (2, weights.shape[2])
            or any(not bool(torch.isfinite(v).all()) for v in (weights, precision, features, targets))):
        raise ValueError('Invalid incremental ridge state')
    result_w, result_p = [], []
    for site in range(2):
        x, y, p, w = features[site].double(), targets[site].double(), precision[site], weights[site]
        px = p @ x
        denominator = 1 + x @ px
        if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
            raise ValueError('Invalid ridge update denominator')
        gain = px / denominator
        new_w = w + torch.outer(gain, y - x @ w)
        new_p = p - torch.outer(px, gain)
        result_w.append(new_w)
        result_p.append((new_p + new_p.T) * 0.5)
    return torch.stack(result_w), torch.stack(result_p)


def score(binding, model, features, config_digest):
    with torch.inference_mode():
        result = binding(features, model, WEIGHT_SHA, config_digest)
    return result


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    individual, source_report = SOURCE / 'individual.pt', SOURCE / 'result.json'
    audit, audit_report, selector_path = AUDIT_SOURCE / 'features.pt', AUDIT_SOURCE / 'result.json', SELECTOR
    protected = {p: sha(p) for p in (MODEL / 'model.safetensors', INDIVIDUAL, individual, source_report,
                                       audit, audit_report, selector_path)}
    if protected[MODEL / 'model.safetensors'] != WEIGHT_SHA or protected[INDIVIDUAL] != INDIVIDUAL_SHA:
        raise ValueError('Protected model mismatch')
    config_digest = feature_files_sha256(MODEL)
    with source_report.open(encoding='utf-8') as stream:
        prior = json.load(stream)
    with audit_report.open(encoding='utf-8') as stream:
        prior_audit = json.load(stream)
    if (prior['saved_individual_sha256'] != protected[individual] or prior['feature_config_sha256'] != config_digest
            or prior['individual_sha256'] != INDIVIDUAL_SHA or prior['weight_sha256'] != WEIGHT_SHA
            or prior_audit['audit_features_sha256'] != protected[audit]):
        raise ValueError('Individual or encoder fingerprint mismatch')
    audit_data = torch.load(audit, map_location='cpu', weights_only=True)
    selector = torch.load(selector_path, map_location='cpu', weights_only=True)
    if (selector.get('format') != 'deepseek-spora-question-pooled-finite-address-v1'
            or selector.get('weight_sha256') != WEIGHT_SHA or selector.get('individual_sha256') != INDIVIDUAL_SHA):
        raise ValueError('Adapter provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    source_model, source_state, _ = load(individual, 'cpu')
    source_digest = frozen_digest(source_model, source_state)
    model, state = copy.deepcopy(source_model), copy.deepcopy(source_state)
    binding = model.named_address_binding
    previous = score(binding, model, audit_data['features'], config_digest)
    if (not torch.equal(previous['indices'], audit_data['indices'])
            or not torch.equal(previous['accepted'], audit_data['accepted'])):
        raise ValueError('Previous binding predictions differ')
    if any(current_memory_status(model, state, key)['status'] != 'unknown' for key in NEW_KEYS):
        raise ValueError('New slots must be unused')
    initial = [(name, key, template.format(name=name)) for name, key in zip(NAMES, KEYS)
               for template in TRAIN_TEMPLATES]
    new_training = [(name, key, template.format(name=name)) for name, key in zip(UNKNOWN[:2], NEW_KEYS)
                    for template in TRAIN_TEMPLATES[:2]]
    if set(q for _, _, q in initial + new_training) & set(
            template.format(name=name) for name in NAMES + UNKNOWN for template in HELD_TEMPLATES):
        raise ValueError('Training and evaluation text overlap')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    features, masks = extract(base, tokenizer, [q for _, _, q in initial + new_training], batch_size=1)
    vectors = pooled(features, masks)
    bases = torch.stack([F.normalize(site.current_basis.detach().float(), dim=-1) for site in model.memories])
    initial_targets = torch.stack([bases[:, key] for _, key, _ in initial])
    weights, precision = ridge_statistics(vectors[:len(initial)], initial_targets)
    baseline_difference = float((weights.float() - binding.weights).abs().max())
    if baseline_difference > 1e-4:
        raise ValueError(f'Reconstructed binding differs: {baseline_difference}')
    new_targets = torch.stack([bases[:, key] for _, key, _ in new_training])
    for feature, target in zip(vectors[len(initial):], new_targets):
        weights, precision = ridge_update(weights, precision, feature, target)
    oracle_weights, _ = ridge_statistics(vectors, torch.cat((initial_targets, new_targets)))
    oracle_difference = float((weights - oracle_weights).abs().max())
    if oracle_difference > 1e-8:
        raise ValueError(f'Incremental update differs from batch fit: {oracle_difference}')
    with torch.no_grad():
        binding.weights.copy_(weights.float())
    events = []
    for key, value in zip(NEW_KEYS, NEW_VALUES):
        state, decision = observe_sourced(model, state, key, value, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('New observation not written')
        events.append({'key': key, 'value': value, 'action': decision['action']})
    expected = torch.tensor(NEW_VALUES, dtype=torch.long)
    keys = torch.tensor(NEW_KEYS, dtype=torch.long)
    with torch.inference_mode():
        actual = query(model, state, keys, torch.zeros_like(keys), torch.zeros_like(keys)).argmax(-1)
    if not torch.equal(actual, expected):
        raise ValueError('New values do not match native memory')
    updated = score(binding, model, audit_data['features'], config_digest)
    expected_keys = [key for key in KEYS for _ in HELD_TEMPLATES] + [key for key in NEW_KEYS for _ in HELD_TEMPLATES] + [None] * 4
    rows = []
    names = [name for name in NAMES + UNKNOWN for _ in HELD_TEMPLATES]
    questions = [template.format(name=name) for name in NAMES + UNKNOWN for template in HELD_TEMPLATES]
    for index, (name, question, key) in enumerate(zip(names, questions, expected_keys)):
        row = {'name': name, 'question': question, 'expected_key': key,
               'old_indices': previous['indices'][index].tolist(), 'old_accepted': bool(previous['accepted'][index]),
               'new_indices': updated['indices'][index].tolist(), 'new_accepted': bool(updated['accepted'][index]),
               'new_scores': updated['scores'][index].tolist()}
        row['new_correct'] = key is not None and row['new_accepted'] and row['new_indices'] == [key, key]
        rows.append(row)
    adapter = ScaledReadout(base.config.hidden_size, model.config.memory_width, 0.5).to('cuda').eval()
    adapter.load_state_dict(selector['readout'], strict=True)
    with torch.inference_mode():
        for row in rows[16:20]:
            if not row['new_accepted']:
                continue
            key = row['new_indices'][0]
            addresses = [bases[site, key:key+1].to('cuda') for site in range(2)]
            inputs = closed_think_inputs(tokenizer, row['question'], 'cuda')
            with memory_hooks([base.model.layers[5], base.model.layers[17]], adapter, state,
                              current_only=True, addresses=addresses):
                generated = base.generate(**inputs, max_new_tokens=8, do_sample=False,
                                          pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            row['answer'] = text
            row['answer_correct'] = row['new_correct'] and numeric_correct(text, NEW_VALUES[NEW_KEYS.index(row['expected_key'])])
    if (frozen_digest(source_model, source_state) != source_digest or feature_files_sha256(MODEL) != config_digest
            or any(sha(p) != d for p, d in protected.items())):
        raise ValueError('Protected source changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT / 'research_individual.pt'
    schema, tensors = flatten(model, state)
    with checkpoint.open('xb') as stream:
        torch.save({'format': 'named-online-ridge-research-v1', 'schema': schema,
                    'tensors': {k: v.detach().cpu().clone() for k, v in tensors.items()},
                    'precision': precision, 'ridge': RIDGE, 'samples': len(initial) + len(new_training)}, stream)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    loaded_model, loaded_state = restore(saved['schema'], saved['tensors'])
    loaded = score(loaded_model.named_address_binding, loaded_model, audit_data['features'], config_digest)
    if (not torch.equal(loaded['indices'], updated['indices']) or not torch.equal(loaded['accepted'], updated['accepted'])
            or not torch.equal(saved['precision'], precision) or frozen_digest(loaded_model, loaded_state) != frozen_digest(model, state)):
        raise ValueError('Saved research state differs')
    summary = {'old_held': 16, 'old_address_correct_before': sum(bool(previous['accepted'][i]) and previous['indices'][i].tolist() == [expected_keys[i]]*2 for i in range(16)),
               'old_address_correct_after': sum(r['new_correct'] for r in rows[:16]),
               'new_held': 4, 'new_address_correct': sum(r['new_correct'] for r in rows[16:20]),
               'new_answer_correct': sum(r.get('answer_correct', False) for r in rows[16:20]),
               'unseen_held': 4, 'unseen_false_accept': sum(r['new_accepted'] for r in rows[20:])}
    report = {'status': 'supervised_incremental_binding_development_only', 'summary': summary, 'rows': rows,
              'events': events, 'reconstructed_max_difference': baseline_difference,
              'incremental_vs_batch_max_difference': oracle_difference, 'precision_bytes': precision.numel() * precision.element_size(),
              'checkpoint_bytes': checkpoint.stat().st_size, 'checkpoint_sha256': sha(checkpoint),
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Two new names and slots are explicitly supervised; no automatic entity allocation or source validation.',
                              'Old training features were reconstructed once; updates use retained inverse covariance, not raw historical text.',
                              'All evaluation names and templates were used in previous development; no independent open-world validation.',
                              'Fixed threshold was not recalibrated after update; false acceptance estimates are tiny-sample.',
                              'Custom research checkpoint stores state and inverse covariance together, but is not integrated into the standard codec.',
                              'RLS inverse covariance is dense and its memory cost grows quadratically with feature width.',
                              'Only same-process checkpoint reload, not crash recovery, is tested.',
                              'Memory values remain finite integers and query safety is not evaluated.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'summary': summary, 'precision_bytes': report['precision_bytes'],
                      'checkpoint_bytes': report['checkpoint_bytes'], 'oracle_difference': oracle_difference}), flush=True)


if __name__ == '__main__':
    run()
