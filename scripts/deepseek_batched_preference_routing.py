"""Evaluate the existing restricted preference route with per-row memory masks."""
import json
import math
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

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_cluener_locator_train import OUTPUT as LOCATOR
from scripts.deepseek_integrated_checkpoint_probe import OUTPUT as CHECKPOINT
from scripts.deepseek_learned_name_span import single_span_mask
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_named_binding_probe import pooled
from scripts.deepseek_named_span_persistence import CONTRACT
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_preference_intent_probe import FORMAT
from scripts.deepseek_preference_routing_composite import GATE, build_cases
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.current_memory_status import current_memory_status
from spora.integrated_checkpoint import load_integrated
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-batched-preference-routing-full-20261110.json'


def left_pad(items, pad_id):
    if not items or type(pad_id) is not int:
        raise ValueError('Nonempty inputs and padding token required')
    length = max(item['input_ids'].shape[1] for item in items)
    result = {}
    for key in ('input_ids', 'attention_mask'):
        value = pad_id if key == 'input_ids' else 0
        result[key] = torch.cat([F.pad(item[key], (length - item[key].shape[1], 0), value=value)
                                 for item in items], dim=0)
    return result


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = {'weight': MODEL / 'model.safetensors', 'original': INDIVIDUAL,
             'individual': CHECKPOINT / 'individual.pt', 'individual_report': CHECKPOINT / 'result.json',
             'locator': LOCATOR / 'locator.pt', 'locator_report': LOCATOR / 'result.json',
             'gate': GATE / 'gate.pt', 'gate_report': GATE / 'result.json'}
    before = {name: sha(path) for name, path in files.items()}
    if before['weight'] != WEIGHT_SHA or before['original'] != INDIVIDUAL_SHA:
        raise ValueError('Protected source mismatch')
    reports = {}
    for name in ('individual', 'locator', 'gate'):
        with files[name + '_report'].open(encoding='utf-8') as stream:
            reports[name] = json.load(stream)
    if any(reports[name][field] != before[name] for name, field in
           (('individual', 'checkpoint_sha256'), ('locator', 'locator_sha256'), ('gate', 'gate_sha256'))):
        raise ValueError('Component checkpoint mismatch')
    config_sha = feature_files_sha256(MODEL)
    locator_data = torch.load(files['locator'], map_location='cpu', weights_only=True)
    gate_data = torch.load(files['gate'], map_location='cpu', weights_only=True)
    threshold = gate_data.get('threshold_logit')
    if (locator_data.get('format') != 'deepseek-name-span-locator-development-v1'
            or locator_data.get('weight_sha256') != WEIGHT_SHA or locator_data.get('feature_config_sha256') != config_sha
            or locator_data.get('layers') != [6, 18] or locator_data.get('threshold') != 0.5
            or gate_data.get('format') != FORMAT or gate_data.get('weight_sha256') != WEIGHT_SHA
            or gate_data.get('individual_sha256') != INDIVIDUAL_SHA or gate_data.get('feature_config_sha256') != config_sha
            or gate_data.get('hidden_indices') != [6, 18] or type(threshold) is not float
            or not math.isfinite(threshold) or reports['gate']['threshold_logit'] != threshold):
        raise ValueError('Component provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    model, organism, state = load_integrated(files['individual'], base, MODEL)
    model.eval()
    digest = frozen_digest(organism, state)
    cases = build_cases()
    features, masks = extract(base, tokenizer, [row['text'] for row in cases], batch_size=1)
    token_vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
    locator = nn.Linear(token_vectors.shape[-1], 1).eval()
    locator.load_state_dict(locator_data['state'], strict=True)
    gate = nn.Linear(gate_data['hidden_width'], 1).eval()
    gate.load_state_dict(gate_data['state'], strict=True)
    with torch.no_grad():
        location_scores = locator(token_vectors).squeeze(-1).sigmoid()
        intent_scores = gate(pool(features, masks)).squeeze(-1)
    if not bool(torch.isfinite(location_scores).all()) or not bool(torch.isfinite(intent_scores).all()):
        raise ValueError('Nonfinite routing scores')
    bases = [F.normalize(site.current_basis.to('cuda').float(), dim=-1) for site in organism.memories]
    rows, route_keys, read_mask = [], [], []
    for i, case in enumerate(cases):
        mask, single = single_span_mask(location_scores[i], masks[i])
        key, status = None, None
        if float(intent_scores[i]) >= threshold and single:
            with torch.inference_mode():
                binding = organism.named_address_binding(pooled(features[i:i + 1], mask.unsqueeze(0)), organism,
                                                         WEIGHT_SHA, config_sha, feature_contract=CONTRACT)
            if bool(binding['accepted'][0]):
                key = int(binding['indices'][0, 0])
                status = current_memory_status(organism, state, key)['status']
        enabled = key is not None and status == 'known'
        route_keys.append(key if enabled else 0)
        read_mask.append(enabled)
        rows.append({'group': case['group'], 'key': key, 'status': status, 'read': enabled,
                     'expected_key': case.get('key'), 'intent_accepted': float(intent_scores[i]) >= threshold,
                     'single_span': bool(single)})
    # Keep GPU memory bounded while preserving input order.
    comparisons = []
    for left in range(0, len(cases), 2):
        batch_cases = cases[left:left + 2]
        batch = left_pad([closed_think_inputs(tokenizer, row['text'], 'cuda') for row in batch_cases], tokenizer.pad_token_id)
        addresses = [basis[route_keys[left:left + len(batch_cases)]] for basis in bases]
        selected = torch.tensor(read_mask[left:left + len(batch_cases)], dtype=torch.bool)
        with torch.inference_mode():
            plain = model(**batch, use_cache=False).logits[:, -1].float().cpu()
            routed = model(**batch, memory_state=state, memory_addresses=addresses, memory_read_mask=selected,
                           current_only=True, use_cache=False).logits[:, -1].float().cpu()
        if not bool(torch.isfinite(plain).all() and torch.isfinite(routed).all()):
            raise ValueError('Nonfinite output logits')
        for offset, enabled in enumerate(selected.tolist()):
            diff = float((plain[offset] - routed[offset]).abs().max())
            comparisons.append(diff)
            if not enabled and diff != 0:
                raise ValueError('Rejected route changed base output')
        print(f'checked {min(left + 2, len(cases))}/{len(cases)}', flush=True)
    if (any(sha(path) != before[name] for name, path in files.items()) or feature_files_sha256(MODEL) != config_sha
            or frozen_digest(organism, state) != digest):
        raise ValueError('Protected input changed')
    for row, diff in zip(rows, comparisons):
        row['max_logit_difference_from_base'] = diff
        row['route_correct'] = ((row['read'] and row['key'] == row['expected_key']) if row['group'] == 'current'
                                else not row['read'])
    report = {'status': 'batched_restricted_routing_development_only', 'rows': rows,
              'summary': {'examples': len(rows), 'correct_routes': sum(row['route_correct'] for row in rows),
                          'current': sum(row['group'] == 'current' for row in rows),
                          'other': sum(row['group'] != 'current' for row in rows),
                          'other_misrouted': sum(row['read'] for row in rows if row['group'] != 'current'),
                          'rejected_max_logit_difference': max((row['max_logit_difference_from_base']
                                                                 for row in rows if not row['read']), default=None)},
              'component_sha256': before, 'feature_config_sha256': config_sha,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Existing handwritten development cases and finite drink preference task only.',
                              'Gate, span and identity are experimental; no authenticated callers or arbitrary facts.',
                              'Caller-selected placeholder address is still read computationally for masked rows.',
                              'No free generation, truth, writes, concurrency, or open-domain safety evaluated.',
                              'Base fallback can hallucinate; identical logits are not a reliable unknown answer.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    run()
