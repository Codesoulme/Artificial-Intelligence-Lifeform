"""Evaluate a restricted learned read route with a reloaded individual."""
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
from scripts.deepseek_named_binding_probe import KEYS, NAMES, UNKNOWN, pooled
from scripts.deepseek_named_span_persistence import CONTRACT
from scripts.deepseek_observation_value_probe import WORDS, pool
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_preference_intent_probe import FORMAT
from scripts.deepseek_semantic_value_pilot import word_correct
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.current_memory_status import current_memory_status
from spora.integrated_checkpoint import load_integrated
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-preference-routing-composite-20261110.json'
GATE = ROOT / 'runs/deepseek-preference-intent-expanded-20261110'


def build_cases():
    return [
        {'group': 'current', 'text': f'{NAMES[0]}喜欢的饮品是什么？', 'key': KEYS[0], 'value': 6},
        {'group': 'current', 'text': f'请告诉我{NAMES[1]}现在最喜欢喝什么。', 'key': KEYS[1], 'value': 1},
        {'group': 'current', 'text': f'{NAMES[0]}最中意哪种饮料？', 'key': KEYS[0], 'value': 6},
        {'group': 'current', 'text': f'说说{NAMES[1]}偏好喝的东西。', 'key': KEYS[1], 'value': 1},
        {'group': 'negated', 'text': f'请不要回答{NAMES[0]}最爱喝什么。'},
        {'group': 'history', 'text': f'{NAMES[0]}以前喜欢喝什么？'},
        {'group': 'source', 'text': f'关于{NAMES[1]}饮料偏好的记录是谁提供的？'},
        {'group': 'derived', 'text': f'{NAMES[1]}最爱的饮料是不是水？'},
        {'group': 'multiple', 'text': f'{NAMES[0]}与{NAMES[1]}分别爱喝什么？'},
        {'group': 'write', 'text': f'把{NAMES[0]}喜欢的饮品更新为果汁。'},
        {'group': 'chat', 'text': '你好，请给我一个放松建议。'},
        {'group': 'missing', 'text': '现在最喜欢喝什么？'},
        {'group': 'unbound', 'text': f'{UNKNOWN[2]}偏爱的饮品是什么？'},
    ]


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = {'weight': MODEL / 'model.safetensors', 'original': INDIVIDUAL,
             'individual': CHECKPOINT / 'individual.pt', 'individual_report': CHECKPOINT / 'result.json',
             'locator': LOCATOR / 'locator.pt', 'locator_report': LOCATOR / 'result.json',
             'gate': GATE / 'gate.pt', 'gate_report': GATE / 'result.json'}
    before = {name: sha(path) for name, path in files.items()}
    if before['weight'] != WEIGHT_SHA or before['original'] != INDIVIDUAL_SHA:
        raise ValueError('Protected input mismatch')
    reports = {}
    for name in ('individual', 'locator', 'gate'):
        with files[name + '_report'].open(encoding='utf-8') as stream:
            reports[name] = json.load(stream)
    if (reports['individual']['checkpoint_sha256'] != before['individual']
            or reports['locator']['locator_sha256'] != before['locator']
            or reports['gate']['gate_sha256'] != before['gate']):
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
    integrated, organism, state = load_integrated(files['individual'], base, MODEL)
    integrated.eval()
    digest = frozen_digest(organism, state)
    cases = build_cases()
    features, masks = extract(base, tokenizer, [r['text'] for r in cases], batch_size=1)
    token_vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
    locator = nn.Linear(token_vectors.shape[-1], 1).eval()
    locator.load_state_dict(locator_data['state'], strict=True)
    gate = nn.Linear(gate_data['hidden_width'], 1).eval()
    gate.load_state_dict(gate_data['state'], strict=True)
    with torch.no_grad():
        location_scores = locator(token_vectors).squeeze(-1).sigmoid()
        intent_scores = gate(pool(features, masks)).squeeze(-1)
    bases = [F.normalize(site.current_basis.to('cuda').float(), dim=-1) for site in organism.memories]
    rows = []
    for i, case in enumerate(cases):
        mask, single = single_span_mask(location_scores[i], masks[i])
        decision = {'intent_logit': float(intent_scores[i]), 'intent_accepted': float(intent_scores[i]) >= threshold,
                    'single_span': single, 'key': None, 'status': None, 'route': 'base'}
        if decision['intent_accepted'] and single:
            vectors = pooled(features[i:i + 1], mask.unsqueeze(0))
            with torch.inference_mode():
                binding = organism.named_address_binding(vectors, organism, WEIGHT_SHA, config_sha,
                                                         feature_contract=CONTRACT)
            if bool(binding['accepted'][0]):
                decision['key'] = int(binding['indices'][0, 0])
                decision['status'] = current_memory_status(organism, state, decision['key'])['status']
                if decision['status'] == 'known':
                    decision['route'] = 'memory'
        inputs = closed_think_inputs(tokenizer, case['text'], 'cuda')
        with torch.no_grad():
            baseline = integrated.generate(**inputs, max_new_tokens=16, do_sample=False,
                                           pad_token_id=tokenizer.eos_token_id)
            if decision['route'] == 'memory':
                addresses = [basis[decision['key']:decision['key'] + 1] for basis in bases]
                routed = integrated.generate(**inputs, memory_state=state, memory_addresses=addresses,
                                             current_only=True, max_new_tokens=16, do_sample=False,
                                             pad_token_id=tokenizer.eos_token_id)
            else:
                routed = integrated.generate(**inputs, max_new_tokens=16, do_sample=False,
                                             pad_token_id=tokenizer.eos_token_id)
        length = inputs['input_ids'].shape[1]
        base_tokens, routed_tokens = baseline[0, length:].tolist(), routed[0, length:].tolist()
        answer = tokenizer.decode(routed_tokens, skip_special_tokens=True).strip()
        rows.append({**case, **decision, 'answer': answer,
                     'baseline_answer': tokenizer.decode(base_tokens, skip_special_tokens=True).strip(),
                     'same_tokens_as_base': base_tokens == routed_tokens,
                     'route_correct': (decision['route'] == 'memory' and decision['key'] == case['key']
                                       if case['group'] == 'current' else decision['route'] == 'base'),
                     'answer_correct': (decision['route'] == 'memory' and decision['key'] == case['key']
                                        and word_correct(answer, WORDS[case['value']])) if case['group'] == 'current' else None})
        print(f'checked {i + 1}/{len(cases)}', flush=True)
    if (any(sha(path) != before[name] for name, path in files.items()) or feature_files_sha256(MODEL) != config_sha
            or frozen_digest(organism, state) != digest):
        raise ValueError('Protected input changed')
    current = [r for r in rows if r['group'] == 'current']
    negative = [r for r in rows if r['group'] != 'current']
    summary = {'current': len(current), 'current_route_correct': sum(r['route_correct'] for r in current),
               'current_answer_correct': sum(r['answer_correct'] for r in current), 'other': len(negative),
               'other_misrouted': sum(not r['route_correct'] for r in negative),
               'base_routes': sum(r['route'] == 'base' for r in rows),
               'base_routes_same_tokens': sum(r['route'] == 'base' and r['same_tokens_as_base'] for r in rows)}
    report = {'status': 'restricted_routing_development_only', 'summary': summary, 'rows': rows,
              'component_sha256': before, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Handwritten development examples, most names or expressions seen in prior experiments.',
                              'Only current finite drink preferences are supported; no general intent or arbitrary facts.',
                              'Base fallback may hallucinate and is not a validated unknown-memory response.',
                              'No writes, source reliability, history answering, multi-turn references or safety guarantee.',
                              'Single process and greedy 16-token answers; no latency or chat quality benchmark.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    run()
