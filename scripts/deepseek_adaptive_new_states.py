"""Exercise gated, learned-address reads against new controlled SPORA states."""
import json
import random
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
from scripts.deepseek_adaptive_read_probe import GATE
from scripts.deepseek_chat_interference import SELECTOR
from scripts.deepseek_composite_probe import initial_route
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_memory_intent import IntentGate
from scripts.deepseek_pooled_address import PooledAddress, extract
from scripts.deepseek_read_strength_probe import ScaledReadout
from scripts.deepseek_route_calibration import NEW_CASES
from scripts.deepseek_semantic_gate import hide_numbers
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.open_base_trajectory_comparison import numeric_correct
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.current_memory_status import current_memory_status
from spora.finite_question_guard import valid_finite_question_form
from spora.growing import load
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest
from spora.open_base_memory import memory_hooks

OUTPUT = ROOT / 'runs/deepseek-adaptive-new-states-20261110.json'
SEEDS = (4629, 4630)
KEYS = (6, 22, 35, 51, 16, 27, 42, 10)


def make_state(model, seed):
    rng = random.Random(seed)
    state = fresh_controlled_state(model)
    expected, events = {}, []
    interference = [k for k in range(model.config.key_count) if k not in KEYS]
    rng.shuffle(interference)
    for key in list(KEYS[:4]) + list(KEYS[:2]) + interference[:8]:
        value = rng.randrange(model.config.unknown)
        if expected.get(key) == value:
            value = (value + 1) % model.config.unknown
        state, decision = observe_sourced(model, state, key, value, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('Expected direct observation to update current state')
        expected[key] = value
        events.append({'key': key, 'value': value, 'action': decision['action']})
    return state, expected, events


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    protected = {p: sha(p) for p in (MODEL / 'model.safetensors', INDIVIDUAL, SELECTOR, GATE)}
    if protected[MODEL / 'model.safetensors'] != WEIGHT_SHA or protected[INDIVIDUAL] != INDIVIDUAL_SHA:
        raise ValueError('Protected model mismatch')
    selector_data = torch.load(SELECTOR, map_location='cpu', weights_only=True)
    gate_data = torch.load(GATE, map_location='cpu', weights_only=True)
    if (selector_data.get('format') != 'deepseek-spora-question-pooled-finite-address-v1'
            or gate_data.get('format') != 'deepseek-spora-number-invariant-intent-v1'
            or any(d.get('weight_sha256') != WEIGHT_SHA or d.get('individual_sha256') != INDIVIDUAL_SHA
                   or d.get('layers') != [5, 17] for d in (selector_data, gate_data))
            or gate_data.get('threshold') != 0.5):
        raise ValueError('Component provenance mismatch')
    if len(NEW_CASES) < len(KEYS) or any(not NEW_CASES[i][0] or str(key) not in NEW_CASES[i][2]
                                         for i, key in enumerate(KEYS)):
        raise ValueError('Positive case/key metadata mismatch')
    cases = [{'category': 'known' if i < 4 else 'unknown', 'question': NEW_CASES[i][2],
              'key': key} for i, key in enumerate(KEYS)]
    cases.extend({'category': 'negative', 'question': q, 'key': None} for label, _, q in NEW_CASES if not label)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(organism, original)
    trajectories = [(seed, *make_state(organism, seed)) for seed in SEEDS]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    selector = PooledAddress(model.config.hidden_size, organism.config.memory_width).to('cuda').eval()
    selector.load_state_dict(selector_data['selector'], strict=True)
    gate = IntentGate(model.config.hidden_size).to('cuda').eval()
    gate.load_state_dict(gate_data['state'], strict=True)
    adapter = ScaledReadout(model.config.hidden_size, organism.config.memory_width, 0.5).to('cuda').eval()
    adapter.load_state_dict(selector_data['readout'], strict=True)
    bases = [F.normalize(site.current_basis.to('cuda').float(), dim=-1) for site in organism.memories]
    questions = [r['question'] for r in cases]
    raw, raw_masks = extract(model, tokenizer, questions, batch_size=1)
    masked, masked_masks = extract(model, tokenizer, [hide_numbers(q) for q in questions], batch_size=1)
    decisions = []
    with torch.inference_mode():
        for index, case in enumerate(cases):
            score = float(gate(masked[index:index+1].to('cuda'), masked_masks[index:index+1].to('cuda')).sigmoid()[0])
            addresses = selector(raw[index:index+1].to('cuda'), raw_masks[index:index+1].to('cuda'))
            predicted = [int((addresses[i] @ bases[i].T).argmax(-1)) for i in range(2)]
            route = initial_route(valid_finite_question_form(case['question']), score, predicted)
            decisions.append({'score': score, 'predicted': predicted, 'initial_route': route})
    reports = []
    with torch.inference_mode():
        for seed, state, truth, events in trajectories:
            state_before = frozen_digest(organism, state)
            rows = []
            for case, decision in zip(cases, decisions):
                expected_route = {'known': 'answer', 'unknown': 'unknown', 'negative': 'defer'}[case['category']]
                route = decision['initial_route']
                presence = None
                if route == 'inspect':
                    presence = current_memory_status(organism, state, decision['predicted'][0])
                    route = 'unknown' if presence['status'] == 'unknown' else 'answer'
                row = dict(case, **decision, route=route, presence=presence,
                           expected_route=expected_route,
                           route_correct=route == expected_route and (case['key'] is None or
                               decision['predicted'] == [case['key']] * 2), answers={})
                if route == 'answer' and case['category'] == 'known':
                    row['expected_value'] = truth[case['key']]
                    key = decision['predicted'][0]
                    snapped = [basis[key:key+1] for basis in bases]
                    inputs = closed_think_inputs(tokenizer, case['question'], 'cuda')
                    for strength in (0.5, 1.0):
                        adapter.strength = strength
                        with memory_hooks([model.model.layers[5], model.model.layers[17]], adapter, state,
                                          current_only=True, addresses=snapped):
                            generated = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                                       pad_token_id=tokenizer.eos_token_id)
                        tokens = generated[0, inputs['input_ids'].shape[1]:].tolist()
                        text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                        row['answers'][str(strength)] = {'text': text,
                            'correct': row['route_correct'] and numeric_correct(text, truth[case['key']]),
                            'ended_with_eos': tokenizer.eos_token_id in tokens}
                rows.append(row)
            if frozen_digest(organism, state) != state_before:
                raise ValueError('Probe mutated fresh state')
            groups = {}
            for category in ('known', 'unknown', 'negative'):
                group = [r for r in rows if r['category'] == category]
                groups[category] = {'examples': len(group), 'route_correct': sum(r['route_correct'] for r in group),
                                    'answers_correct_0.5': sum(r['answers'].get('0.5', {}).get('correct', False) for r in group),
                                    'answers_correct_1.0': sum(r['answers'].get('1.0', {}).get('correct', False) for r in group)}
            reports.append({'seed': seed, 'events': events, 'summary': groups, 'rows': rows})
            print(f'seed {seed} checked', flush=True)
    if frozen_digest(organism, original) != before or any(sha(p) != d for p, d in protected.items()):
        raise ValueError('Protected source changed')
    report = {'status': 'adaptive_new_states_development_only', 'weight_sha256': WEIGHT_SHA,
              'individual_sha256': INDIVIDUAL_SHA, 'gate_sha256': protected[GATE],
              'selector_sha256': protected[SELECTOR], 'trajectories': reports,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Two fresh states share original model parameters and address bases, not new individuals.',
                              'All 64 addresses were supervised in prior selector training.',
                              'Questions were used in previous development; no final independent test.',
                              'Each trajectory uses 14 trusted direct observations, 4 known and 4 unknown queried keys.',
                              'Negative misroutes are recorded without generating or executing requests.',
                              'Known replies are greedy with closed thinking and at most 8 new tokens.',
                              'Threshold and scale were previously selected on development data.',
                              'No ordinary chat preservation, open entities, history, or crash recovery evaluated.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps([{'seed': r['seed'], 'summary': r['summary']} for r in reports], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
