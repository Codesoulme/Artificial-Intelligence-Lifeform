"""Test paired memory corrections with unchanged DeepSeek questions and addresses."""
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
from spora.finite_question_guard import valid_finite_question_form
from spora.growing import load, query, save
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest
from spora.open_base_memory import memory_hooks

OUTPUT = ROOT / 'runs/deepseek-memory-correction-causal-20261110'
PAIRS = ((6, 3, 11), (22, 13, 4), (35, 7, 2), (51, 9, 15))
INTERFERENCE = ((1, 5), (2, 12), (3, 1), (4, 6))


def observe(model, state, events):
    actions = []
    for key, value in events:
        state, decision = observe_sourced(model, state, key, value, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('Trusted observation was not written')
        actions.append({'key': key, 'value': value, 'action': decision['action']})
    return state, actions


def native_values(model, state):
    keys = torch.tensor([row[0] for row in PAIRS], dtype=torch.long)
    zeros = torch.zeros_like(keys)
    with torch.no_grad():
        return query(model, state, keys, zeros, zeros).argmax(-1).tolist()


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
    if any(not NEW_CASES[i][0] or str(key) not in NEW_CASES[i][2] for i, (key, _, _) in enumerate(PAIRS)):
        raise ValueError('Question/key mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    original_digest = frozen_digest(organism, original)
    before_state, initial_actions = observe(organism, fresh_controlled_state(organism),
                                            [(key, old) for key, old, _ in PAIRS] + list(INTERFERENCE))
    after_state, correction_actions = observe(organism, copy.deepcopy(before_state),
                                               [(key, new) for key, _, new in PAIRS])
    expected_before = [old for _, old, _ in PAIRS]
    expected_after = [new for _, _, new in PAIRS]
    if native_values(organism, before_state) != expected_before or native_values(organism, after_state) != expected_after:
        raise ValueError('Native current memory does not match controlled observations')
    before_digest = frozen_digest(organism, before_state)
    after_digest = frozen_digest(organism, after_state)
    OUTPUT.mkdir(parents=False, exist_ok=False)
    path_before, path_after = OUTPUT / 'before.pt', OUTPUT / 'after.pt'
    save(path_before, organism, before_state, {'stage': 'before_corrections'})
    save(path_after, organism, after_state, {'stage': 'after_corrections'})
    model_before, reload_before, _ = load(path_before, 'cpu')
    model_after, reload_after, _ = load(path_after, 'cpu')
    if (frozen_digest(model_before, reload_before) != before_digest
            or frozen_digest(model_after, reload_after) != after_digest
            or native_values(model_before, reload_before) != expected_before
            or native_values(model_after, reload_after) != expected_after):
        raise ValueError('Reloaded state mismatch')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    selector = PooledAddress(base.config.hidden_size, organism.config.memory_width).to('cuda').eval()
    selector.load_state_dict(selector_data['selector'], strict=True)
    gate = IntentGate(base.config.hidden_size).to('cuda').eval()
    gate.load_state_dict(gate_data['state'], strict=True)
    adapter = ScaledReadout(base.config.hidden_size, organism.config.memory_width, 0.5).to('cuda').eval()
    adapter.load_state_dict(selector_data['readout'], strict=True)
    bases = [F.normalize(site.current_basis.to('cuda').float(), dim=-1) for site in organism.memories]
    questions = [NEW_CASES[i][2] for i in range(len(PAIRS))]
    features, masks = extract(base, tokenizer, questions, batch_size=1)
    hidden, hidden_masks = extract(base, tokenizer, [hide_numbers(q) for q in questions], batch_size=1)
    rows = []
    with torch.inference_mode():
        for index, (key, old, new) in enumerate(PAIRS):
            q = questions[index]
            score = float(gate(hidden[index:index+1].to('cuda'), hidden_masks[index:index+1].to('cuda')).sigmoid()[0])
            candidate = selector(features[index:index+1].to('cuda'), masks[index:index+1].to('cuda'))
            predicted = [int((candidate[i] @ bases[i].T).argmax(-1)) for i in range(2)]
            route = initial_route(valid_finite_question_form(q), score, predicted)
            row = {'question': q, 'key': key, 'predicted': predicted, 'gate_score': score, 'route': route,
                   'expected_before': old, 'expected_after': new, 'answers': {}}
            if route == 'inspect':
                address = [basis[predicted[i]:predicted[i]+1] for i, basis in enumerate(bases)]
                inputs = closed_think_inputs(tokenizer, q, 'cuda')
                for phase, current, expected in (('before', before_state, old), ('after', after_state, new),
                                                  ('reload_before', reload_before, old), ('reload_after', reload_after, new)):
                    with memory_hooks([base.model.layers[5], base.model.layers[17]], adapter, current,
                                      current_only=True, addresses=address):
                        generated = base.generate(**inputs, max_new_tokens=8, do_sample=False,
                                                  pad_token_id=tokenizer.eos_token_id)
                    tokens = generated[0, inputs['input_ids'].shape[1]:].tolist()
                    text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    row['answers'][phase] = {'text': text, 'tokens': tokens,
                                             'correct': predicted == [key, key] and numeric_correct(text, expected),
                                             'ended_with_eos': tokenizer.eos_token_id in tokens}
            rows.append(row)
            print(f'checked {index + 1}/{len(PAIRS)}', flush=True)
    if (frozen_digest(organism, original) != original_digest or frozen_digest(organism, before_state) != before_digest
            or frozen_digest(organism, after_state) != after_digest or any(sha(p) != d for p, d in protected.items())):
        raise ValueError('Protected input changed')
    summary = {'pairs': len(rows), 'address_and_gate_correct': sum(r['route'] == 'inspect' and r['predicted'] == [r['key']]*2 for r in rows),
               'both_answers_correct': sum(r['answers'].get('before', {}).get('correct', False)
                                           and r['answers'].get('after', {}).get('correct', False) for r in rows),
               'reload_identical': sum(bool(r['answers']) and all(r['answers'][a]['tokens'] == r['answers'][b]['tokens']
                                     for a, b in (('before', 'reload_before'), ('after', 'reload_after'))) for r in rows)}
    report = {'status': 'paired_correction_development_only', 'weight_sha256': WEIGHT_SHA,
              'individual_sha256': INDIVIDUAL_SHA, 'selector_sha256': protected[SELECTOR], 'gate_sha256': protected[GATE],
              'before_state_sha256': sha(path_before), 'after_state_sha256': sha(path_after),
              'initial_actions': initial_actions, 'correction_actions': correction_actions,
              'summary': summary, 'rows': rows, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Four preselected finite numeric keys and values; all values and addresses were seen in prior training.',
                              'Question wording, scale, gate and selector were used in prior development.',
                              'Fresh states share one model and basis; correction updates all four queried keys together.',
                              'Known state is established by controlled observations; unknown/status routing not tested.',
                              'Read-only same-process reload is not a crash-recovery or cross-process test.',
                              'Closed-thinking greedy generation has an eight-token limit.',
                              'Original model and individual remain unchanged.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    run()
