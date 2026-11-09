"""Test per-key correction and cross-key interference in read-only DeepSeek output."""
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
from scripts.deepseek_memory_correction_causal import OUTPUT as SOURCE, PAIRS, native_values, observe
from scripts.deepseek_memory_intent import IntentGate
from scripts.deepseek_pooled_address import PooledAddress, extract
from scripts.deepseek_read_strength_probe import ScaledReadout
from scripts.deepseek_route_calibration import NEW_CASES
from scripts.deepseek_semantic_gate import hide_numbers
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.open_base_trajectory_comparison import numeric_correct
from spora.finite_question_guard import valid_finite_question_form
from spora.growing import load
from spora.language_training import frozen_digest
from spora.open_base_memory import memory_hooks

OUTPUT = ROOT / 'runs/deepseek-single-correction-20261110.json'


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_report, source_state = SOURCE / 'result.json', SOURCE / 'before.pt'
    protected = {p: sha(p) for p in (MODEL / 'model.safetensors', INDIVIDUAL, SELECTOR, GATE,
                                       source_report, source_state)}
    if protected[MODEL / 'model.safetensors'] != WEIGHT_SHA or protected[INDIVIDUAL] != INDIVIDUAL_SHA:
        raise ValueError('Protected model mismatch')
    with source_report.open(encoding='utf-8') as stream:
        prior = json.load(stream)
    if (prior['before_state_sha256'] != protected[source_state] or prior['selector_sha256'] != protected[SELECTOR]
            or prior['gate_sha256'] != protected[GATE]):
        raise ValueError('Prior experiment provenance mismatch')
    selector_data = torch.load(SELECTOR, map_location='cpu', weights_only=True)
    gate_data = torch.load(GATE, map_location='cpu', weights_only=True)
    if (selector_data.get('format') != 'deepseek-spora-question-pooled-finite-address-v1'
            or gate_data.get('format') != 'deepseek-spora-number-invariant-intent-v1'
            or any(d.get('weight_sha256') != WEIGHT_SHA or d.get('individual_sha256') != INDIVIDUAL_SHA
                   or d.get('layers') != [5, 17] for d in (selector_data, gate_data))):
        raise ValueError('Component provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    original_digest = frozen_digest(organism, original)
    source_model, baseline, _ = load(source_state, 'cpu')
    baseline_digest = frozen_digest(source_model, baseline)
    if native_values(source_model, baseline) != [old for _, old, _ in PAIRS]:
        raise ValueError('Baseline native values mismatch')
    branches = [('baseline', baseline, [old for _, old, _ in PAIRS], None)]
    for index, (key, _, new) in enumerate(PAIRS):
        state, actions = observe(source_model, copy.deepcopy(baseline), [(key, new)])
        expected = [new if i == index else old for i, (_, old, _) in enumerate(PAIRS)]
        if native_values(source_model, state) != expected:
            raise ValueError('Native value changed unexpectedly')
        branches.append((f'correct_{key}', state, expected, actions))
    state_digests = [frozen_digest(source_model, branch[1]) for branch in branches]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    selector = PooledAddress(base.config.hidden_size, source_model.config.memory_width).to('cuda').eval()
    selector.load_state_dict(selector_data['selector'], strict=True)
    gate = IntentGate(base.config.hidden_size).to('cuda').eval()
    gate.load_state_dict(gate_data['state'], strict=True)
    adapter = ScaledReadout(base.config.hidden_size, source_model.config.memory_width, 0.5).to('cuda').eval()
    adapter.load_state_dict(selector_data['readout'], strict=True)
    bases = [F.normalize(site.current_basis.to('cuda').float(), dim=-1) for site in source_model.memories]
    questions = [NEW_CASES[i][2] for i in range(len(PAIRS))]
    features, masks = extract(base, tokenizer, questions, batch_size=1)
    hidden, hidden_masks = extract(base, tokenizer, [hide_numbers(q) for q in questions], batch_size=1)
    decisions = []
    with torch.inference_mode():
        for index, (key, _, _) in enumerate(PAIRS):
            score = float(gate(hidden[index:index+1].to('cuda'), hidden_masks[index:index+1].to('cuda')).sigmoid()[0])
            addresses = selector(features[index:index+1].to('cuda'), masks[index:index+1].to('cuda'))
            predicted = [int((addresses[i] @ bases[i].T).argmax(-1)) for i in range(2)]
            route = initial_route(valid_finite_question_form(questions[index]), score, predicted)
            decisions.append({'score': score, 'predicted': predicted, 'route': route})
    results = []
    with torch.inference_mode():
        for branch_name, state, expected, actions in branches:
            rows = []
            for index, (key, _, _) in enumerate(PAIRS):
                decision = decisions[index]
                row = {'key': key, 'question': questions[index], 'expected': expected[index], **decision}
                if decision['route'] == 'inspect':
                    address = [basis[decision['predicted'][i]:decision['predicted'][i]+1] for i, basis in enumerate(bases)]
                    inputs = closed_think_inputs(tokenizer, questions[index], 'cuda')
                    with memory_hooks([base.model.layers[5], base.model.layers[17]], adapter, state,
                                      current_only=True, addresses=address):
                        generated = base.generate(**inputs, max_new_tokens=8, do_sample=False,
                                                  pad_token_id=tokenizer.eos_token_id)
                    tokens = generated[0, inputs['input_ids'].shape[1]:].tolist()
                    text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    row.update(text=text, tokens=tokens, ended_with_eos=tokenizer.eos_token_id in tokens,
                               correct=decision['predicted'] == [key, key] and numeric_correct(text, expected[index]))
                else:
                    row.update(text=None, tokens=None, ended_with_eos=False, correct=False)
                rows.append(row)
            results.append({'branch': branch_name, 'actions': actions, 'expected': expected, 'rows': rows})
            print(f'{branch_name} checked', flush=True)
    if (frozen_digest(organism, original) != original_digest or frozen_digest(source_model, baseline) != baseline_digest
            or any(frozen_digest(source_model, branch[1]) != d for branch, d in zip(branches, state_digests))
            or any(sha(p) != d for p, d in protected.items())):
        raise ValueError('Protected input changed')
    baseline_rows = results[0]['rows']
    comparisons = []
    for index, result in enumerate(results[1:]):
        changed = [r['key'] for i, r in enumerate(result['rows']) if i != index and r['tokens'] != baseline_rows[i]['tokens']]
        regressions = [r['key'] for i, r in enumerate(result['rows']) if i != index and baseline_rows[i]['correct'] and not r['correct']]
        comparisons.append({'corrected_key': PAIRS[index][0], 'target_correct': result['rows'][index]['correct'],
                            'other_outputs_changed': changed, 'other_correctness_regressions': regressions})
    report = {'status': 'single_correction_development_only', 'weight_sha256': WEIGHT_SHA,
              'individual_sha256': INDIVIDUAL_SHA, 'source_sha256': protected[source_state],
              'selector_sha256': protected[SELECTOR], 'gate_sha256': protected[GATE],
              'baseline_correct': sum(r['correct'] for r in baseline_rows), 'comparisons': comparisons, 'results': results,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Four one-event branches share a single baseline state and model parameters.',
                              'All values, keys, questions and strength were used in prior development.',
                              'Native read was checked for all four keys after each correction.',
                              'No unknown status, open entities, full dialogue, long-term interference or crash recovery.',
                              'Closed thinking and greedy generation limited to eight tokens.',
                              'Experimental state branches are in memory and never replace source checkpoints.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'baseline_correct': report['baseline_correct'], 'comparisons': comparisons}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
