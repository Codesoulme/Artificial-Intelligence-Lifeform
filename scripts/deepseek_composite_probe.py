"""Read-only end-to-end finite-memory routing experiment; not a user-facing runtime."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_full_answer import thinking_inputs
from scripts.deepseek_guard_adversarial import SECOND_CASES
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_memory_intent import IntentGate
from scripts.deepseek_pooled_address import PooledAddress, extract
from scripts.deepseek_semantic_gate import hide_numbers
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.open_base_trajectory_comparison import numeric_correct
from scripts.spora_unknown_state_probe import CHECKPOINT, OUTPUT as REFERENCE
from spora.current_memory_status import current_memory_status
from spora.finite_question_guard import valid_finite_question_form
from spora.growing import load
from spora.language_training import frozen_digest
from spora.open_base_memory import OpenBaseMemory, memory_hooks


def initial_route(form_ok, score, predicted):
    if not form_ok or not math.isfinite(score) or not 0 <= score <= 1 or score < 0.5:
        return 'defer'
    if (len(predicted) != 2 or any(type(key) is not int or not 0 <= key < 64 for key in predicted)
            or predicted[0] != predicted[1]):
        return 'defer'
    return 'inspect'


def run(selector_path, gate_path, output):
    selector_path, gate_path, output = map(lambda x: Path(x).resolve(), (selector_path, gate_path, output))
    runs = (ROOT / 'runs').resolve()
    if any(not p.is_relative_to(runs) for p in (selector_path, gate_path, output)) or output.exists():
        raise ValueError('Expected sources and new output under runs')
    selector_sha, gate_sha = sha(selector_path), sha(gate_path)
    with REFERENCE.open(encoding='utf-8') as stream:
        reference = json.load(stream)
    state_sha = reference['checkpoint_sha256']
    selector_data = torch.load(selector_path, map_location='cpu', weights_only=True)
    gate_data = torch.load(gate_path, map_location='cpu', weights_only=True)
    if (selector_data.get('format') != 'deepseek-spora-question-pooled-finite-address-v1'
            or gate_data.get('format') != 'deepseek-spora-number-invariant-intent-v1'
            or any(data.get('weight_sha256') != WEIGHT_SHA or data.get('individual_sha256') != INDIVIDUAL_SHA
                   or data.get('layers') != [5, 17] for data in (selector_data, gate_data))
            or gate_data.get('threshold') != 0.5):
        raise ValueError('Component provenance mismatch')
    def verify():
        return (sha(MODEL / 'model.safetensors') == WEIGHT_SHA and sha(INDIVIDUAL) == INDIVIDUAL_SHA
                and sha(CHECKPOINT) == state_sha and sha(selector_path) == selector_sha and sha(gate_path) == gate_sha)
    if not verify():
        raise ValueError('Protected source mismatch')
    torch.set_num_threads(2)
    torch.manual_seed(4613)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, state, _ = load(CHECKPOINT, 'cpu')
    before = frozen_digest(organism, state)
    if before != reference['checkpoint_frozen_digest']:
        raise ValueError('Fresh state digest mismatch')
    reference_rows = reference['phases']['interference']['rows']
    known = [row for row in reference_rows if row['known_expected']][:8]
    unknown = [row for row in reference_rows if not row['known_expected']][:8]
    cases = [{'category': 'known' if row['known_expected'] else 'unknown',
              'question': f"对象{row['key']}的值是多少？", 'label': 1,
              'expected_route': 'answer' if row['known_expected'] else 'unknown',
              'expected_key': row['key'], 'expected_value': row['expected'] if row['known_expected'] else None}
             for row in known + unknown]
    truth = {row['key']: row for row in reference_rows}
    stress_keys = (4, 37, 60, 13, 1, 48)
    if (len(SECOND_CASES) < len(stress_keys)
            or any(SECOND_CASES[i][0] != 1 or str(key) not in SECOND_CASES[i][2] for i, key in enumerate(stress_keys))
            or any(case[0] != 0 for case in SECOND_CASES[len(stress_keys):])):
        raise ValueError('Stress scoring metadata does not match cases')
    for index, (label, category, question) in enumerate(SECOND_CASES):
        # Development labels determine scoring only, never routing or generation.
        reference_row = truth[stress_keys[index]] if label else None
        cases.append({'category': 'stress_' + category, 'question': question, 'label': label,
                      'expected_route': ('answer' if reference_row['known_expected'] else 'unknown') if label else 'defer',
                      'expected_key': stress_keys[index] if label else None,
                      'expected_value': reference_row['expected'] if label and reference_row['known_expected'] else None})
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    selector = PooledAddress(model.config.hidden_size, organism.config.memory_width).to('cuda').eval()
    selector.load_state_dict(selector_data['selector'], strict=True)
    adapter = OpenBaseMemory(model.config.hidden_size, organism.config.memory_width).to('cuda').eval()
    adapter.load_state_dict(selector_data['readout'], strict=True)
    gate = IntentGate(model.config.hidden_size).to('cuda').eval()
    gate.load_state_dict(gate_data['state'], strict=True)
    bases = [F.normalize(site.current_basis.to('cuda').float(), dim=-1) for site in organism.memories]
    questions = [case['question'] for case in cases]
    raw_features, raw_masks = extract(model, tokenizer, questions)
    gate_features, gate_masks = extract(model, tokenizer, [hide_numbers(q) for q in questions])
    rows = []
    with torch.inference_mode():
        for i, case in enumerate(cases):
            question = case['question']
            form_ok = valid_finite_question_form(question)
            score = float(gate(gate_features[i:i + 1].to('cuda'), gate_masks[i:i + 1].to('cuda')).sigmoid()[0])
            addresses = selector(raw_features[i:i + 1].to('cuda'), raw_masks[i:i + 1].to('cuda'))
            predicted = [int((addresses[site] @ bases[site].T).argmax(-1)) for site in range(2)]
            route = initial_route(form_ok, score, predicted)
            row = dict(case, form_ok=form_ok, intent_probability=score, predicted_addresses=predicted, route=route)
            if route == 'inspect':
                key = predicted[0]
                presence = current_memory_status(organism, state, key)
                row['presence'] = presence
                route = 'unknown' if presence['status'] == 'unknown' else 'answer'
                row['route'] = route
                if route == 'answer':
                    inputs = thinking_inputs(tokenizer, question, 'cuda')
                    snapped = [bases[site][key:key + 1] for site in range(2)]
                    with memory_hooks([model.model.layers[5], model.model.layers[17]], adapter, state,
                                      current_only=True, addresses=snapped):
                        generated = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                                   pad_token_id=tokenizer.eos_token_id)
                    raw = tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                    row['generated_text'] = raw
                    row['final_answer'] = raw.split('</think>', 1)[1].strip() if '</think>' in raw else None
            row['route_correct'] = (route == case['expected_route'] and
                                    (not case['label'] or predicted == [case['expected_key']] * 2))
            row['answer_correct'] = (route == 'answer' and row.get('final_answer') is not None and
                                     case['expected_value'] is not None and
                                     numeric_correct(row['final_answer'], case['expected_value'])) if case['label'] else None
            rows.append(row)
            if (i + 1) % 10 == 0:
                print(f'checked {i + 1}/{len(cases)}', flush=True)
    if frozen_digest(organism, state) != before or not verify():
        raise ValueError('Protected input changed')
    summary = {}
    for group in ('known', 'unknown', 'stress_current', 'stress_other'):
        selected = [r for r in rows if (r['category'] == group if group != 'stress_other' else
                                         r['category'].startswith('stress_') and r['category'] != 'stress_current')]
        summary[group] = {'examples': len(selected), 'route_correct': sum(r['route_correct'] for r in selected),
                          'generated': sum(r['route'] == 'answer' for r in selected),
                          'answer_correct': sum(r.get('answer_correct') is True for r in selected)}
    report = {'status': 'read_only_composite_development_only', 'selector_sha256': selector_sha,
              'gate_sha256': gate_sha, 'state_sha256': state_sha, 'summary': summary, 'rows': rows,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Development-only finite numbered-memory route; not safe for open use.',
                              'A symbolic format guard and a learned intent gate run before learned address prediction.',
                              'Native SPORA status is queried by predicted key; status can be wrong and is uncalibrated.',
                              'The original individual weights and address basis are shared with prior training.',
                              'Stress questions were used in previous development assessments.',
                              'Only one saved fresh state, 16 chosen known/unknown keys, and limited adversarial prompts.',
                              'No live writes, history, multi-fact reasoning, independent individuals, or crash recovery.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selector', type=Path, required=True)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.selector, args.gate, args.output)
