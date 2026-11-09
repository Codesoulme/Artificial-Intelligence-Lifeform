"""Compare request-conditioned strength against fixed forced memory reads."""
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

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_chat_interference import PROMPTS, SELECTOR
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_memory_intent import IntentGate
from scripts.deepseek_oasst_route_probe import REVIEW, SOURCE, load_questions
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_read_strength_probe import ScaledReadout
from scripts.deepseek_route_calibration import NEW_CASES
from scripts.deepseek_semantic_gate import hide_numbers
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.open_base_trajectory_comparison import numeric_correct
from scripts.spora_unknown_state_probe import CHECKPOINT, OUTPUT as REFERENCE
from spora.growing import load
from spora.language_training import frozen_digest
from spora.open_base_memory import memory_hooks

GATE = ROOT / 'runs/deepseek-route-hardening-20261110/gate.pt'
OUTPUT = ROOT / 'runs/deepseek-adaptive-read-20261110.json'


def strength(mode, score):
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Invalid gate score')
    if mode == 'fixed':
        return 0.5
    if mode == 'soft':
        return 0.5 * score
    if mode == 'hard':
        return 0.5 if score >= 0.5 else 0.0
    raise ValueError('Unknown strength mode')


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    protected = {p: sha(p) for p in (MODEL / 'model.safetensors', INDIVIDUAL, CHECKPOINT, SELECTOR, REFERENCE,
                                       GATE, REVIEW, SOURCE)}
    if protected[MODEL / 'model.safetensors'] != WEIGHT_SHA or protected[INDIVIDUAL] != INDIVIDUAL_SHA:
        raise ValueError('Protected model mismatch')
    with REFERENCE.open(encoding='utf-8') as stream:
        reference = json.load(stream)
    if protected[CHECKPOINT] != reference['checkpoint_sha256']:
        raise ValueError('Memory state mismatch')
    selector = torch.load(SELECTOR, map_location='cpu', weights_only=True)
    gate_data = torch.load(GATE, map_location='cpu', weights_only=True)
    if (selector.get('format') != 'deepseek-spora-question-pooled-finite-address-v1'
            or gate_data.get('format') != 'deepseek-spora-number-invariant-intent-v1'
            or any(d.get('weight_sha256') != WEIGHT_SHA or d.get('individual_sha256') != INDIVIDUAL_SHA
                   or d.get('layers') != [5, 17] for d in (selector, gate_data))
            or gate_data.get('threshold') != 0.5):
        raise ValueError('Component provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, state, _ = load(CHECKPOINT, 'cpu')
    before = frozen_digest(organism, state)
    if before != reference['checkpoint_frozen_digest']:
        raise ValueError('State digest mismatch')
    known = [r for r in reference['phases']['interference']['rows'] if r['known_expected']][:8]
    if len(known) != 8:
        raise ValueError('Eight known addresses required')
    cases = [{'group': 'chat', 'question': q, 'key': known[0]['key'], 'expected': None} for q in PROMPTS]
    cases.extend({'group': 'memory', 'question': f"对象{r['key']}的值是多少？", 'key': r['key'],
                  'expected': r['expected']} for r in known)
    cases.extend({'group': 'negative', 'question': q, 'key': known[0]['key'], 'expected': None}
                 for label, _, q in NEW_CASES if not label)
    cases.extend({'group': 'oasst', 'question': '对象8：' + r['question'], 'key': known[0]['key'], 'expected': None}
                 for r in load_questions())
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    adapter = ScaledReadout(model.config.hidden_size, organism.config.memory_width, 0.5).to('cuda').eval()
    adapter.load_state_dict(selector['readout'], strict=True)
    gate = IntentGate(model.config.hidden_size).to('cuda').eval()
    gate.load_state_dict(gate_data['state'], strict=True)
    features, masks = extract(model, tokenizer, [hide_numbers(r['question']) for r in cases], batch_size=1)
    scores = []
    with torch.inference_mode():
        for offset in range(0, len(cases), 4):
            scores.extend(gate(features[offset:offset + 4].to('cuda'),
                               masks[offset:offset + 4].to('cuda')).sigmoid().cpu().tolist())
    bases = [F.normalize(site.current_basis.to('cuda').float(), dim=-1) for site in organism.memories]
    layers = [model.model.layers[5], model.model.layers[17]]
    rows = []
    with torch.inference_mode():
        for case, score in zip(cases, scores):
            row = dict(case, gate_score=score, strengths={mode: strength(mode, score)
                                                        for mode in ('fixed', 'soft', 'hard')})
            if case['group'] in ('chat', 'memory'):
                inputs = closed_think_inputs(tokenizer, case['question'], 'cuda')
                max_tokens = 64 if case['group'] == 'chat' else 8
                baseline = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                          pad_token_id=tokenizer.eos_token_id)
                baseline_tokens = baseline[0, inputs['input_ids'].shape[1]:].tolist()
                row['baseline'] = tokenizer.decode(baseline_tokens, skip_special_tokens=True).strip()
                row['results'] = {}
                address = [basis[case['key']:case['key'] + 1] for basis in bases]
                for mode, value in row['strengths'].items():
                    adapter.strength = value
                    with memory_hooks(layers, adapter, state, current_only=True, addresses=address):
                        generated = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                                   pad_token_id=tokenizer.eos_token_id)
                    tokens = generated[0, inputs['input_ids'].shape[1]:].tolist()
                    text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    row['results'][mode] = {'text': text, 'same_as_baseline': tokens == baseline_tokens,
                                            'ended_with_eos': tokenizer.eos_token_id in tokens,
                                            'correct': numeric_correct(text, case['expected'])
                                            if case['group'] == 'memory' else None}
            rows.append(row)
            if case['group'] in ('chat', 'memory'):
                print(f"{case['group']} checked", flush=True)
    if frozen_digest(organism, state) != before or any(sha(p) != d for p, d in protected.items()):
        raise ValueError('Protected source changed')
    summary = {}
    for mode in ('fixed', 'soft', 'hard'):
        summary[mode] = {'chat_identical': sum(r['results'][mode]['same_as_baseline'] for r in rows if r['group'] == 'chat'),
                         'memory_correct': sum(r['results'][mode]['correct'] for r in rows if r['group'] == 'memory'),
                         'negative_read_enabled': sum(r['strengths'][mode] > 0 for r in rows if r['group'] == 'negative'),
                         'oasst_read_enabled': sum(r['strengths'][mode] > 0 for r in rows if r['group'] == 'oasst')}
    report = {'status': 'adaptive_strength_development_only', 'weight_sha256': WEIGHT_SHA,
              'gate_sha256': protected[GATE], 'state_sha256': protected[CHECKPOINT],
              'selector_sha256': protected[SELECTOR], 'summary': summary, 'rows': rows,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Research-only request-level gate with fixed 0.5 scale; not integrated into runtime.',
                              'Uses externally supplied correct memory address and one previously tested state.',
                              'All prompts and gates were used in earlier development; not independent final evaluation.',
                              'No symbolic guard or state-aware unknown check, to expose gate behavior.',
                              'Negative requests are checked for enabled strength, not generated or executed.',
                              'Positive recall is only eight finite numeric memories with closed thinking.',
                              'Chat generation is greedy and limited to 64 tokens; text equality is not quality.',
                              'Soft mode always enables reads for positive sigmoid scores.',
                              'Gate scores are not calibrated probabilities.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
