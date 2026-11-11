"""Paired memory-value interventions and unconditioned baseline on fresh prompts."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_memory_answer_distill import SOURCE_CHECKPOINT, _load_candidate, _memory_args
from scripts.deepseek_spora_recalibrate_name_router import SOURCE
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import query
from spora.integrated_checkpoint import load_integrated
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT/'runs/deepseek-spora-counterfactual-memory-audit-v2-20261111.json'
PARTIAL = ROOT/'runs/deepseek-spora-counterfactual-memory-audit-v2-20261111.jsonl'
CANDIDATE_SOURCE = SOURCE
PROMPTS = (
    (0, '请用一句话说明小林当前最喜欢的饮料。'),
    (0, '根据保存的偏好，小林此刻愿意喝什么？'),
    (6, '请用一句话说明小岚当前最喜欢的饮料。'),
    (6, '根据保存的偏好，小岚此刻愿意喝什么？'),
)
VALUES = (1, 6)


def run():
    if OUTPUT.exists() or PARTIAL.exists():
        raise FileExistsError('Preserve existing audit output')
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    integrated, organism, original = load_integrated(SOURCE_CHECKPOINT, base, MODEL, individual_device='cuda')
    digest = frozen_digest(organism, original)
    original_layers = {i: base.model.layers[i] for i in (6, 12)}
    blocks = {i: _load_candidate(CANDIDATE_SOURCE/f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    rows = []
    with torch.inference_mode(), PARTIAL.open('x', encoding='utf-8', buffering=1) as stream:
        for index, question in PROMPTS:
            prefix = closed_think_inputs(tokenizer, question, 'cuda')
            for condition, value in (('original_base_no_memory', None), ('candidate_no_memory', None),
                                     ('original_reader_value_1', 1), ('original_reader_value_6', 6),
                                     ('candidate_value_1', 1), ('candidate_value_6', 6)):
                for layer in (6, 12):
                    base.model.layers[layer] = original_layers[layer] if condition.startswith('original_') else blocks[layer]
                args = {'use_cache': False}
                native_value = None
                if value is not None:
                    state = fresh_controlled_state(organism)
                    state, decision = observe_sourced(organism, state, KEYS[index], value, 'direct', 1.0)
                    if not decision['current_write']:
                        raise ValueError('Controlled memory write rejected')
                    keys = torch.tensor([KEYS[index]], device='cuda')
                    native_value = int(query(organism, state, keys, torch.zeros_like(keys), torch.zeros_like(keys)).argmax(-1))
                    args = _memory_args(organism, state, {'key': KEYS[index], 'intent': 'current'})
                ids = integrated.generate(**prefix, **args, max_new_tokens=32, do_sample=False,
                                          pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                new = ids[0, prefix['input_ids'].shape[1]:]
                answer = tokenizer.decode(new, skip_special_tokens=True).strip()
                row = {'name': NAMES[index], 'key': KEYS[index], 'question': question, 'condition': condition,
                       'expected_word': None if value is None else WORDS[value], 'native_value': native_value,
                       'answer': answer, 'tokens': int(new.numel()),
                       'contains_expected': None if value is None else WORDS[value] in answer,
                       'contains_other': None if value is None else WORDS[VALUES[1] if value == VALUES[0] else VALUES[0]] in answer}
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                print(json.dumps({'case': len(rows), 'condition': condition, 'tokens': row['tokens'],
                                  'expected_in_answer': row['contains_expected'], 'other_in_answer': row['contains_other']}), flush=True)
    if frozen_digest(organism, original) != digest or sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected input changed')
    paired = []
    for n in range(len(PROMPTS)):
        a, b = rows[n*6+4:n*6+6]
        paired.append({'question': a['question'], 'outputs_differ': a['answer'] != b['answer'],
                       'both_contain_correct_only': a['contains_expected'] and not a['contains_other'] and
                                                   b['contains_expected'] and not b['contains_other']})
    report = {'status': 'controlled_counterfactual_development_audit', 'base_sha256': WEIGHT_SHA,
              'source_sha256': sha(SOURCE_CHECKPOINT), 'candidate_source': str(CANDIDATE_SOURCE),
              'candidate_sha256': {str(i): sha(CANDIDATE_SOURCE/f'layer-{i}.pt') for i in (6, 12)},
              'rows': rows, 'pairs': paired, 'elapsed_seconds': time.perf_counter()-started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Addresses supplied by experiment; no natural-language router is tested.',
                              'Two names and two finite values; no unrestricted facts or multi-turn chat.',
                              'String containment is not a semantic judge, especially for negation.',
                              '32-token capped, closed-thinking greedy generation; not unlimited natural conversation.',
                              'Original and candidate layers are compared with and without memory, but checkpoints were trained differently.',
                              'Swapped states use the same initial individual and differ by one controlled direct write.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'pairs': paired, 'elapsed_seconds': report['elapsed_seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
