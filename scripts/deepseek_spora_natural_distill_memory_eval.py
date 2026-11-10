"""Compare pre/post natural-distillation candidates on addressed memories."""
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
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_fresh_entity_graft_eval import FACTS, TEMPLATES
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, SOURCE_CHECKPOINT, _load_candidate, _memory_args
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import grow
from spora.integrated_checkpoint import load_integrated
from spora.internal_memory import observe_sourced

NEW = ROOT/'runs/deepseek-spora-natural-hidden-distill-20261111'
OUTPUT = ROOT/'runs/deepseek-spora-natural-distill-memory-eval-20261111.json'


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base model hash mismatch')
    training = json.loads((NEW/'result.json').read_text(encoding='utf-8'))
    for layer in (6, 12):
        info = training['layers'][str(layer)]
        if sha(NEW/f'layer-{layer}.pt') != info['candidate_sha256'] or sha(PAIR/f'layer-{layer}.pt') != info['start_sha256']:
            raise ValueError('Candidate provenance mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    integrated, organism, _ = load_integrated(SOURCE_CHECKPOINT, base, MODEL, individual_device='cuda')
    state = fresh_controlled_state(organism)
    rows = []
    for index, (name, key, old, new) in enumerate(FACTS):
        for value in (old, new):
            if state.cursor >= len(state.pages)*organism.config.time_count:
                state = grow(organism, state)
            state, decision = observe_sourced(organism, state, key, value, 'direct', 1.0)
            if not decision['current_write']:
                raise ValueError('Memory write rejected')
        for intent, template in TEMPLATES.items():
            question = template.format(name=name)
            rows.append({'name': name, 'key': key, 'intent': intent, 'question': question,
                         'event_time': 2*index + (intent != 'past'),
                         'value': old if intent == 'past' else new,
                         'other_value': new if intent == 'past' else old,
                         'prefix': closed_think_inputs(tokenizer, question, 'cuda')})
    original = {i: base.model.layers[i] for i in (6, 12)}
    old_blocks = {i: _load_candidate(PAIR/f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    new_blocks = {i: _load_candidate(NEW/f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    results = {}
    with torch.inference_mode():
        for mode, blocks in (('integrated_base', original), ('old_pair', old_blocks), ('new_pair', new_blocks)):
            for i in (6, 12):
                base.model.layers[i] = blocks[i]
            answers = []
            for row in rows:
                prefix = row['prefix']
                ids = integrated.generate(**prefix, **_memory_args(organism, state, row),
                                          max_new_tokens=8, do_sample=False,
                                          pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                text = tokenizer.decode(ids[0, prefix['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                expected, conflict = WORDS[row['value']], WORDS[row['other_value']]
                answers.append({'name': row['name'], 'intent': row['intent'], 'question': row['question'],
                                'expected': expected, 'conflict': conflict, 'answer': text,
                                'exact_correct': text in (expected, expected+'.', expected+'。'),
                                'contains_correct': expected in text and conflict not in text})
            results[mode] = {'total': len(answers), 'exact_correct': sum(x['exact_correct'] for x in answers),
                             'contains_correct': sum(x['contains_correct'] for x in answers), 'rows': answers}
            print(json.dumps({'mode': mode, 'exact': results[mode]['exact_correct'],
                              'contains': results[mode]['contains_correct'], 'total': len(answers)}), flush=True)
    report = {'results': results, 'base_sha256': WEIGHT_SHA, 'source_sha256': sha(SOURCE_CHECKPOINT),
              'old_pair_sha256': {str(i): sha(PAIR/f'layer-{i}.pt') for i in (6, 12)},
              'new_pair_sha256': {str(i): sha(NEW/f'layer-{i}.pt') for i in (6, 12)},
              'elapsed_seconds': time.perf_counter()-started,
              'limitations': ['Eight previously inspected questions in a finite beverage domain.',
                              'Addresses and temporal slots supplied programmatically.',
                              'Short closed-reasoning answers; not autonomous memory or open-chat validation.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    run()
