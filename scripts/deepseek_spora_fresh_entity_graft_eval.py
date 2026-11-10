"""Compare fixed grafts on fresh names, fact assignments and question wording."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_memory_answer_distill import SOURCE_CHECKPOINT, _load_candidate, _memory_args
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import grow
from spora.integrated_checkpoint import load_integrated
from spora.internal_memory import observe_sourced
from spora.structure_transfer import SPORALayerBlend

LAYER18 = ROOT / 'runs/deepseek-spora-layer18-progressive-answer-20261111/layer-18.pt'
OUTPUT = ROOT / 'runs/deepseek-spora-fresh-entity-graft-eval-20261111.json'
FACTS = (('小桐', 2, 7, 3), ('阿舟', 3, 4, 0), ('小榆', 4, 2, 5), ('阿璇', 5, 1, 6))
TEMPLATES = {
    'current': '记忆里{name}如今偏好的饮料是哪一种？',
    'past': '更改记录之前，{name}最中意的饮品是哪种？',
}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected model hash mismatch')
    training = json.loads((LAYER18.parent / 'result.json').read_text(encoding='utf-8'))
    if any(sha(PAIR / f'layer-{i}.pt') != training['upstream_pair_sha256'][str(i)]
           for i in (6, 12)) or sha(LAYER18) != training['candidate_sha256']:
        raise ValueError('Layer-18 candidate and paired upstream blocks do not match training')
    torch.manual_seed(99071)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float16,
        attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    integrated, organism, _ = load_integrated(SOURCE_CHECKPOINT, base, MODEL, individual_device='cuda')
    state = fresh_controlled_state(organism)
    rows = []
    for index, (name, key, old, new) in enumerate(FACTS):
        for value in (old, new):
            if state.cursor >= len(state.pages) * organism.config.time_count:
                state = grow(organism, state)
            state, decision = observe_sourced(organism, state, key, value, 'direct', 1.0)
            if not decision['current_write']:
                raise ValueError(f'Write rejected: {name} {decision}')
        for intent, template in TEMPLATES.items():
            question = template.format(name=name)
            rows.append({'name': name, 'key': key, 'intent': intent, 'question': question,
                         'event_time': 2*index + (intent != 'past'),
                         'value': old if intent == 'past' else new,
                         'other_value': new if intent == 'past' else old,
                         'prefix': closed_think_inputs(tokenizer, question, 'cuda')})
    original = {i: base.model.layers[i] for i in (6, 12, 18)}
    pair = {i: _load_candidate(PAIR / f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    layer18 = _load_candidate(LAYER18, 18, base).eval()
    modes = ('integrated_base', 'pair', 'pair_plus_0.1', 'pair_plus_1.0')
    results = {}
    with torch.inference_mode():
        for mode in modes:
            for i in (6, 12):
                base.model.layers[i] = original[i] if mode == 'integrated_base' else pair[i]
            base.model.layers[18] = (SPORALayerBlend(original[18], layer18,
                                      0.1 if mode == 'pair_plus_0.1' else 1.0).to('cuda').eval()
                                     if mode.startswith('pair_plus_') else original[18])
            out = []
            for row in rows:
                prompt = row['prefix']
                generated = integrated.generate(**prompt, **_memory_args(organism, state, row),
                                                max_new_tokens=8, do_sample=False,
                                                pad_token_id=tokenizer.pad_token_id,
                                                eos_token_id=tokenizer.eos_token_id)
                answer = tokenizer.decode(generated[0, prompt['input_ids'].shape[1]:],
                                          skip_special_tokens=True).strip()
                expected, other = WORDS[row['value']], WORDS[row['other_value']]
                out.append({'name': row['name'], 'intent': row['intent'], 'question': row['question'],
                            'expected': expected, 'other': other, 'answer': answer,
                            'contains_correct': expected in answer and other not in answer,
                            'exact_correct': answer in (expected, expected + '。', expected + '.')})
            results[mode] = {'contains_correct': sum(r['contains_correct'] for r in out),
                             'exact_correct': sum(r['exact_correct'] for r in out),
                             'total': len(out), 'rows': out}
            print(json.dumps({mode: {k: v for k, v in results[mode].items() if k != 'rows'}},
                             ensure_ascii=False), flush=True)
    report = {'facts': FACTS, 'templates': TEMPLATES, 'results': results,
              'base_sha256': WEIGHT_SHA, 'source_sha256': sha(SOURCE_CHECKPOINT),
              'pair_sha256': {str(i): sha(PAIR / f'layer-{i}.pt') for i in (6, 12)},
              'layer18_sha256': sha(LAYER18), 'elapsed_seconds': time.perf_counter()-started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Four entities and eight short questions in one beverage domain.',
                              'Addresses and temporal slots supplied by the evaluator.',
                              'No independent general-language or free-chat evaluation.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    run()
