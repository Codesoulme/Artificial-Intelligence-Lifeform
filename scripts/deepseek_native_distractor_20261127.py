"""Native memory generation with paired clean and distractor-filled states."""
import json
import sys
import time
import traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_memory_answer_distill import LAYERS, _load_candidate
from scripts.deepseek_spora_memory_answer_distill_v4 import VALUE_PAIRS
from scripts.deepseek_spora_online_correction_answer_probe import SOURCE_CHECKPOINT, CANDIDATES, _query
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.integrated_checkpoint import IntegratedSPORARuntime, load_integrated

OUT = ROOT / 'runs/research/native-distractor-20261127-v1'
PARENT = ROOT / 'runs/research/native-state-counterfactual-20261127-v1/result.json'
TARGETS, DISTRACTORS = (0, 1, 2, 3), (4, 5, 6, 7)
ORDINARY = ('二加三等于几？', '请说一句鼓励朋友的话。')


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        paths = {'base': MODEL / 'model.safetensors', 'checkpoint': SOURCE_CHECKPOINT, 'previous': PARENT,
                 **{f'layer_{layer}': CANDIDATES / f'layer-{layer}.pt' for layer in LAYERS}}
        before = {k: sha(v) for k, v in paths.items()}
        if before['base'] != WEIGHT_SHA:
            raise ValueError('Unexpected base')
        old = json.loads(PARENT.read_text(encoding='utf8'))
        for k in ('base', 'checkpoint', *(f'layer_{layer}' for layer in LAYERS)):
            if before[k] != old['input_hashes'][k]:
                raise ValueError('Prior experiment input differs: ' + k)
        torch.manual_seed(112746)
        torch.set_num_threads(2)
        torch.cuda.reset_peak_memory_stats()
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        decoder, organism, _ = load_integrated(SOURCE_CHECKPOINT, model, MODEL, individual_device='cuda')
        runtime = IntegratedSPORARuntime(decoder, organism, fresh_controlled_state(organism), MODEL)
        runtime._probe_blocks = {layer: _load_candidate(CANDIDATES / f'layer-{layer}.pt', layer, model)
                                 for layer in LAYERS}
        rows, events, ordinary = [], [], []
        def observe(index, value, target, variant, condition):
            result = runtime.observe_sourced(KEYS[index], value, 'direct', 1.0)
            events.append({'target': target, 'variant': variant, 'condition': condition,
                           'index': index, 'action': result['action'], 'current_write': result['current_write']})
            if not result['current_write']:
                raise ValueError('Write rejected')
        for index in TARGETS:
            question = f'{NAMES[index]}现在爱喝什么？'
            for variant, value in enumerate(VALUE_PAIRS[index]):
                for condition in ('clean', 'distractors'):
                    runtime.state = fresh_controlled_state(organism)
                    observe(index, value, index, variant, condition)
                    if condition == 'distractors':
                        for other in DISTRACTORS:
                            other_value = next(v for v in VALUE_PAIRS[other] if v != value)
                            observe(other, other_value, index, variant, condition)
                    for path in ('base', 'spora_pair'):
                        result = _query(runtime, tokenizer, KEYS[index], question, 'current', 0, value, path)
                        answer = result['answer'].strip().rstrip('。.!！ ').strip()
                        rows.append({'target': index, 'variant': variant, 'condition': condition, 'path': path,
                                     'answer': result['answer'], 'expected': WORDS[value], 'exact': answer == WORDS[value]})
                    if index == 0 and condition == 'distractors':
                        for question_id, prompt in enumerate(ORDINARY):
                            inputs = closed_think_inputs(tokenizer, prompt, 'cuda')
                            with torch.inference_mode():
                                output = runtime.decoder.generate(**inputs, max_new_tokens=8, do_sample=False,
                                         pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                            unconditioned = tokenizer.decode(output[0, inputs['input_ids'].shape[1]:],
                                                             skip_special_tokens=True).strip()
                            forced = _query(runtime, tokenizer, KEYS[index], prompt, 'current', 0, value, 'base')
                            ordinary.append({'variant': variant, 'question_id': question_id,
                                             'without_memory': unconditioned, 'forced_memory': forced['answer']})
            print(json.dumps({'completed_target': index + 1, 'total': len(TARGETS)}), flush=True)
        after = {k: sha(v) for k, v in paths.items()}
        result = {'rows': rows, 'events': events, 'ordinary': ordinary, 'input_hashes': before,
                  'input_hashes_after': after, 'seconds': time.perf_counter() - started,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'limitations': ['Previously used persons and finite values, addresses supplied by experiment.',
                                  'Forced routing does not estimate actual unwanted activation.',
                                  'No persistence, automatic routing, independent names or natural dialogue.',
                                  'Ordinary-question output is limited to 8 tokens.']}
        (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        if before != after:
            raise ValueError('Protected input changed')
        print(json.dumps({'complete': True, 'seconds': result['seconds']}), flush=True)
    except BaseException:
        (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf8')
        raise


if __name__ == '__main__':
    main()
