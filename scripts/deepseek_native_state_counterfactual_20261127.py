"""Hold question/address fixed while changing isolated native memory state."""
import json
import sys
import time
import traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_memory_answer_distill import LAYERS, _load_candidate
from scripts.deepseek_spora_memory_answer_distill_v4 import VALUE_PAIRS
from scripts.deepseek_spora_online_correction_answer_probe import SOURCE_CHECKPOINT, CANDIDATES, _query
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.integrated_checkpoint import IntegratedSPORARuntime, load_integrated

OUT = ROOT / 'runs/research/native-state-counterfactual-20261127-v1'
INDICES = (0, 1, 2, 3)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        paths = {'base': MODEL / 'model.safetensors', 'checkpoint': SOURCE_CHECKPOINT,
                 **{f'layer_{layer}': CANDIDATES / f'layer-{layer}.pt' for layer in LAYERS}}
        before = {name: sha(path) for name, path in paths.items()}
        if before['base'] != WEIGHT_SHA:
            raise ValueError('Unexpected base')
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
        rows, events = [], []
        for index in INDICES:
            values = VALUE_PAIRS[index]
            if values[0] == values[1]:
                raise ValueError('Pair must use distinct values')
            question = f'{NAMES[index]}现在爱喝什么？'
            for state_number, value in enumerate(values):
                runtime.state = fresh_controlled_state(organism)
                decision = runtime.observe_sourced(KEYS[index], value, 'direct', 1.0)
                if not decision['current_write'] or runtime.state.cursor != 1:
                    raise ValueError('Write rejected')
                events.append({'index': index, 'state': state_number, 'action': decision['action']})
                for condition in ('base', 'spora_pair'):
                    result = _query(runtime, tokenizer, KEYS[index], question, 'current', 0, value, condition)
                    answer = result['answer'].strip().rstrip('。.!！ ').strip()
                    rows.append({'index': index, 'state': state_number, 'condition': condition,
                                 'exact': answer == WORDS[value], 'contains': result['correct'],
                                 'answer': result['answer'], 'expected': WORDS[value]})
            print(json.dumps({'completed_entities': index + 1, 'total': len(INDICES)}), flush=True)
        after = {name: sha(path) for name, path in paths.items()}
        result = {'rows': rows, 'events': events, 'input_hashes': before, 'input_hashes_after': after,
                  'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'limitations': ['Known names and finite values; addresses supplied by experiment.',
                                  'Temporary fresh states, no autonomous write, routing, save or reload.',
                                  'Short generated outputs; not independent or general chat evaluation.']}
        (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        if before != after:
            raise ValueError('Protected inputs changed')
        print(json.dumps({'complete': True, 'seconds': result['seconds']}), flush=True)
    except BaseException:
        (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf8')
        raise


if __name__ == '__main__':
    main()
