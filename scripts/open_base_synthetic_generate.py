"""Free-generation check for the bounded synthetic memory adapter."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.open_base_baseline import MODEL, INDIVIDUAL, INDIVIDUAL_SHA, WEIGHT_SHA, REVISION, sha
from scripts.open_base_synthetic_alignment import HELD_PROMPTS, make_state
from spora.growing import load
from spora.language_training import frozen_digest
from spora.open_base_memory import OpenBaseMemory, memory_hooks

ADAPTER = ROOT/'runs/open-base-synthetic-alignment-400-20261109/adapter.pt'
ADAPTER_SHA = '5dc23b761209464c588caa1f5c497c6d164220a442162a4c47af22eb9f59a014'
OUTPUT = ROOT/'runs/open-base-synthetic-free-generation-20261109.json'


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if (sha(MODEL/'model.safetensors') != WEIGHT_SHA or sha(INDIVIDUAL) != INDIVIDUAL_SHA
            or sha(ADAPTER) != ADAPTER_SHA):
        raise ValueError('Input digest mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='eager').to('cuda').eval()
    organism, original, _ = load(INDIVIDUAL, 'cuda')
    before = frozen_digest(organism, original)
    payload = torch.load(ADAPTER, map_location='cpu', weights_only=True)
    if (payload.get('format') != 'spora-open-base-synthetic-v1' or payload['weight_sha256'] != WEIGHT_SHA
            or payload['individual_sha256'] != INDIVIDUAL_SHA or payload['layers'] != [5, 17]):
        raise ValueError('Adapter provenance mismatch')
    adapter = OpenBaseMemory(model.config.hidden_size, organism.config.memory_width).to('cuda').eval()
    adapter.load_state_dict(payload['state'], strict=True)
    layers = [model.model.layers[5], model.model.layers[17]]
    rows = []
    with torch.inference_mode():
        for seed in (700, 701):
            for prompt in HELD_PROMPTS:
                formatted = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                                           add_generation_prompt=True)
                inputs = tokenizer(formatted, return_tensors='pt').to('cuda')
                inputs = {key: value.expand(2, -1) for key, value in inputs.items()}
                for mode in ('normal', 'swapped', 'empty'):
                    state = make_state(organism.config.memory_width, seed, 'cuda', swap=mode == 'swapped', empty=mode == 'empty')
                    with memory_hooks(layers, adapter, state):
                        generated = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                                   pad_token_id=tokenizer.eos_token_id)
                    outputs = [tokenizer.decode(x[inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                               for x in generated]
                    expected = ['2', '1'] if mode == 'swapped' else ['1', '2']
                    rows.append({'seed': seed, 'prompt': prompt, 'mode': mode, 'outputs': outputs,
                                 'exact_digits': [a == b for a, b in zip(outputs, expected)]})
    if (sha(MODEL/'model.safetensors') != WEIGHT_SHA or sha(INDIVIDUAL) != INDIVIDUAL_SHA
            or sha(ADAPTER) != ADAPTER_SHA or frozen_digest(organism, original) != before):
        raise ValueError('Protected input changed')
    summary = {}
    for mode in ('normal', 'swapped', 'empty'):
        selected = [row for row in rows if row['mode'] == mode]
        summary[mode] = {'pairs': len(selected), 'both_exact': sum(all(row['exact_digits']) for row in selected),
                         'individual_exact': sum(sum(row['exact_digits']) for row in selected)}
    report = {'status': 'synthetic_free_generation_not_real_memory_alignment', 'revision': REVISION,
              'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA, 'adapter_sha256': ADAPTER_SHA,
              'summary': summary, 'rows': rows, 'seconds': time.perf_counter()-start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Only four pairs, artificial matrices and held prompts from the same fixed synthetic task.',
                              'Empty-state labels are arbitrary, so the empty condition only diagnoses base preference.',
                              'No original SPORA memory semantics are learned or evaluated.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
