"""Use the persisted v4 adapter after checkpoint reload on two controlled values."""
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
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import query
from spora.integrated_checkpoint import load_integrated
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest

CHECKPOINT = ROOT/'runs/deepseek-spora-memory-output-integrated-v5-20261111/individual-with-output-adapter.pt'
ADAPTER_REPORT = ROOT/'runs/deepseek-spora-memory-output-adapter-20261111/result.json'
OUTPUT = ROOT/'runs/deepseek-spora-persisted-output-probe-v2-20261111.json'


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    report = json.loads(ADAPTER_REPORT.read_text(encoding='utf-8'))
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA or sha(CHECKPOINT) == '':
        raise ValueError('Protected input mismatch')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    decoder, organism, original_state = load_integrated(CHECKPOINT, base, MODEL, individual_device='cuda')
    if decoder.output_adapter is None or sha(CHECKPOINT) != json.loads(
            (CHECKPOINT.parent/'result.json').read_text(encoding='utf-8'))['checkpoint_sha256']:
        raise ValueError('Persisted adapter checkpoint mismatch')
    before = frozen_digest(organism, original_state)
    key = KEYS[0]
    rows = []
    for value in (1, 6):
        state = fresh_controlled_state(organism)
        state, decision = observe_sourced(organism, state, key, value, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('Controlled write rejected')
        keys = torch.tensor([key], dtype=torch.long, device='cuda')
        zeros = torch.zeros_like(keys)
        native = int(query(organism, state, keys, zeros, zeros).argmax(-1))
        question = f'根据保存的偏好，{NAMES[0]}此刻愿意喝什么？'
        prefix = closed_think_inputs(tokenizer, question, 'cuda')
        addresses = [torch.nn.functional.normalize(site.current_basis[key:key+1].to('cuda').float(), dim=-1)
                     for site in organism.memories]
        ids = decoder.generate(**prefix, memory_state=state, memory_addresses=addresses,
                               memory_slots=torch.tensor([0], device='cuda'), current_only=False,
                               memory_output_logits=query(organism, state, keys, zeros, zeros),
                               max_new_tokens=8, do_sample=False,
                               pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(ids[0, prefix['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        rows.append({'value': value, 'expected': WORDS[value], 'native_value': native,
                     'answer': answer, 'contains_expected': WORDS[value] in answer})
    if frozen_digest(organism, original_state) != before or sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Reloaded source changed')
    output = {'status': 'persisted_output_adapter_runtime_probe', 'checkpoint_sha256': sha(CHECKPOINT),
              'adapter_sha256': report['adapter_sha256'], 'rows': rows,
              'elapsed_seconds': time.perf_counter()-started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Two controlled finite values and one question.',
                              'The memory key is supplied by the caller; routing is not tested here.',
                              'This verifies persistence and invocation, not open-domain dialogue.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'rows': rows, 'elapsed_seconds': output['elapsed_seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
