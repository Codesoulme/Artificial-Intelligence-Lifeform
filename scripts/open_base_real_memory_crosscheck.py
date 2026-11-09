"""Separate unseen-key and unseen-prompt effects for one trained adapter."""
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
from scripts.open_base_synthetic_alignment import choice_ids
from scripts.open_base_real_memory_alignment import TRAIN_PAIRS, HELD_PAIRS, TRAIN_PROMPTS, HELD_PROMPTS, variants, evaluate
from spora.growing import load
from spora.language_training import frozen_digest
from spora.open_base_memory import OpenBaseMemory

ADAPTER = ROOT/'runs/open-base-real-memory-addressed-20261109/adapter.pt'
OUTPUT = ROOT/'runs/open-base-real-memory-crosscheck-20261109.json'


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA or sha(INDIVIDUAL) != INDIVIDUAL_SHA:
        raise ValueError('Protected source mismatch')
    adapter_sha = sha(ADAPTER)
    checkpoint = torch.load(ADAPTER, map_location='cpu', weights_only=True)
    if (checkpoint.get('format') != 'spora-open-base-real-memory-v1'
            or checkpoint.get('weight_sha256') != WEIGHT_SHA or checkpoint.get('individual_sha256') != INDIVIDUAL_SHA
            or checkpoint.get('layers') != [5, 17] or checkpoint.get('current_only') is not True):
        raise ValueError('Adapter provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(organism, original)
    states, events = variants(organism, original, TRAIN_PAIRS + HELD_PAIRS)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='eager').to('cuda').eval()
    adapter = OpenBaseMemory(model.config.hidden_size, organism.config.memory_width).to('cuda').eval()
    adapter.load_state_dict(checkpoint['state'], strict=True)
    layers = [model.model.layers[5], model.model.layers[17]]
    choices = choice_ids(tokenizer)
    train = {pair: states[pair] for pair in TRAIN_PAIRS}
    held = {pair: states[pair] for pair in HELD_PAIRS}
    result = {'known_keys_new_prompts': evaluate(model, layers, adapter, tokenizer, choices, train, HELD_PROMPTS, current_only=True),
              'new_keys_known_prompts': evaluate(model, layers, adapter, tokenizer, choices, held, TRAIN_PROMPTS, current_only=True),
              'new_keys_new_prompts': evaluate(model, layers, adapter, tokenizer, choices, held, HELD_PROMPTS, current_only=True)}
    if (frozen_digest(organism, original) != before or sha(INDIVIDUAL) != INDIVIDUAL_SHA
            or sha(MODEL/'model.safetensors') != WEIGHT_SHA or sha(ADAPTER) != adapter_sha):
        raise ValueError('Protected input changed')
    report = {'status': 'development_diagnostic_not_acceptance', 'revision': REVISION, 'weight_sha256': WEIGHT_SHA,
              'individual_sha256': INDIVIDUAL_SHA, 'adapter_sha256': adapter_sha, 'current_only': True,
              'train_pairs': TRAIN_PAIRS, 'held_pairs': HELD_PAIRS, 'events_verified': len(events),
              'summary': result, 'seconds': time.perf_counter()-start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Keys and prompts have been used for development; results are not independent acceptance.',
                              'Each pair is derived from the same original background and two controlled events.',
                              'Score compares only the logits of 1 and 2, not unrestricted generation.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
