"""Audit free generation across all eight finite values, without training."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import scripts.deepseek_semantic_value_pilot as pilot
from scripts.deepseek_named_binding_probe import KEYS, NAMES, UNKNOWN
from scripts.deepseek_named_incremental_probe import NEW_KEYS
from scripts.deepseek_named_readout_alignment import snapshot
from scripts.deepseek_read_strength_probe import ScaledReadout
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load
from spora.language_training import frozen_digest

ADAPTER = ROOT / 'runs/deepseek-semantic-value-continuation-20261110/adapter.pt'
OUTPUT = ROOT / 'runs/deepseek-semantic-value-all-values-audit-20261110.json'


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    paths = (pilot.MODEL / 'model.safetensors', INDIVIDUAL, ADAPTER)
    before = {p: sha(p) for p in paths}
    if before[paths[0]] != pilot.WEIGHT_SHA or before[INDIVIDUAL] != INDIVIDUAL_SHA:
        raise ValueError('Protected input mismatch')
    checkpoint = torch.load(ADAPTER, map_location='cpu', weights_only=True)
    if (checkpoint.get('format') != 'deepseek-spora-semantic-value-development-v1'
            or checkpoint.get('weight_sha256') != pilot.WEIGHT_SHA
            or checkpoint.get('individual_sha256') != INDIVIDUAL_SHA
            or checkpoint.get('layers') != [5, 17] or checkpoint.get('strength') != 0.5):
        raise ValueError('Adapter provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    digest = frozen_digest(organism, original)
    names = (NAMES[0], NAMES[2], UNKNOWN[0], UNKNOWN[1])
    keys = (KEYS[0], KEYS[2], NEW_KEYS[0], NEW_KEYS[1])
    states = {(key, value): snapshot(organism, key, value) for key in keys for value in range(8)}
    state_digests = {key: frozen_digest(organism, value) for key, value in states.items()}
    tokenizer = AutoTokenizer.from_pretrained(pilot.MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        pilot.MODEL, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    adapter = ScaledReadout(base.config.hidden_size, organism.config.memory_width, 0.5).to('cuda').eval()
    adapter.load_state_dict(checkpoint['state'], strict=True)
    bases = [F.normalize(site.current_basis.to('cuda').float(), dim=-1) for site in organism.memories]
    results = []
    old_values = pilot.HELD_VALUES
    try:
        for pair in ((0, 1), (2, 3), (4, 5), (6, 7)):
            pilot.HELD_VALUES = pair
            result = pilot.evaluate(base, tokenizer, adapter, bases, states, names, keys)
            results.append({'values': pair, **result})
            print(json.dumps({'values': pair, 'summary': result['summary']}), flush=True)
    finally:
        pilot.HELD_VALUES = old_values
    if (any(sha(p) != before[p] for p in paths) or frozen_digest(organism, original) != digest
            or any(frozen_digest(organism, state) != state_digests[key] for key, state in states.items())):
        raise ValueError('Protected input changed')
    report = {'status': 'development_only', 'adapter_sha256': before[ADAPTER], 'results': results,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Artificially supplied addresses and finite values; not open fact memory.',
                              'Names and templates are development examples; not independent validation.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    run()
