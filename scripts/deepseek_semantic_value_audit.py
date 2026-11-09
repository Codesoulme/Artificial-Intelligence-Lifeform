"""Audit free generation across all eight finite values, without training."""
import argparse
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
from spora.growing import load, query
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest

ADAPTER = ROOT / 'runs/deepseek-semantic-value-continuation-20261110/adapter.pt'
OUTPUT = ROOT / 'runs/deepseek-semantic-value-all-values-audit-20261110.json'


def run(adapter_path=ADAPTER, output=OUTPUT, fresh=False, distractor=None, correction=False):
    runs = (ROOT / 'runs').resolve()
    adapter_path, output = Path(adapter_path).resolve(), Path(output).resolve()
    if not adapter_path.is_relative_to(runs) or not output.is_relative_to(runs) or output.exists():
        raise ValueError('Expected an adapter and a new output file under runs')
    paths = (pilot.MODEL / 'model.safetensors', INDIVIDUAL, adapter_path)
    before = {p: sha(p) for p in paths}
    if before[paths[0]] != pilot.WEIGHT_SHA or before[INDIVIDUAL] != INDIVIDUAL_SHA:
        raise ValueError('Protected input mismatch')
    checkpoint = torch.load(adapter_path, map_location='cpu', weights_only=True)
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
    names = ('林栩', '许澄', '江屿', '叶芮') if fresh else (NAMES[0], NAMES[2], UNKNOWN[0], UNKNOWN[1])
    keys = (2, 3, 4, 5) if fresh else (KEYS[0], KEYS[2], NEW_KEYS[0], NEW_KEYS[1])
    if fresh and (set(keys) & set(KEYS + NEW_KEYS) or set(names) & set(NAMES + UNKNOWN)):
        raise ValueError('Fresh audit overlaps prior development bindings')
    states, writes = {}, []
    for key in keys:
        for value in range(8):
            initial = (value + 1) % 8 if correction else value
            state = snapshot(organism, key, initial, distractor=distractor)
            if correction:
                state, decision = observe_sourced(organism, state, key, value, 'direct', 1.0)
                indices = torch.tensor([key], dtype=torch.long)
                with torch.no_grad():
                    actual = int(query(organism, state, indices, torch.zeros_like(indices),
                                       torch.zeros_like(indices)).argmax(-1)[0])
                writes.append({'key': key, 'old': initial, 'target': value, 'actual': actual,
                               'decision': decision})
            states[(key, value)] = state
    if correction:
        print(json.dumps({'correction_states': len(writes),
                          'write_accepted': sum(w['decision']['current_write'] for w in writes),
                          'native_target_correct': sum(w['actual'] == w['target'] for w in writes)}), flush=True)
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
    old_values, old_templates = pilot.HELD_VALUES, pilot.HELD_TEMPLATES
    try:
        if fresh:
            pilot.HELD_TEMPLATES = ('{name}最爱喝哪种饮料？', '说说{name}偏好喝的东西。')
        for pair in ((0, 1), (2, 3), (4, 5), (6, 7)):
            pilot.HELD_VALUES = pair
            result = pilot.evaluate(base, tokenizer, adapter, bases, states, names, keys)
            results.append({'values': pair, **result})
            print(json.dumps({'values': pair, 'summary': result['summary']}), flush=True)
    finally:
        pilot.HELD_VALUES = old_values
        pilot.HELD_TEMPLATES = old_templates
    if (any(sha(p) != before[p] for p in paths) or frozen_digest(organism, original) != digest
            or any(frozen_digest(organism, state) != state_digests[key] for key, state in states.items())):
        raise ValueError('Protected input changed')
    report = {'status': 'development_only', 'adapter_sha256': before[adapter_path], 'fresh_cases': fresh,
              'fixed_distractor': distractor,
              'correction': correction, 'writes': writes,
              'names': names, 'keys': keys, 'results': results,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Artificially supplied addresses and finite values; not open fact memory.',
                              'The manually written cases and fixed vocabulary are not open-domain validation.',
                              'The language model may have encountered similar text during pretraining.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, default=ADAPTER)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--distractor', type=int, choices=range(16))
    parser.add_argument('--correction', action='store_true')
    args = parser.parse_args()
    run(args.adapter, args.output, args.fresh, args.distractor, args.correction)
