"""Compare selective and legacy masked memory forwards on the same DeepSeek model."""
import json
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_integrated_checkpoint_probe import OUTPUT as CHECKPOINT
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.integrated_checkpoint import load_integrated
from spora.integrated_open_base import _MemoryDecoderLayer
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-read-mask-benchmark-20261110.json'


def legacy_forward(layer, *args, **kwargs):
    output = layer.layer(*args, **kwargs)
    active = layer._memory_context.get()
    if active is None:
        return output
    matrices, addresses, read_mask, memory_slots = active
    if memory_slots is not None:
        raise ValueError('Legacy read-mask benchmark does not cover temporal slots')
    address = None if addresses is None else addresses[layer.site]
    if read_mask is not None and read_mask.shape[0] not in (1, output.shape[0]):
        raise ValueError('Memory read mask must match the decoder batch')
    injected = layer._reader(output, matrices[layer.site], layer.site, address)
    if read_mask is None:
        return injected
    return torch.where(read_mask.to(device=output.device).reshape(-1, 1, 1), injected, output)


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = {'weight': MODEL / 'model.safetensors', 'original': INDIVIDUAL,
             'individual': CHECKPOINT / 'individual.pt', 'individual_report': CHECKPOINT / 'result.json'}
    before = {name: sha(path) for name, path in files.items()}
    if before['weight'] != WEIGHT_SHA or before['original'] != INDIVIDUAL_SHA:
        raise ValueError('Protected source mismatch')
    with files['individual_report'].open(encoding='utf-8') as stream:
        if json.load(stream)['checkpoint_sha256'] != before['individual']:
            raise ValueError('Integrated checkpoint provenance mismatch')
    config_sha = feature_files_sha256(MODEL)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    model, organism, state = load_integrated(files['individual'], base, MODEL)
    model.eval()
    digest = frozen_digest(organism, state)
    inputs = closed_think_inputs(tokenizer, f'{NAMES[0]}喜欢的饮品是什么？', 'cuda')
    inputs = {name: value.expand(2, -1) for name, value in inputs.items()}
    addresses = [F.normalize(site.current_basis.to('cuda').float(), dim=-1)[KEYS[0]:KEYS[0] + 1]
                 for site in organism.memories]
    masks = {'none': torch.tensor([False, False]), 'half': torch.tensor([True, False]),
             'all': torch.tensor([True, True])}
    original_forward = _MemoryDecoderLayer.forward
    timings = {name: {'legacy': [], 'selective': []} for name in masks}
    logits = {name: {} for name in masks}
    memory_args = {'memory_state': state, 'memory_addresses': addresses, 'current_only': True, 'use_cache': False}
    with torch.inference_mode():
        for mask in masks.values():
            model(**inputs, **memory_args, memory_read_mask=mask)
            with patch.object(_MemoryDecoderLayer, 'forward', legacy_forward):
                model(**inputs, **memory_args, memory_read_mask=mask)
        torch.cuda.synchronize()
        for iteration in range(8):
            for name, mask in masks.items():
                order = ('legacy', 'selective') if iteration % 2 == 0 else ('selective', 'legacy')
                for mode in order:
                    method = legacy_forward if mode == 'legacy' else original_forward
                    with patch.object(_MemoryDecoderLayer, 'forward', method):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        result = model(**inputs, **memory_args, memory_read_mask=mask).logits[:, -1].float()
                        torch.cuda.synchronize()
                        timings[name][mode].append(time.perf_counter() - start)
                        if iteration == 0:
                            logits[name][mode] = result.cpu()
            print(f'benchmark iteration {iteration + 1}/8', flush=True)
    comparisons = {}
    for name in masks:
        difference = float((logits[name]['legacy'] - logits[name]['selective']).abs().max())
        if not all(torch.isfinite(t).all() for t in logits[name].values()) or difference > 1e-4:
            raise ValueError(f'Legacy/selective output mismatch: {name}: {difference}')
        medians = {mode: statistics.median(values) for mode, values in timings[name].items()}
        comparisons[name] = {'max_logit_difference': difference, 'median_seconds': medians,
                             'selective_over_legacy': medians['selective'] / medians['legacy'],
                             'samples_seconds': timings[name]}
    if (any(sha(path) != before[name] for name, path in files.items()) or feature_files_sha256(MODEL) != config_sha
            or frozen_digest(organism, state) != digest or any(p.grad is not None for p in base.parameters())):
        raise ValueError('Protected input or state changed')
    report = {'status': 'local_read_mask_microbenchmark', 'comparisons': comparisons,
              'batch_size': 2, 'input_tokens': inputs['input_ids'].shape[1], 'iterations_per_mode': 8,
              'device': torch.cuda.get_device_name(0), 'component_sha256': before,
              'feature_config_sha256': config_sha, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Short duplicated prompt, one memory page and batch size two; not a chat or scaling benchmark.',
                              'Measures full prefill forwards including snapshot, not autoregressive generation or concurrency.',
                              'Legacy forward is reconstructed locally and swapped sequentially, not separate model processes.',
                              'Eight alternating samples per condition; GPU contention and thermal changes are uncontrolled.',
                              'Manual address and mask do not demonstrate automatic routing or memory safety.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({name: {'median_seconds': result['median_seconds'],
                             'selective_over_legacy': result['selective_over_legacy']}
                      for name, result in comparisons.items()}), flush=True)


if __name__ == '__main__':
    run()
