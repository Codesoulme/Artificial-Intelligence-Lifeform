"""Probe whether TTT reads preserve value labels across held-out identities.

This is a supervised finite-label diagnostic, not an output adapter. Centered
features use all unlabeled values for each identity and are transductive.
"""
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch.nn import functional as F
from scripts import deepseek_local_baseline as p
from scripts import deepseek_spora_ttt_crossed_values_pilot as crossed
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_ttt_selective_train_pilot import OUT as SELECTED_SOURCE
from scripts.deepseek_spora_ttt_selective_write_audit import selected_memory
from spora.research_ttt_ntp import attach_read_adapters, install, use_memory

OUT = ROOT / 'runs/deepseek-spora-ttt-value-probe-20261114.json'
VALUES = crossed.TRAIN_VALUES + crossed.HELD_VALUES


def probe(features, labels, train_names, held_names, *, center=False, shuffle_seed=None):
    """Nearest centroid with separate source identities and optional centering."""
    x = features.clone().float()
    if center:
        # The test identity's unlabeled cohort is available to this diagnostic.
        x -= x.mean(dim=1, keepdim=True)
    source = x[:len(train_names)].reshape(-1, x.shape[-1])
    targets = x[len(train_names):].reshape(-1, x.shape[-1])
    training_labels = list(labels) * len(train_names)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(training_labels)
    label_tensor = torch.tensor(training_labels)
    prototypes = torch.stack([source[label_tensor == i].mean(0) for i in labels])
    scores = F.normalize(targets, dim=-1) @ F.normalize(prototypes, dim=-1).T
    predicted = scores.argmax(-1)
    expected = torch.tensor(list(labels) * len(held_names))
    return predicted.tolist(), int((predicted == expected).sum())


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    sources = {'whole': crossed.OUT, 'selected': SELECTED_SOURCE}
    metadata, payloads, hashes = {}, {}, {}
    for mode, source in sources.items():
        metadata[mode] = json.loads((source / 'result.json').read_text(encoding='utf-8'))
        hashes[mode] = p.sha(source / 'adapters.pt')
        payloads[mode] = torch.load(source / 'adapters.pt', map_location='cpu', weights_only=True)
        if hashes[mode] != metadata[mode]['adapter_sha256'] or payloads[mode]['base_sha256'] != p.WEIGHT_SHA:
            raise ValueError('Candidate provenance mismatch')
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Unexpected base weight')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    wrappers = install(model, tuple(payloads['whole']['layers']))
    adapters = attach_read_adapters(wrappers, rank=payloads['whole']['rank'])
    model.requires_grad_(False)
    names = crossed.TRAIN_NAMES + crossed.HELD_NAMES
    all_features = {}
    rows = []
    for mode in ('whole', 'selected'):
        for layer, adapter in adapters.items():
            adapter.load_state_dict(payloads[mode]['adapters'][str(layer)], strict=True)
        per_name = []
        for name in names:
            question = crossed.TEMPLATES[0].format(name=name)
            prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
            per_value = []
            for value in VALUES:
                fact = f'{name}喜欢喝{value}。'
                if mode == 'whole':
                    memory = pilot.device_memory(pilot.make_memory(model, tokenizer, wrappers, fact))
                else:
                    memory, _ = selected_memory(model, tokenizer, wrappers, fact, value)
                captured = {}
                handles = []
                for layer, adapter in adapters.items():
                    def capture(module, args, output, layer=layer):
                        captured[layer] = args[0][0, -1].detach().float().cpu()
                    handles.append(adapter.register_forward_hook(capture))
                try:
                    with torch.inference_mode(), use_memory(memory):
                        model(**prompt, use_cache=False, return_dict=True)
                finally:
                    for handle in handles:
                        handle.remove()
                if set(captured) != set(adapters):
                    raise ValueError('Missing layer feature')
                per_value.append(torch.cat([captured[layer] for layer in sorted(captured)]))
            per_name.append(torch.stack(per_value))
            print(json.dumps({'mode': mode, 'name': name}, ensure_ascii=False), flush=True)
        all_features[mode] = torch.stack(per_name)
    labels = list(range(len(VALUES)))
    for mode in ('whole', 'selected'):
        for centered in (False, True):
            predictions, correct = probe(all_features[mode], labels, crossed.TRAIN_NAMES,
                                         crossed.HELD_NAMES, center=centered)
            shuffled = [probe(all_features[mode], labels, crossed.TRAIN_NAMES,
                              crossed.HELD_NAMES, center=centered, shuffle_seed=seed)[1]
                        for seed in range(200)]
            rows.append({'mode': mode, 'centered_transductive': centered,
                         'correct': correct, 'total': len(predictions),
                         'predictions': [VALUES[i] for i in predictions],
                         'expected': list(VALUES) * len(crossed.HELD_NAMES),
                         'shuffled_mean_correct': sum(shuffled) / len(shuffled),
                         'shuffled_at_least_observed': sum(n >= correct for n in shuffled) / len(shuffled)})
    if p.sha(p.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Base weight changed')
    for mode, source in sources.items():
        if p.sha(source / 'adapters.pt') != hashes[mode]:
            raise ValueError('Candidate weight changed')
    report = {'status': 'cross_identity_finite_value_probe', 'rows': rows,
              'values': list(VALUES), 'source_names': list(crossed.TRAIN_NAMES),
              'held_names': list(crossed.HELD_NAMES), 'candidate_sha256': hashes,
              'base_sha256': base_sha, 'seconds': time.perf_counter() - start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['All six values, including adapter-held values, are labeled in probe training.',
                              'Centered variant uses all six unlabeled observations from each held identity.',
                              'Selected writes use the ground-truth value token span.',
                              'One question template and one fact template; no open-world semantic test.',
                              'Probe accuracy is not the language model answer accuracy or causal proof.',
                              'Permutations are a descriptive control on a tiny correlated development set.']}
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(rows, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
