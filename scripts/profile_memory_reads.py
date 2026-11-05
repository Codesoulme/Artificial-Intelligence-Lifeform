"""Interleaved CPU/GPU timing, including one-position reads; no model updates."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit_memory_reads import interface, compare, digest
from spora.growing import load, grow, GrowingState


def paired(functions, device, repeats=21):
    names = list(functions)
    for _ in range(3):
        for function in functions.values():
            function()
    values = {name: [] for name in names}
    for repeat in range(repeats):
        offset = repeat % len(names)
        for name in names[offset:] + names[:offset]:
            if device == 'cuda':
                torch.cuda.synchronize()
            started = time.perf_counter()
            functions[name]()
            if device == 'cuda':
                torch.cuda.synchronize()
            values[name].append((time.perf_counter()-started)*1000)
    return {name: {'median_ms': statistics.median(times), 'p10_ms': sorted(times)[2],
                   'p90_ms': sorted(times)[-3], 'repeats': repeats} for name, times in values.items()}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(2)
    torch.manual_seed(2114000)
    parent = ROOT/'checkpoints/individual-prelanguage.pt'
    interface_path = ROOT/'runs/p7-preparation/untrained-interface.pt'
    before = [digest(parent), digest(interface_path)]
    payload = torch.load(interface_path, weights_only=True)
    result = {'seed': 2114000, 'ordering': 'three modes interleaved, rotated every repetition',
              'warmup': 3, 'training_steps': 0, 'rows': []}
    for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
        for layout in ('candidate', 'occupied4'):
            source = parent if layout == 'candidate' else ROOT/'runs/p6-seed7-entities64/world0.pt'
            model, base, _ = load(source, device)
            bridge = interface(model, payload)
            for pages in (1, 4, 8, 16):
                if layout == 'occupied4' and pages < 4:
                    continue
                state = (GrowingState(base.current, base.pages[:pages], base.cursor) if pages <= 4
                         else grow(model, base, pages-4))
                for batch, length in ((1, 1), (4, 128)):
                    tokens = torch.randint(0, 256, (batch, length), device=device)
                    correctness = compare(bridge, state, tokens, 1e-4)
                    def call(mode):
                        bridge.read_mode = mode
                        return bridge(tokens, state)
                    timing = paired({mode: lambda mode=mode: call(mode)
                                     for mode in ('legacy', 'exact', 'bounded')}, device)
                    row = {'device': device, 'layout': layout, 'pages': pages,
                           'batch': batch, 'length': length, 'timing': timing,
                           'correctness': correctness,
                           'exact_speedup': timing['legacy']['median_ms']/timing['exact']['median_ms']}
                    result['rows'].append(row)
                    print(json.dumps({'device': device, 'layout': layout, 'pages': pages,
                                      'shape': [batch, length], 'speedup': row['exact_speedup']}), flush=True)
    result['protected_files_unchanged'] = before == [digest(parent), digest(interface_path)]
    result['passed'] = result['protected_files_unchanged'] and all(
        r['passed'] for row in result['rows'] for r in row['correctness'].values())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
