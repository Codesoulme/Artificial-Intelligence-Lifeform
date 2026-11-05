"""Fixed P8 correctness and timing audit; no text training or individual writes."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spora.growing import load, grow, GrowingState
from spora.language_bridge import LanguageBridge
from spora.memory_read import MemoryReadPlan


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def legacy_read(matrices, q, keys):
    values = torch.stack([torch.einsum('bmd,bnd->bnm', matrix.expand(q.shape[0], -1, -1), q)
                          for matrix in matrices], 2)
    scores = (values*q.unsqueeze(2)).sum(-1)/q.shape[-1]**.5
    scores = scores+torch.einsum('bnm,pm->bnp', q, keys)/q.shape[-1]**.5
    return (values*scores.softmax(2).unsqueeze(-1)).sum(2)


def measured(function, device, repeats=15):
    for _ in range(3):
        function()
    if device == 'cuda':
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        if device == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        function()
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter()-started)*1000)
    return {'median_ms': statistics.median(times), 'min_ms': min(times), 'repeats': repeats,
            'peak_additional_cuda_bytes': torch.cuda.max_memory_allocated()-baseline if device == 'cuda' else None}


def interface(organism, payload):
    bridge = LanguageBridge(organism)
    state = bridge.state_dict()
    state.update(payload['state'])
    bridge.load_state_dict(state)
    return bridge.to(next(organism.parameters()).device)


def compare(bridge, state, tokens, tolerance):
    bridge.read_mode = 'legacy'
    expected = bridge(tokens, state)
    rows = {}
    for mode in ('exact', 'bounded'):
        bridge.read_mode = mode
        actual = bridge(tokens, state)
        row = {'max_abs_logit_error': float((expected-actual).abs().max()),
               'top1_agreement': float((expected.argmax(-1)==actual.argmax(-1)).float().mean()),
               'reads': bridge.last_read_stats}
        row['passed'] = row['max_abs_logit_error'] <= tolerance and row['top1_agreement'] == 1
        rows[mode] = row
    return rows


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'all'), default='all')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    plan = json.loads((ROOT/'configs/p8-read-plan.json').read_text())
    torch.set_num_threads(2)
    torch.manual_seed(plan['seed'])
    protected = [ROOT/'checkpoints'/name for name in
                 ('individual.pt', 'individual-p2-candidate.pt', 'individual-p3-candidate.pt',
                  'individual-p5b-candidate.pt', 'individual-prelanguage.pt')]
    interface_path = ROOT/'runs/p7-preparation/untrained-interface.pt'
    protected.append(interface_path)
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    payload = torch.load(interface_path, weights_only=True)
    assert payload['trained_tokens'] == 0
    report = {'plan': plan, 'hashes_before': hashes, 'language_training_steps': 0,
              'read_tests': [], 'end_to_end': [], 'benchmarks': [], 'device_names': {},
              'limitations': ['random bytes assess numerical preservation, not language ability',
                             'selective fixtures test routing mechanism, not learned relevance',
                             'eight/sixteen-page allocation tests add unoccupied pages',
                             'error bounds concern each memory read, not a formal final-logit bound',
                             'prepared plans copy derived tensors in RAM; no persistent size reduction']}
    devices = (['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
               if args.device == 'all' else [args.device])
    for device in devices:
        report['device_names'][device] = torch.cuda.get_device_name() if device == 'cuda' else 'CPU, two PyTorch threads'
        # Fresh numerical queries; concentrated, ambiguous and misleading scores.
        for seed in plan['audit_seeds']:
            print(json.dumps({'device':device,'numerical_seed':seed}),flush=True)
            torch.manual_seed(seed)
            for pages in (4, 8, 16):
                matrices = [torch.randn(1, 90, 90, device=device)*.02 for _ in range(pages+1)]
                for case in ('ambiguous', 'selective', 'misleading', 'random_selective'):
                    q = torch.zeros(2, 32, 90, device=device); q[..., 0] = 1
                    keys = torch.randn(pages+1, 90, device=device)*.01
                    if case not in ('ambiguous', 'random_selective'):
                        keys[1, 0] = 280
                    if case == 'random_selective':
                        q = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
                        keys = torch.randn_like(keys)*600
                    banks = [x.clone() for x in matrices]
                    if case == 'misleading':
                        banks[-1][0, 0, 0] = 600
                    reader = MemoryReadPlan(banks)
                    actual, stats = reader.read(q, keys, 'bounded', plan['read_error_l2'])
                    expected = legacy_read(banks, q, keys)
                    error = float((expected-actual).norm(dim=-1).max())
                    row = {'device': device, 'seed': seed, 'pages': pages, 'case': case,
                           'max_l2_error': error, 'stats': stats,
                           'passed': error <= plan['read_error_l2']+1e-5 and
                                     (case != 'ambiguous' or stats['pruned_queries'] == 0)}
                    report['read_tests'].append(row)
        # Forty-eight existing independently audited individuals, original weights.
        for seed in (7, 11, 23):
            for entities in (4, 64):
                for world in range(8):
                    path = ROOT/f'runs/p6-seed{seed}-entities{entities}/world{world}.pt'
                    organism, state, _ = load(path, device)
                    bridge = interface(organism, payload)
                    tokens = torch.randint(0, 256, (2, 32), device=device)
                    report['end_to_end'].append({'device': device, 'seed': seed,
                        'entities': entities, 'world': world,
                        'results': compare(bridge, state, tokens, plan['end_to_end_logit_max_abs'])})
        print(json.dumps({'device': device, 'numerical_audit_complete': True}), flush=True)

        # Time both current sparse allocation and fully occupied four-page history.
        for layout in ('candidate', 'occupied4'):
            source = ROOT/('checkpoints/individual-prelanguage.pt' if layout == 'candidate'
                          else 'runs/p6-seed7-entities64/world0.pt')
            organism, base, _ = load(source, device)
            bridge = interface(organism, payload)
            for pages in plan['page_counts']:
                if layout == 'occupied4' and pages < 4:
                    continue
                state = (GrowingState(base.current, base.pages[:pages], base.cursor)
                         if pages <= len(base.pages) else grow(organism, base, pages-len(base.pages)))
                tokens = torch.randint(0, 256, (4, 128), device=device)
                q = torch.nn.functional.normalize(torch.randn(4, 128, 90, device=device), dim=-1)
                matrices = [state.current[0]] + [page[0][0] for page in state.pages]
                keys = bridge.page_key(torch.arange(len(matrices), device=device))
                if device == 'cuda':
                    torch.cuda.synchronize()
                started = time.perf_counter()
                reader = MemoryReadPlan(matrices)
                if device == 'cuda':
                    torch.cuda.synchronize()
                preparation_ms = (time.perf_counter()-started)*1000
                row = {'device': device, 'layout': layout, 'pages': pages, 'batch': 4, 'length': 128,
                       'prepare_ms': preparation_ms, 'read_only': {}, 'full_interface': {},
                       'correctness': compare(bridge, state, tokens, plan['end_to_end_logit_max_abs'])}
                row['read_only']['legacy'] = measured(lambda: legacy_read(matrices, q, keys), device)
                for mode in ('exact', 'bounded'):
                    row['read_only'][mode] = measured(lambda: reader.read(q, keys, mode), device)
                    _, stats = reader.read(q, keys, mode)
                    row['read_only'][mode]['stats'] = stats
                row['plan_tensor_bytes_after_bounds'] = reader.tensor_bytes
                for mode in ('legacy', 'exact', 'bounded'):
                    bridge.read_mode = mode
                    row['full_interface'][mode] = measured(lambda: bridge(tokens, state), device, repeats=9)
                    row['full_interface'][mode]['stats'] = bridge.last_read_stats
                report['benchmarks'].append(row)
                print(json.dumps({'device': device, 'layout': layout, 'pages': pages,
                      'full_ms': {k: round(v['median_ms'], 3) for k, v in row['full_interface'].items()}}), flush=True)
    report['hashes_after'] = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    report['passed'] = (hashes == report['hashes_after'] and
                        all(x['passed'] for x in report['read_tests']) and
                        all(r['passed'] for x in report['end_to_end'] for r in x['results'].values()) and
                        all(r['passed'] for x in report['benchmarks'] for r in x['correctness'].values()))
    (output/'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'passed': report['passed'], 'output': str(output/'audit.json')}), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
