"""Forced process termination and bounded commit-cost audit of neural storage."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import queue
import threading

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spora.durable import TensorStore, adopt, canonical
from spora.growing import load, save, observe, grow, GrowingState
from spora.state_codec import flatten


def fingerprint(model, state):
    schema, tensors = flatten(model, state)
    digest = hashlib.sha256(canonical(schema).encode())
    for name, value in sorted(tensors.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def size(path):
    return sum(p.stat().st_size for p in (path, Path(str(path)+'-wal'), Path(str(path)+'-shm')) if p.exists())


def worker_line(process, timeout=45):
    """Bounded marker wait; caller's finally block can always kill the worker."""
    result = queue.Queue()
    threading.Thread(target=lambda: result.put(process.stdout.readline()), daemon=True).start()
    return result.get(timeout=timeout)


def child(path, stage):
    torch.set_num_threads(2)
    with TensorStore(path) as store:
        model, state, _, revision = store.read()
        state = observe(model, state, 12, 7, 1)
        def hook(at):
            if at == stage:
                print(json.dumps({'paused_at': at}), flush=True)
                sys.stdin.readline()
        store.commit(model, state, revision, hook=hook)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('--child')
    parser.add_argument('--stage')
    args = parser.parse_args()
    if args.child:
        return child(args.child, args.stage)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True); torch.set_num_threads(2)
    plan = json.loads((ROOT/'configs/p9-p10-plan.json').read_text())
    source = ROOT/'checkpoints/individual-prelanguage.pt'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    model, original, meta = load(source)
    result = {'sqlite_policy': 'WAL, synchronous=FULL, autocheckpoint=256',
              'parent_sha256': source_hash, 'crashes': [], 'benchmarks': [],
              'limitations': ['process termination, not actual power-cut testing',
                             'timing includes WAL and periodic checkpoint work',
                             'payload counters exclude SQLite page/journal amplification',
                             'eight/sixteen page states include empty allocation']}
    baseline = output/'baseline.sqlite'
    with adopt(source, baseline) as initial:
        a, b, _, _ = initial.read()
        assert fingerprint(a, b) == fingerprint(model, original)
        initial.checkpoint()
    for stage in plan['storage']['crash_stages']:
        path = output/f'crash-{stage}.sqlite'
        with TensorStore(baseline) as store:
            store.backup(path)
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
            '--child', str(path.resolve()), '--stage', stage], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            line = worker_line(process)
            if not line or json.loads(line).get('paused_at') != stage:
                raise RuntimeError('Child failed before requested interruption: '+process.stderr.read())
            # A reader concurrent with an uncommitted writer sees the old whole state.
            with TensorStore(path) as reader:
                rm, rs, _, revision = reader.read()
                expected = observe(model, original, 12, 7, 1) if stage == 'after_commit' else original
                assert fingerprint(rm, rs) == fingerprint(model, expected)
            process.kill(); process.wait(timeout=10)
            with TensorStore(path) as recovered:
                rm, rs, _, revision = recovered.read()
                integrity = recovered.connection.execute('PRAGMA integrity_check').fetchone()[0]
                exact = fingerprint(rm, rs) == fingerprint(model, expected)
                next_a = observe(model, expected, 13, 8, 0)
                next_b = observe(rm, rs, 13, 8, 0)
                continuation = fingerprint(model, next_a) == fingerprint(rm, next_b)
                recovered.commit(rm, next_b, revision)
                backup = output/f'recovered-{stage}.sqlite'
                recovered.backup(backup)
                checkpoint = recovered.checkpoint()
            with TensorStore(backup) as replay:
                bm, bs, _, _ = replay.read()
                backup_exact = fingerprint(bm, bs) == fingerprint(model, next_a)
            row = {'stage': stage, 'expected_cursor': expected.cursor, 'recovered_cursor': rs.cursor,
                   'revision': revision, 'exact': exact, 'next_write_exact': continuation,
                   'reader_snapshot_exact': True, 'backup_exact': backup_exact,
                   'integrity': integrity, 'checkpoint': checkpoint,
                   'exit_code': process.returncode}
            result['crashes'].append(row); print(json.dumps(row), flush=True)
        finally:
            if process.poll() is None:
                process.kill(); process.wait(timeout=10)
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()
    for count in plan['storage']['page_counts']:
        state = (GrowingState(original.current, original.pages[:count], original.cursor) if count <= 4
                 else grow(model, original, count-4))
        full = output/f'full-{count}.pt'; database = output/f'incremental-{count}.sqlite'
        full_times, incremental_times, payload = [], [], []
        with TensorStore.create(database, model, state) as store:
            for index in range(plan['storage']['commit_repetitions']):
                state = observe(model, state, 12, index % 16, 1)
                if index % 2 == 0:
                    started = time.perf_counter(); save(full, model, state, meta)
                    full_times.append((time.perf_counter()-started)*1000)
                started = time.perf_counter()
                stats = store.commit(model, state, store.revision)
                incremental_times.append((time.perf_counter()-started)*1000); payload.append(stats)
                if index % 2:
                    started = time.perf_counter(); save(full, model, state, meta)
                    full_times.append((time.perf_counter()-started)*1000)
            before_checkpoint = size(database)
            started = time.perf_counter(); checkpoint = store.checkpoint()
            checkpoint_ms = (time.perf_counter()-started)*1000
            restored, rs, _, _ = store.read()
            assert fingerprint(model, state) == fingerprint(restored, rs)
            row = {'pages': count, 'commits': len(payload), 'full_median_ms': statistics.median(full_times),
                   'incremental_median_ms': statistics.median(incremental_times),
                   'full_file_bytes': full.stat().st_size, 'tensor_payload_bytes': [x['tensor_bytes_written'] for x in payload],
                   'changed_tensors': [x['changed_tensors'] for x in payload],
                   'database_with_wal_before_checkpoint': before_checkpoint,
                   'database_with_wal_after_checkpoint': size(database),
                   'checkpoint_ms': checkpoint_ms, 'checkpoint': checkpoint,
                   'raw_full_ms': full_times, 'raw_incremental_ms': incremental_times}
            result['benchmarks'].append(row); print(json.dumps(row), flush=True)
    result['parent_unchanged'] = hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    result['passed'] = result['parent_unchanged'] and all(
        r['exact'] and r['next_write_exact'] and r['backup_exact'] and r['integrity'] == 'ok' for r in result['crashes'])
    (output/'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
