"""Clock integration, real bounded execution, lineage and transaction audits."""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spora.growing import load, save, observe, query
from spora.online_time import OnlineState, OnlineSession
from spora.durable import TensorStore
from spora.state_codec import flatten
from spora.sleep_memory import consolidate_store
from audit_durable import fingerprint, worker_line
from audit_time_world import load_module


def protected_files():
    previous = json.loads((ROOT/'reports/P9-P10-results.json').read_text())
    hashes = dict(previous['protected_hashes'])
    hashes.update({'checkpoints/'+name: info['sha256'] for name, info in previous['candidate']['files'].items()})
    for name, expected in hashes.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise AssertionError('Protected file changed: '+name)
    return hashes


@torch.no_grad()
def history_equal(model, state, other, after):
    count = model.config.time_count*len(state.pages)
    keys = torch.arange(model.config.key_count).repeat_interleave(count)
    times = torch.arange(count).repeat(model.config.key_count)
    for start in range(0, len(keys), 128):
        k, t = keys[start:start+128], times[start:start+128]
        modes = torch.ones_like(k)
        if not torch.equal(query(model, state, k, modes, t), query(other, after, k, modes, t)):
            return False
    k = torch.arange(model.config.key_count)
    return torch.equal(query(model, state, k, torch.zeros_like(k), torch.zeros_like(k)),
                       query(other, after, k, torch.zeros_like(k), torch.zeros_like(k)))


@torch.no_grad()
def changed_snapshot(model, state):
    # Change weights, working state, clock metadata and symbolic cursor together.
    clock = [0]
    session = OnlineSession(model.time_world, state.runtime, lambda: clock[0], lambda: 2*10**12+clock[0])
    session.observe_world([0., .4, -.2], [.5, .2], [.4, .3], 'synthetic')
    model.time_world.period_head[-1].bias.add_(.125)
    clock[0] = 300_000_000
    session.tick()
    state.runtime.checkpoint_utc_ns = 2*10**12+clock[0]
    return observe(model, state, 12, 8, 1)


def child(path, stage):
    torch.set_num_threads(2)
    with TensorStore(path) as store:
        model, state, _, revision = store.read()
        state = changed_snapshot(model, state)
        def hook(current):
            if current == stage:
                print(json.dumps({'paused': stage}), flush=True)
                sys.stdin.readline()
        store.commit(model, state, revision, hook=hook)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output')
    p.add_argument('--child')
    p.add_argument('--stage')
    a = p.parse_args()
    if a.child:
        return child(a.child, a.stage)
    output = Path(a.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    protected = protected_files()
    if not json.loads((ROOT/'runs/p11-time-world-audit/audit.json').read_text())['passed']:
        raise RuntimeError('Neural audit failed')
    parent = ROOT/'checkpoints/individual-sleep-candidate.pt'
    original, old, meta = load(parent)
    model, state, _ = load(parent)
    model.time_world, _ = load_module(ROOT/'runs/p11-seed7/model.pt')
    state.runtime = OnlineState()
    base_path = output/'baseline.sqlite'
    with TensorStore.create(base_path, model, state, dict(meta, stage='P11 timing audit')) as store:
        base_model, base_state, _, _ = store.read()
        baseline_exact = fingerprint(model, state) == fingerprint(base_model, base_state)
        store.checkpoint()
        store.backup(output/'real-run.sqlite')
    result = {'protected_hashes': protected, 'restart_exact': baseline_exact,
              'history_logits_exact_after_attachment': history_equal(original, old, model, state),
              'crashes': []}
    # Separate process proves a cold-loaded module runs without user messages.
    command = [sys.executable, '-m', 'spora.online_cli', '--individual', str(output/'real-run.sqlite'),
               'run', '--seconds', '2.2', '--tick-ms', '50', '--synthetic-pulse-period', '.3', '--demo-world']
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    real = json.loads(completed.stdout)
    (output/'real-run.json').write_text(json.dumps(real, indent=2))
    with TensorStore(output/'real-run.sqlite') as store:
        rm, rs, _, _ = store.read()
        result['history_logits_exact_after_idle'] = history_equal(original, old, rm, rs)
        schema, tensors = flatten(rm, rs)
        save(output/'real-run.pt', rm, rs, store.metadata)
        restored, restored_state, _ = load(output/'real-run.pt')
        result['runtime_portable_exact'] = fingerprint(rm, rs) == fingerprint(restored, restored_state)
        total_tensor_bytes = sum(t.numel()*t.element_size() for t in tensors.values())
        runtime_bytes = sum(t.numel()*t.element_size() for k, t in tensors.items() if k.startswith('runtime/'))
        store.checkpoint()
    previous_counts = (0, 0)
    incremental_ok = True
    for commit in real['commits']:
        counts = (commit['world_observations'], commit['observed_pulses'])
        idle = counts == previous_counts
        commit['idle_without_new_observations'] = idle
        incremental_ok &= commit['changed_tensors'] <= (1 if idle else 5) and commit['tensor_bytes_written'] <= (12 if idle else 48)
        previous_counts = counts
    real['passed'] = (len(real['events']) == 1 and real['cursor'] == old.cursor
                      and real['status']['observed_pulses'] == 3
                      and real['status']['world_observations'] == 1
                      and real['status']['world_status'] == 'needs_observation'
                      and incremental_ok)
    result['real_run'] = real
    result['cost'] = {'new_parameters': sum(p.numel() for p in model.time_world.parameters()),
                      'new_weight_bytes': sum(p.numel()*p.element_size() for p in model.time_world.parameters()),
                      'working_state_tensor_bytes': runtime_bytes, 'total_tensor_bytes': total_tensor_bytes,
                      'metadata_note': 'JSON clocks, tensor manifest and SQLite page/WAL overhead are additional.'}
    stages = json.loads((ROOT/'configs/p9-p10-plan.json').read_text())['storage']['crash_stages']
    before = fingerprint(model, state)
    next_model, next_state = copy.deepcopy(model), copy.deepcopy(state)
    next_state = changed_snapshot(next_model, next_state)
    after = fingerprint(next_model, next_state)
    for stage in stages:
        path = output/f'crash-{stage}.sqlite'
        with TensorStore(base_path) as base:
            base.backup(path)
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--child', str(path), '--stage', stage],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            marker = worker_line(process)
            if not marker or json.loads(marker).get('paused') != stage:
                raise RuntimeError('Worker missed commit boundary')
            expected = after if stage == 'after_commit' else before
            with TensorStore(path) as reader:
                m, s, _, _ = reader.read()
                reader_exact = fingerprint(m, s) == expected
            process.kill()
            process.wait(timeout=10)
            with TensorStore(path) as recovered:
                m, s, _, rev = recovered.read()
                exact = fingerprint(m, s) == expected
                reference_m, reference_s = (next_model, next_state) if stage == 'after_commit' else (model, state)
                next_write = observe(m, s, 13, 4, 1)
                reference_write = observe(reference_m, reference_s, 13, 4, 1)
                continuation = fingerprint(m, next_write) == fingerprint(reference_m, reference_write)
                recovered.commit(m, next_write, rev)
            row = {'stage': stage, 'reader_exact': reader_exact, 'recovery_exact': exact, 'next_write_exact': continuation}
            result['crashes'].append(row)
            print(json.dumps(row), flush=True)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()
    # Run another actual sleep on an audit-only descendant after attaching time.
    with TensorStore(base_path) as store:
        store.backup(output/'sleep.sqlite')
    with TensorStore(output/'sleep.sqlite') as store:
        m, s, _, rev = store.read()
        before_time = {k: v.clone() for k, v in m.time_world.state_dict().items()}
        runtime_metadata = s.runtime.metadata()
        for i in range(s.cursor, 128):
            s = observe(m, s, i % m.config.key_count, i % m.config.unknown, 1)
        store.commit(m, s, rev)
        sleep = consolidate_store(store, m, s, store.revision, 1, rank=16, steps=600, seed=2319000)
        sm, ss, _, _ = store.read()
        sleep['time_weights_exact'] = all(torch.equal(v, sm.time_world.state_dict()[k]) for k, v in before_time.items())
        sleep['runtime_exact'] = runtime_metadata == ss.runtime.metadata() and all(torch.equal(v, ss.runtime.tensors[k]) for k, v in s.runtime.tensors.items())
        result['sleep_compatibility'] = sleep
    protected_files()
    result['originals_unchanged'] = True
    result['passed'] = (result['restart_exact'] and result['history_logits_exact_after_attachment']
                        and result['history_logits_exact_after_idle'] and result['runtime_portable_exact']
                        and real['passed'] and sleep['accepted'] and sleep['time_weights_exact'] and sleep['runtime_exact']
                        and all(r['reader_exact'] and r['recovery_exact'] and r['next_write_exact'] for r in result['crashes']))
    (output/'audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'passed': result['passed'], 'real_run': real, 'cost': result['cost']}), flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
