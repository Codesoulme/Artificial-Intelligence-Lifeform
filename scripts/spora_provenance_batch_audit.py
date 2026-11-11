"""Compare scalar and batched neural provenance on the same reloaded state."""
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import capacity, grow, load, save
from spora.internal_memory import observe_sourced, read_provenance, read_provenance_batch
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/spora-provenance-batch-audit-v2-20261111.json'


def child(checkpoint, output):
    torch.set_num_threads(2)
    model, state, _ = load(checkpoint, 'cpu')
    pairs = [(key, event) for key in range(18) for event in range(state.cursor)]
    pairs.extend([(0, state.cursor), (17, state.cursor + 5)])
    result = read_provenance_batch(model, state, pairs, batch_size=16)
    Path(output).write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(INDIVIDUAL) != INDIVIDUAL_SHA:
        raise ValueError('Protected individual mismatch')
    torch.set_num_threads(2)
    model, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(model, original)
    rng = random.Random(8303)
    state = fresh_controlled_state(model)
    writes = []
    for _ in range(96):
        if state.cursor == capacity(model, state):
            state = grow(model, state)
        key, value = rng.randrange(16), rng.randrange(model.config.unknown)
        source = rng.choices(('direct', 'report', 'inference'), weights=(7, 2, 1))[0]
        state, decision = observe_sourced(model, state, key, value, source, 1.0)
        writes.append({'key': key, 'value': value, 'source': source, **decision})
    pairs = [(key, event) for key in range(18) for event in range(state.cursor)]
    pairs.extend([(0, state.cursor), (17, state.cursor + 5)])
    start = time.perf_counter()
    scalar = [read_provenance(model, state, key, event) for key, event in pairs]
    scalar_seconds = time.perf_counter()-start
    timings, outputs = {}, {}
    for size in (1, 8, 16, 32):
        start = time.perf_counter()
        outputs[size] = read_provenance_batch(model, state, pairs, batch_size=size)
        timings[str(size)] = time.perf_counter()-start
    mismatches = {str(size): [i for i, (a, b) in enumerate(zip(scalar, value)) if a != b]
                  for size, value in outputs.items()}
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)/'individual.pt'
        save(path, model, state, {'purpose': 'provenance-batch-audit'})
        model2, state2, _ = load(path, 'cpu')
        reloaded = read_provenance_batch(model2, state2, pairs, batch_size=16)
        reload_mismatches = [i for i, (a, b) in enumerate(zip(outputs[16], reloaded)) if a != b]
        digest_equal = frozen_digest(model, state) == frozen_digest(model2, state2)
        child_output = Path(temp)/'child.json'
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child', str(path), str(child_output)],
                       check=True, timeout=90)
        separate = json.loads(child_output.read_text(encoding='utf-8'))
        process_mismatches = [i for i, (a, b) in enumerate(zip(outputs[16], separate)) if a != b]
        if len(separate) != len(outputs[16]):
            raise ValueError('Independent process returned wrong result count')
    if sha(INDIVIDUAL) != INDIVIDUAL_SHA or frozen_digest(model, original) != before:
        raise ValueError('Protected individual changed')
    report = {'status': 'bounded_batch_equivalence_development_only', 'seed': 8303,
              'individual_sha256': INDIVIDUAL_SHA, 'writes': writes, 'query_count': len(pairs),
              'scalar_seconds': scalar_seconds, 'batch_seconds': timings,
              'mismatch_indices_vs_scalar': mismatches, 'reload_mismatch_indices': reload_mismatches,
              'independent_process_mismatch_indices': process_mismatches, 'reload_digest_equal': digest_equal,
              'limitations': ['CPU, one individual and one synthetic trajectory.',
                              'Timing excludes state loading and generation; no GPU benchmark.',
                              'Equivalence does not establish semantic correctness or openness.',
                              'Independent-process reload is not an abrupt termination test.',
                              'Numerical threshold boundaries may differ by hardware and batch size.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('writes', 'limitations')}), flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--child':
        child(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 1:
        run()
    else:
        raise SystemExit('Invalid arguments')
