"""Attach the audited random-origin time module to a preserved individual copy."""
import hashlib
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spora.growing import load, save, query
from spora.online_time import OnlineState
from spora.durable import TensorStore
from spora.state_codec import flatten
from audit_durable import fingerprint
from audit_time_world import load_module
from audit_time_runtime import protected_files, history_equal


def main():
    torch.set_num_threads(2)
    protected_files()
    for source in ('runs/p11-time-world-audit/audit.json', 'runs/p11-runtime-audit-v2/audit.json'):
        if not json.loads((ROOT/source).read_text())['passed']:
            raise RuntimeError('P11 gate failed')
    parent = ROOT/'checkpoints/individual-sleep-candidate.pt'
    database = ROOT/'checkpoints/individual-time-candidate.sqlite'
    portable = ROOT/'checkpoints/individual-time-candidate.pt'
    if database.exists() or portable.exists():
        raise FileExistsError('Time candidate already exists')
    model, state, metadata = load(parent)
    _, previous_tensors = flatten(model, state)
    old_bytes = sum(v.numel()*v.element_size() for v in previous_tensors.values())
    source = ROOT/'runs/p11-seed7/model.pt'
    model.time_world, training = load_module(source)
    state.runtime = OnlineState()
    metadata = dict(metadata, stage='P11 elapsed time and finite dynamics',
                    parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
                    time_module_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    time_module_training=training, Chinese_training_started=False,
                    time_state_note='Unobserved initial working state; learned time module trained on separate synthetic transitions.')
    _, tensors = flatten(model, state)
    if not all(torch.equal(v, tensors[k]) for k, v in previous_tensors.items()):
        raise AssertionError('Existing tensors changed')
    with TensorStore.create(database, model, state, metadata) as store:
        restored, rs, _, rev = store.read()
        if fingerprint(model, state) != fingerprint(restored, rs):
            raise AssertionError('Database mismatch')
        store.checkpoint()
    save(portable, model, state, metadata)
    restored, rs, _ = load(portable)
    if fingerprint(model, state) != fingerprint(restored, rs):
        raise AssertionError('Portable mismatch')
    old_model, old_state, _ = load(parent)
    exact_history = history_equal(old_model, old_state, restored, rs)
    with torch.no_grad():
        answers = query(restored, rs, torch.tensor([12]*4), torch.tensor([0, 1, 1, 1]), torch.tensor([0, 0, 1, 2])).argmax(-1).tolist()
    if not exact_history or answers != [7, 5, 9, 3]:
        raise AssertionError('Personal history mismatch')
    protected_files()
    report = {'parent': str(parent.relative_to(ROOT)), 'parent_sha256': metadata['parent_sha256'],
              'selected_seed': 7, 'symbolic_observations': state.cursor, 'world_observations': 0, 'observed_pulses': 0,
              'existing_tensors_exact': True, 'history_logits_exact': exact_history, 'demo_answers': answers,
              'tensor_bytes_before': old_bytes,
              'tensor_bytes_after': sum(v.numel()*v.element_size() for v in tensors.values()),
              'parent_portable_bytes': parent.stat().st_size,
              'new_module_parameters': training['parameters'], 'revision': rev,
              'files': {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size,
                         'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in (database, portable)},
              'Chinese_training_started': False, 'passed': True}
    (ROOT/'runs/p11-candidate.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
