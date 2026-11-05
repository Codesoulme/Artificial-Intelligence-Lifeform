"""Three fresh foundation streams; repeated sleep, corrections and cold restore."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from spora.checkpoint import load_checkpoint
from spora.growing import new_state, grow, observe, query, save, load, scalars, GrowingState
from spora.sleep_memory import propose, behavior_gate, SealedPage
from spora.durable import TensorStore
from spora.state_codec import flatten
from audit_growth import isolated


@torch.no_grad()
def metrics(model, state, keys, values, trusts):
    device = next(model.parameters()).device
    k = torch.tensor(keys, device=device); v = torch.tensor(values, device=device)
    t = torch.arange(len(keys), device=device); m = torch.ones_like(k)
    predictions = []
    for start in range(0, len(keys), 128):
        predictions.append(query(model, state, k[start:start+128], m[start:start+128], t[start:start+128]).argmax(-1))
    p = torch.cat(predictions)
    u = torch.tensor(trusts, device=device) == 0
    invalid = query(model, state, (k+19)%64, m, t).argmax(-1)
    current = {}
    for key, value, trusted in zip(keys, values, trusts):
        if trusted:
            current[key] = value
    ck = torch.arange(64, device=device); zeros = torch.zeros_like(ck)
    actual = query(model, state, ck, zeros, zeros).argmax(-1)
    expected = torch.tensor([current.get(i, 16) for i in range(64)], device=device)
    unknown = expected == 16
    return {'history': float((p==v).float().mean()), 'oldest64': float((p[:64]==v[:64]).float().mean()),
            'untrusted_history': float((p[u]==v[u]).float().mean()) if u.any() else None,
            'invalid_unknown': float((invalid==16).float().mean()),
            'current': float((actual==expected).float().mean()),
            'current_unknown': float((actual[unknown]==16).float().mean()) if unknown.any() else None}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    args = parser.parse_args(); output = Path(args.output)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True); torch.set_num_threads(2)
    selection_path = ROOT/'configs/p10-selection.json'
    selection = json.loads(selection_path.read_text())
    plan = json.loads((ROOT/'configs/p9-p10-plan.json').read_text())['sleep']
    result = {'selection': selection, 'selection_sha256': hashlib.sha256(selection_path.read_bytes()).hexdigest(),
              'audit_seed': plan['audit_seed'], 'chains': [], 'language_training_steps': 0,
              'limitations': ['teacher probes cover finite address domain; not unseen-fact generalization',
                             'dense replay is step-matched, MAC counts differ',
                             'SVD is a strong non-learning compression control, no superiority assumed',
                             'current bank remains active; sleep seals history only',
                             'disk file can retain free pages; VACUUM is separate maintenance']}
    for seed in plan['foundation_seeds']:
        for world in range(plan['worlds_per_seed']):
            folder = output/f'seed{seed}-world{world}'; folder.mkdir()
            model, _, _, _ = load_checkpoint(ROOT/f'checkpoints/p5b-orthogonal90-seed{seed}.pt', 'cuda')
            state = grow(model, new_state(model), 3)
            backbone_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            gen = torch.Generator().manual_seed(plan['audit_seed']+world)
            keys, values, trusts, stages = [], [], [], []
            store = TensorStore.create(folder/'individual.sqlite', model, state,
                                       {'stage': 'P10 independent audit', 'seed': seed, 'world': world})
            try:
                for stage in range(plan['stages']):
                    corrections = []
                    for event in range(64):
                        key = int(torch.randint(64, (1,), generator=gen))
                        value = int(torch.randint(16, (1,), generator=gen))
                        trusted = int(torch.rand(1, generator=gen) > .25)
                        if stage and event < 4:
                            key = keys[event]; value = (values[event]+stage+5)%16; trusted = 1
                        state = observe(model, state, key, value, trusted)
                        keys.append(key); values.append(value); trusts.append(trusted)
                        store.commit(model, state, store.revision)
                        if stage and event < 4:
                            with torch.no_grad():
                                answer = query(model, state, torch.tensor([key],device='cuda'),
                                               torch.zeros(1,dtype=torch.long,device='cuda'),
                                               torch.zeros(1,dtype=torch.long,device='cuda')).argmax(-1)
                            corrections.append(int(answer)==value)
                    before = metrics(model, state, keys, values, trusts)
                    dense_path = folder/f'before-stage{stage}.pt'; save(dense_path, model, state)
                    controls = {}
                    if stage == 0:
                        for method in ('svd', 'dense-control'):
                            control, cost = propose(model, state, stage, rank=selection['rank'],
                                steps=selection['steps'], seed=plan['audit_seed']+world, method=method)
                            controls[method] = {'cost': cost, 'metrics': metrics(model, control, keys, values, trusts)}
                        initial = new_state(model)
                        pages = list(state.pages); pages[stage] = initial.pages[0]
                        reset = GrowingState(state.current, pages, state.cursor)
                        controls['no-transfer-reset'] = {'metrics': metrics(model, reset, keys, values, trusts)}
                    candidate, training = propose(model, state, stage, rank=selection['rank'],
                        steps=selection['steps'], seed=plan['audit_seed']+world+stage*10)
                    gate = behavior_gate(model, state, candidate, stage)
                    after = metrics(model, candidate, keys, values, trusts)
                    passed = (gate['passed'] and training['state_reduction'] >= plan['minimum_state_reduction']
                              and all(x is None or x >= plan['threshold'] for x in after.values())
                              and all(corrections))
                    row = {'stage': stage, 'before': before, 'after': after, 'training': training,
                           'gate': gate, 'corrections': corrections, 'controls': controls,
                           'state_before': scalars(state), 'state_after': scalars(candidate), 'passed': passed}
                    if not passed:
                        stages.append(row); break
                    stats = store.commit(model, candidate, store.revision,
                        {'stage': 'P10 slept', 'seed': seed, 'world': world, 'sealed_pages': stage+1})
                    state = candidate
                    compact_path = folder/f'after-stage{stage}.pt'; save(compact_path, model, state)
                    row.update(commit=stats, before_file_bytes=dense_path.stat().st_size,
                               after_file_bytes=compact_path.stat().st_size)
                    # Restart both serialization paths before the next stream segment.
                    model, state, _, _ = store.read('cuda')
                    row['restored_metrics'] = metrics(model, state, keys, values, trusts)
                    cpu_model, cpu_state, _ = load(compact_path)
                    ck = keys[:4]+keys[-4:]; times = list(range(4))+list(range(len(keys)-4,len(keys)))
                    cold = isolated(compact_path, ck, [1]*8, times)
                    with torch.no_grad():
                        expected = query(cpu_model,cpu_state,torch.tensor(ck),torch.ones(8,dtype=torch.long),torch.tensor(times)).softmax(-1).tolist()
                    row['cold_exact'] = cold['probabilities'] == expected
                    row['passed'] = row['passed'] and row['cold_exact'] and all(
                        x is None or x >= plan['threshold'] for x in row['restored_metrics'].values())
                    stages.append(row)
                    print(json.dumps({'seed':seed,'world':world,'stage':stage,'after':after,
                                      'state_reduction':training['state_reduction'],'passed':row['passed']}),flush=True)
                    if not row['passed']: break
                unchanged = all(torch.equal(value.detach().cpu(), backbone_before[name])
                                for name, value in model.state_dict().items())
                chain = {'seed':seed,'world':world,'stages':stages,'backbone_exact':unchanged,
                         'passed':len(stages)==plan['stages'] and all(r['passed'] for r in stages) and unchanged}
                result['chains'].append(chain)
                (output/'audit.json').write_text(json.dumps(result,indent=2))
            finally:
                store.close()
    result['passed'] = all(x['passed'] for x in result['chains'])
    result['selection_unchanged'] = hashlib.sha256(selection_path.read_bytes()).hexdigest()==result['selection_sha256']
    (output/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'passed':result['passed'],'chains':len(result['chains'])}),flush=True)
    if not result['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
