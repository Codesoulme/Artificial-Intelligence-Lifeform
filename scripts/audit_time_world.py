"""Independent finite dynamics audits; no tuning on these examples."""
import argparse
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spora.world_time import TimeWorld, WorldTimeConfig
from spora.online_time import OnlineSession, OnlineState
from train_time_world import sample, environment_step


def load_module(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data['format'] != 'spora-time-world-v1':
        raise ValueError('Unknown time module')
    model = TimeWorld(WorldTimeConfig(**data['config']))
    model.load_state_dict(data['state'])
    return model.eval(), data['training']


@torch.no_grad()
def audit(module, plan):
    gen = torch.Generator().manual_seed(plan['audit_seed'])
    state, action, env = sample(gen, 4096)
    dt = torch.rand(4096, generator=gen)*1.98+.02
    expected = environment_step(state, action, env, dt)
    prediction = module.predict(state, action, env, dt)
    rmse = lambda x: (x-expected).square().mean(0).sqrt().tolist()
    score = rmse(prediction)
    fixed = module.predict(state, action, env, .25)
    ratio = float((fixed-expected).square().mean().sqrt()/(prediction-expected).square().mean().sqrt())
    horizon_scores = {}
    for horizon in (.1, .5, 1., 2.):
        target = environment_step(state, action, env, state.new_full((len(state),), horizon))
        pred = module.predict(state, action, env, horizon)
        horizon_scores[str(horizon)] = (pred-target).square().mean(0).sqrt().tolist()
    period = torch.rand(4096, generator=gen)*1.8+.2
    observed = period[:, None]*(1+(torch.rand(4096, 2, generator=gen)*2-1)*.02)
    error = float(((module.period(observed)-period).abs()/period).mean())
    mean_error = float(((observed.mean(-1)-period).abs()/period).mean())
    zero = torch.zeros(512, 3)
    parameters = env[:512]
    plus = module.predict(zero, torch.ones(512, 2)*.8, parameters, 1.)
    minus = module.predict(zero, -torch.ones(512, 2)*.8, parameters, 1.)
    direction = float((plus > minus).all(-1).float().mean())
    clock = [0]
    run = OnlineSession(module, OnlineState(), lambda: clock[0], lambda: 10**12+clock[0])
    run.observe_world([0., .4, -.2], [.5, .2], [.4, .3], 'synthetic')
    clock[0] = 10**9
    a = torch.tensor(run.tick()['world_prediction'])
    clock[0] = 0
    run = OnlineSession(module, OnlineState(), lambda: clock[0], lambda: 10**12+clock[0])
    run.observe_world([0., .4, -.2], [.5, .2], [.4, .3], 'synthetic')
    for _ in range(100):
        clock[0] += 10**7
        b = torch.tensor(run.tick()['world_prediction'])
    partition_error = float((a-b).abs().max())
    long_horizon_rejected = False
    try:
        module.predict(state, action, env, 3.)
    except ValueError:
        long_horizon_rejected = True
    threshold = plan['thresholds']
    passed = (max(score) <= threshold['world_rmse_each_component']
              and max(max(x) for x in horizon_scores.values()) <= threshold['world_rmse_each_component']
              and error <= threshold['period_relative_mae']
              and ratio >= threshold['time_ablation_error_ratio_min']
              and partition_error <= threshold['partition_prediction_max_abs']
              and direction == 1 and long_horizon_rejected)
    return {'cases': 4096, 'rmse': score, 'horizon_rmse': horizon_scores,
            'persistence_rmse': rmse(state), 'fixed_250ms_rmse': rmse(fixed),
            'zero_action_rmse': rmse(module.predict(state, torch.zeros_like(action), env, dt)),
            'time_ablation_error_ratio': ratio, 'period_relative_mae': error,
            'interval_mean_baseline_relative_mae': mean_error,
            'action_direction_agreement': direction, 'partition_max_abs': partition_error,
            'long_horizon_rejected': long_horizon_rejected, 'passed': passed}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    path = Path(a.output)
    if path.exists():
        raise FileExistsError(path)
    path.mkdir(parents=True)
    torch.set_num_threads(2)
    plan = json.loads((ROOT/'configs/p11-time-plan.json').read_text())
    report = {'plan': plan, 'seeds': [], 'selected_seed_fixed_before_audit': 7}
    for seed in plan['seeds']:
        module, training = load_module(ROOT/f'runs/p11-seed{seed}/model.pt')
        row = {'seed': seed, 'training': training, 'audit': audit(module, plan)}
        report['seeds'].append(row)
        print(json.dumps(row), flush=True)
    report['passed'] = all(x['audit']['passed'] for x in report['seeds'])
    (path/'audit.json').write_text(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
