"""Measure bounded counterfactual rollouts against the synthetic P11 environment."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from audit_time_world import ROOT, load_module
from train_time_world import environment_step
from spora.online_time import OnlineSession, OnlineState


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def audit(module, seed):
    generator = torch.Generator().manual_seed(seed)
    starts = (2 * torch.rand(512, 3, generator=generator) - 1) * torch.tensor([2.8, 1.8, 1.8])
    parameters = .15 + .65 * torch.rand(512, 2, generator=generator)
    plans = 2 * torch.rand(512, 5, 4, 2, generator=generator) - 1
    goals = (2 * torch.rand(512, 2, generator=generator) - 1) * torch.tensor([2.8, 1.8])
    durations = torch.full((4,), .25)
    valid_total = 0
    stopped = 0
    errors = torch.zeros(3)
    persistence_errors = torch.zeros(3)
    decision_matches = 0
    comparable = 0
    goal_matches = 0
    goal_comparable = 0
    unsupported_choices = 0
    no_supported_plan = 0
    goal_regret = 0.
    session = OnlineSession(module, OnlineState(), lambda: 0, lambda: 0)
    for start, environment, actions, goal in zip(starts, parameters, plans, goals, strict=True):
        predicted, in_domain = module.imagine(start[None], actions, environment[None], durations)
        current = start.expand(5, -1)
        simulated = []
        for step, action in enumerate(actions.unbind(1)):
            current = environment_step(current, action, environment.expand(5, -1), durations[step].expand(5))
            simulated.append(current)
        reference = torch.stack(simulated, dim=1)
        reference_domain = (reference.abs() <= start.new_tensor([3., 2., 2.])).all(dim=2).cumprod(1).bool()
        shared = in_domain & reference_domain
        stopped += int((~in_domain[:, -1]).sum())
        valid_total += int(shared.sum())
        errors += (predicted[shared] - reference[shared]).square().sum(0)
        persistence_errors += (start - reference[shared]).square().sum(0)
        if int(shared[:, -1].sum()) >= 2:
            available = shared[:, -1]
            decision_matches += int(predicted[available, -1, 0].argmax() == reference[available, -1, 0].argmax())
            comparable += 1
        session.observe_world(start, actions[0, 0], environment, 'synthetic')
        simulation = dict(kind='simulation', observation=session.state.world_count, states=predicted.tolist(),
                          in_domain=in_domain.tolist())
        suggested = session.choose_imagined_plan(simulation, goal)
        if suggested['kind'] == 'no_supported_plan':
            no_supported_plan += 1
            continue
        choice = suggested['plan'] - 1
        if not reference_domain[choice, -1]:
            unsupported_choices += 1
            continue
        error = ((reference[:, -1, (0, 2)] - goal) / goal.new_tensor([3., 2.])).norm(dim=1)
        best = int(error.masked_fill(~reference_domain[:, -1], float('inf')).argmin())
        goal_matches += int(choice == best)
        goal_comparable += 1
        goal_regret += float(error[choice] - error[best])
    clock = [0]
    session = OnlineSession(module, OnlineState(), lambda: clock[0], lambda: 0)
    actions = torch.zeros(1, 1, 2)
    before = session.imagine(actions, [.25])['kind']
    session.observe_world([0., .4, -.2], [.5, .2], [.4, .3], 'synthetic')
    observed = session.state.metadata()
    tensors = {key: value.clone() for key, value in session.state.tensors.items()}
    after = session.imagine(actions, [.25])['kind']
    state_preserved = observed == session.state.metadata() and all(torch.equal(session.state.tensors[key], tensor)
                                                                  for key, tensor in tensors.items())
    clock[0] = int((module.config.max_horizon + .1) * 1e9)
    stale = session.imagine(actions, [.25])['kind']
    restarted = OnlineSession(module, session.state, lambda: clock[0], lambda: 0)
    result = dict(contexts=len(starts), plans=len(starts)*5, stopped_plans=stopped, comparable_steps=valid_total,
                  rmse=(errors/valid_total).sqrt().tolist(), persistence_rmse=(persistence_errors/valid_total).sqrt().tolist(),
                  plan_choice_matches=decision_matches, comparable_plan_choices=comparable, state_preserved=state_preserved,
                  goal_choice_matches=goal_matches, comparable_goals=goal_comparable,
                  unsupported_choices=unsupported_choices, no_supported_plan=no_supported_plan,
                  mean_goal_regret=goal_regret / max(1, goal_comparable),
                  before=before, after=after, stale=stale, restarted=restarted.imagine(actions, [.25])['kind'])
    if not state_preserved or (before, after, stale, result['restarted']) != (
            'needs_observation', 'simulation', 'needs_observation', 'needs_observation'):
        raise AssertionError(result)
    return result


def main(output):
    torch.set_num_threads(2)
    started = time.perf_counter()
    results = []
    for seed in (7, 11, 23):
        path = ROOT / f'runs/p11-seed{seed}/model.pt'
        source = digest(path)
        module, _ = load_module(path)
        result = dict(seed=seed, checkpoint_sha256=source, **audit(module, 326041))
        if digest(path) != source:
            raise ValueError('World model checkpoint changed')
        results.append(result)
        print(json.dumps(result), flush=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(dict(results=results, seconds=time.perf_counter()-started), stream, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    main(args.output)
