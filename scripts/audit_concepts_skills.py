"""Fixed finite concept and feedback-learning experiments, no text training."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spora.checkpoint import load_checkpoint
from spora.growing import new_state, query, grow, save, load
from spora.internal_memory import attach, observe_sourced, read_provenance
from spora.semantic_memory import attributes, propose_semantic, semantic_query
from spora.skills import FeedbackPolicy, learn_feedback
from audit_internal_memory import controller
from audit_durable import fingerprint


@torch.no_grad()
def concept_world(seed, world, audit_seed, output):
    model, _, _, _ = load_checkpoint(ROOT/f'checkpoints/p5b-orthogonal90-seed{seed}.pt')
    control, _ = controller(seed)
    state = attach(model, new_state(model), control)
    generator = torch.Generator().manual_seed(audit_seed+world)
    keys = torch.arange(64)
    bits = attributes(keys).long()
    axes = torch.randperm(6, generator=generator)[:4]
    flips = torch.randint(2, (4,), generator=generator)
    values = ((bits[:, axes] ^ flips) << torch.arange(4)).sum(-1)
    train = keys[bits.sum(-1) % 2 == 0]
    held = keys[bits.sum(-1) % 2 != 0]
    for key in train.tolist():
        state, result = observe_sourced(model, state, key, int(values[key]), 'direct', .95)
        assert result['current_write']
    baseline = fingerprint(model, state)
    candidate, gate = propose_semantic(model, state, seed=seed)
    assert fingerprint(model, state) == baseline
    model.semantic_memory = candidate
    prediction = semantic_query(model, state, held)
    accuracy = float((prediction['values'] == values[held]).float().mean())
    inference_rate = float(prediction['inferred'].float().mean())
    zero = torch.zeros_like(held)
    episodic_unknown = bool((query(model, state, held, zero, zero).argmax(-1) == 16).all())
    torch.manual_seed(seed+100)
    from spora.semantic_memory import SemanticMemory
    ablated = copy.deepcopy(model)
    ablated.semantic_memory = SemanticMemory()
    ablation_unknown = bool((semantic_query(ablated, state, held)['values'] == 16).all())
    # Untrusted counterexamples must not become consolidation targets.
    for key in held[:4].tolist():
        state, _ = observe_sourced(model, state, key, int(values[key]) ^ 15, 'simulation', 1.)
    stale_unknown = bool((semantic_query(model, state, held)['values'] == 16).all())
    next_candidate, next_gate = propose_semantic(model, state, seed=seed+1)
    model.semantic_memory = next_candidate
    after_hypotheses = semantic_query(model, state, held)
    # A factual exception overrides the general rule and makes it stale.
    k = int(train[0]); old = int(values[k]); t = state.cursor
    state, correction = observe_sourced(model, state, k, old ^ 15, 'direct', 1.)
    result = semantic_query(model, state, torch.tensor([k]))
    corrected = int(result['values'][0]) == (old ^ 15) and bool(result['observed'][0])
    prior_semantic = {k: v.clone() for k, v in model.semantic_memory.state_dict().items()}
    _, single_exception_gate = propose_semantic(model, state, seed=seed+2)
    # A single corner exception IS linearly separable on the parity subset.
    # Use the XOR of independent attributes: this cannot be represented by a
    # linear bit head, so the all-fact acceptance gate must reject it.
    for key in train.tolist():
        if state.cursor == len(state.pages)*64:
            state = grow(model, state)
        contradictory = (int(values[key]) & 14) | int(bits[key, 0] ^ bits[key, 1])
        state, _ = observe_sourced(model, state, key, contradictory, 'direct', 1.)
    rejected, conflict_gate = propose_semantic(model, state, seed=seed+3)
    rollback_exact = all(torch.equal(v, model.semantic_memory.state_dict()[k]) for k, v in prior_semantic.items())
    history = query(model, state, train, torch.ones_like(train), torch.arange(32)).argmax(-1)
    sources = [read_provenance(model, state, int(k), i)['source'] for i, k in enumerate(train)]
    path = output/f'concept-seed{seed}-world{world}.pt'
    save(path, model, state)
    rm, rs, _ = load(path)
    row = {'seed': seed, 'world': world, 'rule_axes': axes.tolist(), 'rule_complements': flips.tolist(),
        'gate': gate, 'held_out_combinations': 32, 'held_out_accuracy': accuracy,
        'inferred_fraction': inference_rate, 'episodic_unknown_unchanged': episodic_unknown,
        'ablation_unknown': ablation_unknown, 'stale_abstains': stale_unknown,
        'nonfactual_exclusion_gate': next_gate,
        'after_nonfactual_accuracy': float((after_hypotheses['values'] == values[held]).float().mean()),
        'correction_observed_priority': corrected, 'single_exception_gate': single_exception_gate,
        'conflict_gate': conflict_gate,
        'rejected_candidate_preserves_previous': rollback_exact,
        'history_accuracy': float((history == values[train]).float().mean()),
        'sources_exact': all(s == 'direct' for s in sources),
        'restart_exact': fingerprint(model, state) == fingerprint(rm, rs)}
    row['passed'] = gate['accepted'] and next_gate['accepted'] and not conflict_gate['accepted'] and all(
        row[k] for k in ('episodic_unknown_unchanged', 'ablation_unknown', 'stale_abstains',
        'correction_observed_priority', 'rejected_candidate_preserves_previous', 'sources_exact', 'restart_exact')) and min(
        accuracy, row['after_nonfactual_accuracy'], row['history_accuracy']) >= .95
    return row


MOVES = torch.tensor([[0., 0.], [.2, 0.], [-.2, 0.], [0., .2], [0., -.2]])


def environment(position, goal, actions):
    outcome = position + MOVES[actions]
    feedback = -(goal-outcome).square().sum(-1) - .01*(actions != 0)
    return outcome, feedback


def sample(generator, count, training):
    pool = torch.rand(count*4, 2, generator=generator)*2-1
    corner = (pool.abs().amin(-1) > .6) & (pool.prod(-1) > 0)
    selected = pool[~corner if training else corner]
    if selected.shape[0] < count:
        return torch.cat([selected, sample(generator, count-selected.shape[0], training)])
    return selected[:count]


@torch.no_grad()
def rollout(policy, delta, goals, mode, generator):
    position = goals-delta
    success = (goals-position).norm(dim=-1) <= .16
    first_success = torch.where(success, 0, 13)
    for step in range(12):
        if mode == 'oracle':
            cost = (goals[:, None]-position[:, None]-MOVES).square().sum(-1)+.01*(torch.arange(5) != 0)
            actions = cost.argmin(-1)
        elif mode == 'random':
            actions = torch.randint(5, (len(delta),), generator=generator)
        else:
            # Diagnostic baselines can wander outside the learned domain; stop
            # those episodes, rather than silently extrapolating the policy.
            relative = goals-position
            outside = relative.abs().amax(-1) > policy.radius
            actions = policy(relative.clamp(-policy.radius, policy.radius)).argmax(-1)
            actions[outside] = 0
        actions[success] = 0
        position, _ = environment(position, goals, actions)
        reached = (goals-position).norm(dim=-1) <= .16
        first_success[(~success) & reached] = step+1
        success |= reached
    return {'success_rate': float(success.float().mean()), 'mean_distance': float((goals-position).norm(dim=-1).mean()),
        'mean_steps_capped13': float(first_success.float().mean())}


def skill_world(seed, audit_seed, output):
    torch.manual_seed(seed)
    policy = FeedbackPolicy()
    untrained = copy.deepcopy(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=.02)
    generator = torch.Generator().manual_seed(2510000+seed)
    started = time.perf_counter()
    for step in range(1500):
        delta = sample(generator, 128, True)
        goal = torch.rand(128, 2, generator=generator)*4-2
        position = goal-delta
        action = torch.randint(5, (128,), generator=generator)
        outcome, feedback = environment(position, goal, action)
        loss = learn_feedback(policy, optimizer, position, goal, action, outcome, feedback)
    audit = torch.Generator().manual_seed(audit_seed+seed)
    delta = sample(audit, 512, False)
    goals = torch.rand(512, 2, generator=audit)*20-10
    controls = {mode: rollout(untrained if mode == 'untrained' else policy, delta, goals, mode, audit)
        for mode in ('learned', 'untrained', 'random', 'oracle')}
    with torch.no_grad():
        a = policy.act(goals-delta, goals)
        b = policy.act(goals-delta+50, goals+50)
        translation = float((a == b).float().mean())
    row = {'seed': seed, 'updates': 1500, 'feedback_presentations': 1500*128, 'last_loss': loss,
        'training_seconds': time.perf_counter()-started, 'held_out_episodes': 512, 'controls': controls,
        'translated_action_agreement': translation,
        'parameters': sum(p.numel() for p in policy.parameters())}
    row['passed'] = controls['learned']['success_rate'] >= .95 and translation >= .99 and controls['learned']['success_rate'] > controls['untrained']['success_rate']+.5
    torch.save({'config': policy.config_dict(), 'state': policy.state_dict(), 'training': row}, output/f'skill-seed{seed}.pt')
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--audit-seed', type=int, default=2520000)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    report = {'concepts': [], 'skills': [], 'audit_seed': args.audit_seed}
    for seed in (7, 11, 23):
        for world in range(2):
            row = concept_world(seed, world, args.audit_seed, output)
            report['concepts'].append(row)
            print(json.dumps(row), flush=True)
        row = skill_world(seed, args.audit_seed, output)
        report['skills'].append(row)
        print(json.dumps(row), flush=True)
    report['passed'] = all(r['passed'] for r in report['concepts']+report['skills'])
    (output/'audit.json').write_text(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
