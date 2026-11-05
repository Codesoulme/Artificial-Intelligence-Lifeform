"""Finite sleep consolidation into trained low-rank, read-only history weights.

The source dense history is absent from an accepted state. A packed symmetric
novelty matrix remains necessary and is charged. No raw events are persisted.
"""
from dataclasses import dataclass
import time
import torch


@dataclass
class SealedPage:
    factors: list
    rank: int

    def validate(self, model):
        width = model.config.memory_width
        if type(self.rank) is not int or not 1 <= self.rank <= width or len(self.factors) != len(model.memories):
            raise ValueError('Invalid sealed page rank/layers')
        for left, right, upper in self.factors:
            if left.shape != (width, self.rank) or right.shape != (self.rank, width) or upper.shape != (width*(width+1)//2,):
                raise ValueError('Invalid sealed page factors')

    def materialize(self, site):
        left, right, upper = self.factors[site]
        width = left.shape[0]
        indices = torch.triu_indices(width, width, device=left.device)
        precision = upper.new_zeros((width, width))
        precision[indices[0], indices[1]] = upper
        precision[indices[1], indices[0]] = upper
        return [(left @ right).unsqueeze(0), precision.unsqueeze(0)]

    def scalar_count(self):
        return sum(x.numel() for site in self.factors for x in site)


def history_site(page, site):
    return page[site] if isinstance(page, list) else page.materialize(site)


def train_page(model, page, rank=16, steps=600, seed=2210000, method='distill'):
    if isinstance(page, SealedPage) or model.config.history_writer != 'rls':
        raise ValueError('Sleep requires an active RLS history page')
    if method not in ('distill', 'svd', 'dense-control') or type(steps) is not int or not 1 <= steps <= 600:
        raise ValueError('Invalid bounded consolidation request')
    width = model.config.memory_width
    if type(rank) is not int or not 1 <= rank <= width:
        raise ValueError('Invalid bottleneck rank')
    device = next(model.parameters()).device
    started = time.perf_counter()
    keys = torch.arange(model.config.key_count, device=device).repeat_interleave(model.config.time_count)
    times = torch.arange(model.config.time_count, device=device).repeat(model.config.key_count)
    modes = torch.ones_like(keys)
    with torch.no_grad():
        _, bound = model.representation(keys, modes, times)
        queries = [model.memory_address(site, bound, keys, modes).detach() for site in model.memories]
        targets = [q @ tensors[0][0].detach().T for q, tensors in zip(queries, page)]
    generator = torch.Generator(device=device).manual_seed(seed)
    parameters, initial_losses, tensors = [], [], []
    for params, target, q in zip(page, targets, queries):
        if method == 'svd':
            u, s, vh = torch.linalg.svd(params[0][0].detach(), full_matrices=False)
            left = u[:, :rank]*s[:rank].sqrt()
            right = s[:rank].sqrt()[:, None]*vh[:rank]
        elif method == 'dense-control':
            left = params[0][0].detach().clone().requires_grad_(True)
            right = None
            parameters.append(left)
        else:
            left = (torch.randn(width, rank, generator=generator, device=device)*.03).requires_grad_(True)
            right = (torch.randn(rank, width, generator=generator, device=device)*.03).requires_grad_(True)
            parameters.extend([left, right])
        tensors.append((left, right))
        prediction = q @ left.T if right is None else (q @ right.T) @ left.T
        initial_losses.append(float((prediction-target).square().mean().detach()))
    completed_steps = 0
    if method != 'svd':
        optimizer = torch.optim.Adam(parameters, lr=.03 if method == 'distill' else .001)
        with torch.enable_grad():
            for _ in range(steps):
                losses = []
                for (left, right), q, target in zip(tensors, queries, targets):
                    prediction = q @ left.T if right is None else (q @ right.T) @ left.T
                    losses.append((prediction-target).square().mean() / target.square().mean().clamp_min(1e-6))
                loss = sum(losses)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 10., error_if_nonfinite=True)
                optimizer.step()
                completed_steps += 1
    final_losses, factors, dense = [], [], []
    with torch.no_grad():
        indices = torch.triu_indices(width, width, device=device)
        for (left, right), q, target, params in zip(tensors, queries, targets, page):
            matrix = left if right is None else left @ right
            final_losses.append(float((q @ matrix.T-target).square().mean()))
            precision = params[1][0]
            if not torch.equal(precision, precision.T):
                raise ValueError('Precision must be symmetric before sealing')
            if right is not None:
                factors.append([left.detach().clone(), right.detach().clone(),
                                precision[indices[0], indices[1]].detach().clone()])
            dense.append([matrix.detach()[None].clone(), params[1].detach().clone()])
    if device.type == 'cuda':
        torch.cuda.synchronize()
    student = dense if method == 'dense-control' else SealedPage(factors, rank)
    before = sum(x.numel() for site in page for x in site)
    after = before if isinstance(student, list) else student.scalar_count()
    # Matrix multiply-add counts only; excludes Adam, activations and packing.
    per_step_macs = sum(q.shape[0]*(width*width if method == 'dense-control' else 2*width*rank)
                        for q in queries)
    report = {'method': method, 'rank': rank, 'steps': completed_steps,
              'teacher_probe_queries_per_site': keys.numel(), 'initial_mse': initial_losses,
              'final_mse': final_losses, 'source_scalars': before, 'student_scalars': after,
              'state_reduction': 1-after/before, 'seconds': time.perf_counter()-started,
              'student_forward_macs_per_step': per_step_macs,
              'student_forward_macs_total': per_step_macs*completed_steps,
              'limitation': 'Behavioral probe distillation, not unseen-fact generalization; dense control is step-matched, not FLOP-matched'}
    return student, report


def propose(model, state, page_index, **training):
    from .growing import GrowingState
    if type(page_index) is not int or not 0 <= page_index < len(state.pages):
        raise ValueError('Invalid page index')
    if (page_index+1)*model.config.time_count > state.cursor:
        raise ValueError('Cannot consolidate an unfinished history page')
    student, report = train_page(model, state.pages[page_index], **training)
    pages = list(state.pages); pages[page_index] = student
    internal=state.internal.invalidate_router() if state.internal is not None else None
    return GrowingState(state.current, pages, state.cursor, state.runtime,internal), report


@torch.no_grad()
def behavior_gate(model, before, after, page_index, minimum=.95, batch_size=128):
    from .growing import query
    if not .95 <= minimum <= 1:
        raise ValueError('Behavior threshold cannot be below .95')
    device = next(model.parameters()).device
    keys = torch.arange(model.config.key_count, device=device).repeat_interleave(model.config.time_count)
    times = torch.arange(model.config.time_count, device=device).repeat(model.config.key_count)+page_index*model.config.time_count
    modes = torch.ones_like(keys)
    expected, actual, maximum = [], [], 0.
    for start in range(0, keys.numel(), batch_size):
        sl = slice(start, start+batch_size)
        old = query(model, before, keys[sl], modes[sl], times[sl])
        new = query(model, after, keys[sl], modes[sl], times[sl])
        expected.append(old.argmax(-1)); actual.append(new.argmax(-1))
        maximum = max(maximum, float((old.softmax(-1)-new.softmax(-1)).abs().max()))
    expected, actual = torch.cat(expected), torch.cat(actual)
    present = expected != model.config.unknown
    groups = {}
    for name, mask in (('all', torch.ones_like(present)), ('teacher_known', present), ('teacher_unknown', ~present)):
        groups[name] = {'count': int(mask.sum()), 'agreement': float((expected[mask]==actual[mask]).float().mean()) if mask.any() else None}
    ck = torch.arange(model.config.key_count, device=device); zeros = torch.zeros_like(ck)
    current_exact = torch.equal(query(model, before, ck, zeros, zeros), query(model, after, ck, zeros, zeros))
    passed = current_exact and all(x['agreement'] is None or x['agreement'] >= minimum for x in groups.values())
    return {'groups': groups, 'current_logits_exact': current_exact, 'max_probability_difference': maximum, 'passed': passed}


def consolidate_store(store, model, state, revision, page_index, rank=16, steps=600, seed=2210000):
    candidate, training = propose(model, state, page_index, rank=rank, steps=steps, seed=seed)
    gate = behavior_gate(model, state, candidate, page_index)
    accepted = gate['passed'] and training['state_reduction'] >= .2
    result = {'accepted': accepted, 'training': training, 'gate': gate, 'parent_revision': revision}
    if accepted:
        if candidate.internal is not None:
            candidate.internal=candidate.internal.invalidate_router(informative=0,corrections=0)
        metadata = dict(store.metadata or {})
        metadata['last_sleep'] = {'page': page_index, 'rank': rank, 'steps': steps,
                                  'parent_revision': revision, 'gate': gate}
        result['commit'] = store.commit(model, candidate, revision, metadata)
    return result
