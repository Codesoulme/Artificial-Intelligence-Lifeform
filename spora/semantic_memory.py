"""Bounded attribute generalization from accepted neural facts.

Six key bits and four value bits are supplied structural priors, not discovered
semantics. Predictions are hypotheses; this module never writes observations.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


def attributes(keys, width=6):
    return ((keys[..., None] >> torch.arange(width, device=keys.device)) & 1).float()


class SemanticMemory(nn.Module):
    def __init__(self, confidence=.9):
        super().__init__()
        if not .9 <= confidence < 1:
            raise ValueError('Invalid semantic confidence')
        self.confidence = confidence
        self.head = nn.Linear(6, 4)
        self.register_buffer('trained_cursor', torch.tensor(-1, dtype=torch.long))
        self.register_buffer('evidence_count', torch.tensor(0, dtype=torch.long))

    def config_dict(self):
        return {'confidence': self.confidence}

    def forward(self, keys):
        if keys.dtype != torch.long or bool(((keys < 0) | (keys >= 64)).any()):
            raise ValueError('Six-bit symbol keys required')
        return self.head(attributes(keys)*2-1)


@torch.no_grad()
def accepted_facts(model, state):
    from .growing import query
    if model.config.key_count != 64 or model.config.unknown != 16 or state.internal is None:
        raise ValueError('Finite 64-key sourced memory required')
    keys = torch.arange(64, device=next(model.parameters()).device)
    zero = torch.zeros_like(keys)
    values = query(model, state, keys, zero, zero).argmax(-1)
    # Only evidence actually supplied through the controlled current writer.
    evidence = state.internal.evidence @ model.memories[-1].current_basis.T
    eligible = (values < 16) & (evidence[0] > .9) & (evidence[1] >= .8)
    return keys[eligible], values[eligible]


def propose_semantic(model, state, steps=800, seed=2510000):
    """Return a detached candidate. Caller publishes only after acceptance."""
    if type(steps) is not int or not 1 <= steps <= 800:
        raise ValueError('Bounded semantic training required')
    keys, values = accepted_facts(model, state)
    if keys.numel() < 16:
        return None, {'accepted': False, 'reason': 'insufficient_sourced_facts', 'facts': keys.numel()}
    # Re-read all current accepted facts, including corrections, from the fast
    # matrices. This replay is transient and is not an external answer store.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        config = model.semantic_memory.config_dict() if hasattr(model, 'semantic_memory') else {}
        candidate = SemanticMemory(**config).to(keys.device)
    if hasattr(model, 'semantic_memory'):
        candidate.load_state_dict(copy.deepcopy(model.semantic_memory.state_dict()))
    labels = attributes(values, 4)
    optimizer = torch.optim.Adam(candidate.parameters(), lr=.05)
    with torch.enable_grad():
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(candidate(keys), labels)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        probabilities = candidate(keys).sigmoid()
        correct = (probabilities >= .5) == labels.bool()
        confidence = torch.where(labels.bool(), probabilities, 1-probabilities)
        accepted = bool(correct.all()) and float(confidence.min()) >= candidate.confidence
        if accepted:
            candidate.trained_cursor.fill_(state.cursor)
            candidate.evidence_count.fill_(keys.numel())
    return candidate.eval(), {'accepted': accepted, 'facts': keys.numel(), 'steps': steps,
        'loss': float(loss.detach()), 'bit_accuracy': float(correct.float().mean()),
        'minimum_correct_bit_confidence': float(confidence.min()),
        'reason': 'accepted' if accepted else 'conflicting_or_unlearned_facts'}


@torch.no_grad()
def semantic_query(model, state, keys):
    from .growing import query
    if keys.ndim != 1:
        raise ValueError('Vector of keys required')
    zero = torch.zeros_like(keys)
    factual = query(model, state, keys, zero, zero).argmax(-1)
    values = factual.clone()
    inferred = torch.zeros_like(keys, dtype=torch.bool)
    module = getattr(model, 'semantic_memory', None)
    if module is not None and int(module.trained_cursor) == state.cursor:
        probabilities = module(keys).sigmoid()
        certain = torch.maximum(probabilities, 1-probabilities).amin(-1) >= module.confidence
        inferred = (factual == model.config.unknown) & certain
        decoded = ((probabilities >= .5).long() << torch.arange(4, device=keys.device)).sum(-1)
        values = torch.where(inferred, decoded, factual)
    # The explicit inference mask must accompany any answer; caller cannot
    # silently present inferred outputs as recorded experiences.
    return {'values': values, 'inferred': inferred, 'observed': factual != model.config.unknown}


def consolidate_semantic_store(store, model, state, revision, **training):
    """Atomic gated publication; rejected candidates leave the individual intact."""
    candidate, report = propose_semantic(model, state, **training)
    if not report['accepted']:
        return report
    descendant = copy.deepcopy(model)
    descendant.semantic_memory = candidate
    report['commit'] = store.commit(descendant, state, revision,
        dict(store.metadata or {}, last_semantic={'cursor': state.cursor, 'gate': dict(report)}))
    return report
