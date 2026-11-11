"""Finite learned write decisions and associative provenance, without event logs.

Source/quality are declared environment inputs. They are not inferred truth.
Hypotheses cannot become factual beliefs through this write path.
"""
from dataclasses import dataclass, replace
import math
import torch
from torch import nn
from torch.nn import functional as F

SOURCES = ('direct', 'report', 'inference', 'simulation')
ACTIONS = ('new', 'correct', 'support', 'repeat', 'defer')
PROVENANCE_WIDTH = 27


class WriteControl(nn.Module):
    def __init__(self, hidden=64, confidence=.8):
        super().__init__()
        if type(hidden) is not int or not 8 <= hidden <= 256 or not .5 <= confidence <= 1:
            raise ValueError('Invalid write control configuration')
        self.hidden, self.confidence = hidden, confidence
        self.network = nn.Sequential(nn.Linear(10, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 5))

    def config_dict(self):
        return {'hidden': self.hidden, 'confidence': self.confidence}

    def forward(self, features):
        if features.ndim != 2 or features.shape[-1] != 10 or not bool(torch.isfinite(features).all()):
            raise ValueError('Finite batch of write features required')
        return self.network(features)


class NoveltyWriteControl(nn.Module):
    """Versioned write controller with one additional novelty feature."""
    kind = 'novelty'

    def __init__(self, hidden=64, confidence=.8):
        super().__init__()
        if type(hidden) is not int or not 8 <= hidden <= 256 or not .5 <= confidence <= 1:
            raise ValueError('Invalid novelty controller configuration')
        self.hidden, self.confidence = hidden, confidence
        self.network = nn.Sequential(nn.Linear(11, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 5))

    @classmethod
    def from_legacy(cls, legacy):
        if not isinstance(legacy, WriteControl):
            raise ValueError('A legacy WriteControl is required')
        candidate = cls(legacy.hidden, legacy.confidence)
        with torch.no_grad():
            candidate.network[0].weight[:, :10].copy_(legacy.network[0].weight)
            candidate.network[0].weight[:, 10].zero_()
            candidate.network[0].bias.copy_(legacy.network[0].bias)
            candidate.network[2].load_state_dict(legacy.network[2].state_dict())
            candidate.network[4].load_state_dict(legacy.network[4].state_dict())
        return candidate

    def config_dict(self):
        return {'kind': self.kind, 'hidden': self.hidden, 'confidence': self.confidence}

    def forward(self, features):
        if features.ndim != 2 or features.shape[-1] != 11 or not bool(torch.isfinite(features).all()):
            raise ValueError('Finite batch of novelty write features required')
        return self.network(features)


@dataclass
class InternalState:
    evidence: torch.Tensor
    provenance: list
    start_cursor: int
    informative: int = 0
    corrections: int = 0
    router_factors: object = None
    router_cursor: int = -1
    encoding: int = 2
    event_keys: object = None

    def metadata(self):
        return {k: getattr(self, k) for k in ('start_cursor', 'informative', 'corrections', 'router_cursor', 'encoding')}

    def validate(self, model, state):
        width = model.config.memory_width
        if model.config.current_address != 'orthogonal' or model.config.history_writer != 'rls':
            raise ValueError('Internal controls require orthogonal current and RLS history')
        if any(type(x) is not int or x < 0 for x in (self.start_cursor, self.informative, self.corrections)) or self.start_cursor > state.cursor:
            raise ValueError('Invalid internal control counters')
        if type(self.encoding) is not int or self.encoding not in (1,2) or type(self.router_cursor) is not int or self.corrections>self.informative:
            raise ValueError('Invalid router revision or correction counter')
        if self.evidence.shape != (3, width) or len(self.provenance) != len(state.pages):
            raise ValueError('Invalid evidence extent')
        tensors = [self.evidence] + self.provenance
        if any(p.shape != (11 if self.encoding==1 else PROVENANCE_WIDTH, width) for p in self.provenance):
            raise ValueError('Invalid provenance extent')
        if self.event_keys is not None:
            if (type(state.cursor) is not int or not 0 <= state.cursor <= len(state.pages) * model.config.time_count
                    or not isinstance(self.event_keys, list) or len(self.event_keys) != len(state.pages)
                    or any(not isinstance(page, torch.Tensor) or page.shape != (model.config.time_count,)
                           or page.dtype != torch.int64 or page.device != self.evidence.device for page in self.event_keys)):
                raise ValueError('Invalid event-key index extent')
            keys = torch.cat(self.event_keys)
            if (bool(((keys[:state.cursor] < 0) | (keys[:state.cursor] >= model.config.key_count)).any())
                    or bool((keys[state.cursor:] != -1).any())):
                raise ValueError('Incomplete or invalid event-key index')
        if self.router_factors is None:
            if self.router_cursor != -1:
                raise ValueError('Missing router factors')
        else:
            if self.router_cursor != state.cursor or len(self.router_factors) != len(model.memories):
                raise ValueError('Stale router factors')
            for sites in self.router_factors:
                if len(sites) != len(state.pages)+1:
                    raise ValueError('Invalid router page count')
                for left, right in sites:
                    if left.ndim != 2 or right.shape != (left.shape[1], width) or left.shape[0] != width or not 1 <= left.shape[1] <= width:
                        raise ValueError('Invalid router factor shape')
                    tensors.extend([left, right])
        if not all(t.dtype == torch.float32 and bool(torch.isfinite(t).all()) for t in tensors):
            raise ValueError('Invalid internal tensor precision or value')

    def invalidate_router(self, **changes):
        return replace(self, router_factors=None, router_cursor=-1, **changes)


def attach(model, state, controller):
    if state.internal is not None or hasattr(model, 'write_control'):
        raise ValueError('Internal control already attached')
    if model.config.current_address != 'orthogonal' or model.config.history_writer != 'rls':
        raise ValueError('Compatible frozen memory required')
    device = next(model.parameters()).device
    internal = InternalState(torch.zeros(3, model.config.memory_width, device=device),
                             [torch.zeros(PROVENANCE_WIDTH, model.config.memory_width, device=device) for _ in state.pages], state.cursor)
    model.write_control = controller.to(device).eval()
    return replace(state, internal=internal)


def enable_event_key_index(model, state):
    """Opt into exact event-key metadata on a fresh controlled individual.

    This is an auxiliary index, not learned neural memory. Existing histories
    cannot be backfilled safely, and callers must persist state transactionally.
    """
    if state.internal is None or state.cursor != 0 or state.internal.event_keys is not None:
        raise ValueError('Fresh controlled state without an event-key index required')
    state.internal.validate(model, state)
    pages = [torch.full((model.config.time_count,), -1, dtype=torch.int64, device=state.internal.evidence.device)
             for _ in state.pages]
    return replace(state, internal=state.internal.invalidate_router(event_keys=pages))


@torch.no_grad()
def write_features(model, state, key, value, source, quality, valid_at):
    from .growing import query
    device = next(model.parameters()).device
    key_tensor = torch.tensor([key], device=device)
    zeros = torch.zeros_like(key_tensor)
    old_value = int(query(model, state, key_tensor, zeros, zeros).argmax(-1))
    address = model.memories[-1].current_basis[key]
    evidence = state.internal.evidence @ address
    has_evidence = float(evidence[0]) > .5
    known = old_value != model.config.unknown
    old_quality = float(evidence[1].clamp(0, 1)) if has_evidence else (1. if known else 0.)
    old_time = round(float(evidence[2])*1024) if has_evidence else -1
    freshness = (valid_at > old_time)-(valid_at < old_time)
    feature = F.one_hot(torch.tensor(source, device=device), 4).float().tolist()
    feature += [quality, float(known), float(known and old_value == value), old_quality, float(freshness), float(has_evidence)]
    return torch.tensor([feature], dtype=torch.float32, device=device), address, old_value, old_quality, old_time


@torch.no_grad()
def observe_sourced(model, state, key, value, source, quality, valid_at=None):
    from .growing import _observe_base, capacity
    if state.internal is None or not hasattr(model, 'write_control'):
        raise ValueError('Individual has no internal write controls')
    if state.internal.encoding!=2:
        raise ValueError('Legacy continuous provenance is read-only; create a new audited descendant')
    if type(key) is not int or not 0 <= key < model.config.key_count or type(value) is not int or not 0 <= value < model.config.unknown:
        raise ValueError('Invalid finite-symbol observation')
    if source not in SOURCES or isinstance(quality, bool) or not isinstance(quality, (int, float)) or not math.isfinite(quality) or not 0 <= quality <= 1:
        raise ValueError('Explicit source and quality in [0,1] required')
    valid_at = state.cursor if valid_at is None else valid_at
    if type(valid_at) is not int or not 0 <= valid_at <= state.cursor or state.cursor >= capacity(model, state):
        raise ValueError('Invalid effective time or explicit growth required')
    si = SOURCES.index(source)
    features, address, old_value, old_quality, old_time = write_features(model, state, key, value, si, quality, valid_at)
    from .observation_surprise import observation_surprise
    surprise = observation_surprise(known=old_value != model.config.unknown, old_value=old_value,
                                    new_value=value, old_quality=old_quality, new_quality=quality,
                                    old_time=old_time, new_time=valid_at)
    control_features = (torch.cat([features, features.new_tensor([[surprise['score']]])], dim=1)
                        if isinstance(model.write_control, NoveltyWriteControl) else features)
    logits = model.write_control(control_features)
    if logits.shape != (1, len(ACTIONS)) or not bool(torch.isfinite(logits).all()):
        raise ValueError('Write controller must return finite action logits')
    probabilities = logits.softmax(-1)[0]
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError('Write controller returned invalid probabilities')
    action = int(probabilities.argmax())
    confidence = float(probabilities[action])
    reason = 'learned_decision'
    if si >= 2:
        action, reason = 4, 'nonfactual_source_guard'
    elif confidence < model.write_control.confidence:
        action, reason = 4, 'uncertain_decision'
    # Do not allow inconsistent neural action labels to manufacture a new belief.
    known = old_value != model.config.unknown
    if (action == 0 and known) or (action in (1, 2, 3) and not known) or (action in (2, 3) and old_value != value):
        action, reason = 4, 'inconsistent_decision'
    internal = state.internal
    page = state.cursor//model.config.time_count
    local = state.cursor % model.config.time_count
    keys = torch.tensor([key], device=features.device)
    modes, times = torch.ones_like(keys), torch.tensor([local], device=features.device)
    _, bound = model.representation(keys, modes, times)
    q = model.memory_address(model.memories[-1], bound, keys, modes)[0]
    precision = state.pages[page][-1][1][0]
    projected = precision @ q
    gain = projected/(1+(q*projected).sum())
    quality_code=round(quality*255)
    bits=[(quality_code>>i)&1 for i in range(8)]+[(valid_at>>i)&1 for i in range(10)]
    target = torch.cat([F.one_hot(torch.tensor(si, device=q.device), 4),
                        F.one_hot(torch.tensor(action, device=q.device), 5), q.new_tensor(bits)]).float()
    provenance = list(internal.provenance)
    provenance[page] = (provenance[page]+(target-provenance[page]@q)[:, None]*gain[None]).detach()
    evidence = internal.evidence
    if action in (0, 1, 2):
        evidence_target = q.new_tensor([1., max(quality, old_quality) if action == 2 else quality, valid_at/1024])
        evidence = (evidence+(evidence_target-evidence@address)[:, None]*address[None]).detach()
    updated = _observe_base(model, state, key, value, int(action in (0, 1)))
    event_keys = internal.event_keys
    if event_keys is not None:
        event_keys = list(event_keys)
        event_keys[page] = event_keys[page].clone()
        event_keys[page][local] = key
    updated.internal = internal.invalidate_router(evidence=evidence, provenance=provenance, event_keys=event_keys,
                          informative=internal.informative+int(action in (0, 1, 2)),
                          corrections=internal.corrections+int(action == 1))
    return updated, {'action': ACTIONS[action], 'confidence': confidence, 'reason': reason,
                     'source': source, 'history_time': state.cursor, 'current_write': action in (0, 1),
                     'novelty': surprise['score'], 'novelty_components': surprise['components']}


@torch.no_grad()
def read_provenance(model, state, key, event_time):
    from .growing import query
    if state.internal is None or type(key) is not int or not 0 <= key < model.config.key_count or type(event_time) is not int or event_time < 0:
        raise ValueError('Internal state and valid query required')
    if event_time < state.internal.start_cursor or event_time >= state.cursor:
        return {'source': None, 'reason': 'unrecorded_provenance'}
    device = next(model.parameters()).device
    k, m, t = [torch.tensor([v], device=device) for v in (key, 1, event_time)]
    if int(query(model, state, k, m, t).argmax(-1)) == model.config.unknown:
        return {'source': None, 'reason': 'unknown_event'}
    _, bound = model.representation(k, m, t % model.config.time_count)
    q = model.memory_address(model.memories[-1], bound, k, m)[0]
    result = state.internal.provenance[event_time//model.config.time_count] @ q
    return _decode_provenance(result, event_time, state.internal.encoding)


def _decode_provenance(result, event_time, encoding):
    src, act = result[:4], result[4:9]
    if float(src.max()) < .7 or float(act.max()) < .7:
        return {'source': None, 'reason': 'uncertain_provenance'}
    labels={'source': SOURCES[int(src.argmax())], 'action': ACTIONS[int(act.argmax())]}
    if encoding==1:
        return dict(labels,quality_estimate=float(result[9].clamp(0,1)),valid_at_estimate=float(result[10])*1024,
                    note='Legacy continuous regression; not exact timestamps')
    bits=result[9:]/src.sum().clamp_min(.1)
    if bool(((bits>.2)&(bits<.8)).any()):
        return dict(labels,quality=None,valid_at=None,reason='uncertain_numeric_provenance')
    code=(bits>.5).tolist()
    quality_code=sum(int(b)<<i for i,b in enumerate(code[:8]))
    valid_at=sum(int(b)<<i for i,b in enumerate(code[8:]))
    if valid_at>event_time:
        return dict(labels,quality=None,valid_at=None,reason='invalid_numeric_provenance')
    return dict(labels,quality=quality_code/255,valid_at=valid_at,quality_resolution=1/255)


@torch.no_grad()
def read_provenance_batch(model, state, pairs, batch_size=16):
    """Read bounded batches of (key, event time); results retain input order.

    This evaluates the same neural evidence as read_provenance. It neither
    certifies absence nor resolves event-time or natural-language semantics.
    """
    from .growing import query
    if state.internal is None or not isinstance(pairs, (list, tuple)) or type(batch_size) is not int or not 1 <= batch_size <= 64:
        raise ValueError('Internal state, query pairs and bounded batch size required')
    for pair in pairs:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2 or type(pair[0]) is not int
                or not 0 <= pair[0] < model.config.key_count or type(pair[1]) is not int or pair[1] < 0):
            raise ValueError('Valid integer key and event time required')
    results = [None] * len(pairs)
    recorded = []
    for index, (key, event_time) in enumerate(pairs):
        if event_time < state.internal.start_cursor or event_time >= state.cursor:
            results[index] = {'source': None, 'reason': 'unrecorded_provenance'}
        else:
            recorded.append((index, key, event_time))
    device = next(model.parameters()).device
    for start in range(0, len(recorded), batch_size):
        chunk = recorded[start:start+batch_size]
        keys = torch.tensor([x[1] for x in chunk], dtype=torch.long, device=device)
        times = torch.tensor([x[2] for x in chunk], dtype=torch.long, device=device)
        modes = torch.ones_like(keys)
        known = query(model, state, keys, modes, times).argmax(-1) != model.config.unknown
        _, bound = model.representation(keys, modes, times % model.config.time_count)
        addresses = model.memory_address(model.memories[-1], bound, keys, modes)
        matrices = torch.stack([state.internal.provenance[x[2]//model.config.time_count] for x in chunk])
        decoded = torch.bmm(matrices, addresses.unsqueeze(-1)).squeeze(-1)
        for i, (index, _, event_time) in enumerate(chunk):
            if not bool(known[i]):
                results[index] = {'source': None, 'reason': 'unknown_event'}
            else:
                results[index] = _decode_provenance(decoded[i], event_time, state.internal.encoding)
    return results


def consolidation_request(model, state, min_information=16, max_pages=1):
    """Bounded scheduling decision only; publication still requires P10 gates."""
    if type(min_information) is not int or min_information < 1 or max_pages != 1:
        raise ValueError('Positive information threshold and one-page budget required')
    internal = state.internal
    if internal is None or internal.informative < min_information:
        return {'pages': [], 'reason': 'insufficient_new_information'}
    eligible = [i for i, p in enumerate(state.pages) if isinstance(p, list) and (i+1)*model.config.time_count <= state.cursor]
    return {'pages': eligible[:1], 'reason': 'new_information_and_full_history' if eligible else 'no_complete_active_page',
            'informative': internal.informative, 'corrections': internal.corrections}
