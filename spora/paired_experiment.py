"""Paired random-model training primitives. No data approval is implied."""
import copy
from dataclasses import asdict
import hashlib
import io
import math
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .document_stream import trim_ignored_tail
from .document_stream_observation import batch_with_windows
from .language_training import atomic_save, frozen_digest, interface_state
from .random_baseline import make_paired_models
from .token_objective import backward_token_budget


def language_mode(model, training):
    """Change the language path without changing any organism module's mode."""
    organism = getattr(model, 'organism', None)
    modes = [] if organism is None else [(module, module.training) for module in organism.modules()]
    try:
        model.train(training)
    finally:
        for module, mode in modes:
            module.training = mode


class RecordingRandom(random.Random):
    def __init__(self, seed):
        super().__init__(seed)
        self.draws = []

    def randrange(self, *args):
        value = super().randrange(*args)
        self.draws.append(value)
        return value


def _digest_batches(batches):
    h = hashlib.sha256()
    for x, y in batches:
        h.update(str(tuple(x.shape)).encode())
        h.update(x.contiguous().numpy().tobytes())
        h.update(y.contiguous().numpy().tobytes())
    return h.hexdigest()


def sample_update(documents, rng, batch_size, length, budget):
    """Use DocumentStream itself; retain sampled document and target intervals."""
    batches, spans, total = [], [], 0
    while total < budget:
        x, y, windows = batch_with_windows(documents, rng, batch_size, length, 'cpu')
        x, y = trim_ignored_tail(x, y)
        for window, row in zip(windows, y):
            count = window.end - window.start
            if int((row != -100).sum()) != count:
                raise ValueError('Supervision changed after tail trimming')
            spans.append((window.document, window.start, window.end))
        total += int((y != -100).sum())
        batches.append((x, y))
    return batches, spans, total


def coverage(spans):
    groups = {}
    for doc, start, end in spans:
        groups.setdefault(doc, []).append((start, end))
    unique = 0
    for intervals in groups.values():
        intervals.sort()
        end = -1
        for a, b in intervals:
            unique += max(0, b - max(a, end))
            end = max(end, b)
    total = sum(b-a for _, a, b in spans)
    return {'documents': len(groups), 'unique_targets': unique,
            'presentations': total, 'repeat_presentations': total-unique}


@torch.no_grad()
def evaluate(model, state, documents, length):
    """Fixed, disjoint, document-contained windows; token weighted NLL."""
    device = next(model.parameters()).device
    language_mode(model, False)
    total, count, per_document = 0., 0, []
    for start, end in zip(documents.starts, documents.ends):
        start, end = int(start), int(end)
        doc_sum, doc_count = 0., 0
        for offset in range(start, end-1, length):
            seq = torch.as_tensor(np.array(documents.stream[offset:min(end, offset+length+1)], dtype=np.int64), device=device)
            x, y = seq[:-1].unsqueeze(0), seq[1:].unsqueeze(0)
            loss = F.cross_entropy(model(x, state).flatten(0, 1), y.flatten(), reduction='sum')
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite evaluation')
            doc_sum += float(loss); doc_count += y.numel()
        per_document.append({'nll': doc_sum/doc_count, 'targets': doc_count})
        total += doc_sum; count += doc_count
    if not count:
        raise ValueError('Empty development set')
    return {'nll': total/count, 'targets': count, 'documents': per_document}


class PairedTrainer:
    """Joint checkpoint keeps both arms and their sampler in one transaction.

    Callers are responsible for validating data provenance and file protection.
    """
    def __init__(self, organism, state, config, documents, *, seed, data_seed,
                 provenance, batch_size=4, length=512, budget=8192,
                 schedule_steps=100, warmup=14, lr=1e-4, device='cpu'):
        if not 1 <= schedule_steps <= 100 or not 0 < warmup < schedule_steps:
            raise ValueError('Schedule must be within the 100-update bound')
        if not 0 < length <= config.context or min(batch_size, budget) <= 0:
            raise ValueError('Invalid sampling configuration')
        self.organism, self.state, self.documents = organism, state, documents
        self.provenance = copy.deepcopy(provenance)
        self.spec = dict(seed=seed, data_seed=data_seed, batch_size=batch_size,
                         length=length, budget=budget, schedule_steps=schedule_steps,
                         warmup=warmup, lr=lr, device=str(device), torch_version=str(torch.__version__),
                         model_config=asdict(config))
        self.models = list(make_paired_models(organism, config, seed=seed))
        self.models = [m.to(device) for m in self.models]
        self.optimizers = [torch.optim.AdamW(m.trainable_parameters(), lr=lr, betas=(.9,.95), weight_decay=.01) for m in self.models]
        self.rng = RecordingRandom(data_seed)
        self.steps, self.targets, self.spans, self.trace, self.logs, self.evaluations = 0, 0, [], [], [], []
        self.before = frozen_digest(organism, state)
        self.organism_modes = tuple((name, module.training) for name, module in organism.named_modules())
        self._dirty = False
        self.guard_seconds = 0.0

    def check_frozen(self):
        if any(p.requires_grad or p.grad is not None for p in self.organism.parameters()) or frozen_digest(self.organism, self.state) != self.before:
            raise ValueError('Protected individual changed')
        if tuple((name, module.training) for name, module in self.organism.named_modules()) != self.organism_modes:
            raise ValueError('Protected individual module mode changed')

    def step(self):
        if self._dirty:
            raise ValueError('Interrupted update; restore the last checkpoint')
        if self.steps >= self.spec['schedule_steps']:
            raise ValueError('Schedule exhausted')
        self.check_frozen()
        self._dirty = True
        s = self.spec
        batches, spans, n = sample_update(self.documents, self.rng, s['batch_size'], s['length'], s['budget'])
        step = self.steps + 1
        fraction = step/s['warmup'] if step <= s['warmup'] else .1+.9*.5*(1+math.cos(math.pi*(step-s['warmup'])/(s['schedule_steps']-s['warmup'])))
        losses, seconds = [], []
        for model, opt in zip(self.models, self.optimizers):
            started = time.perf_counter()
            language_mode(model, True)
            opt.param_groups[0]['lr'] = s['lr'] * fraction
            opt.zero_grad(set_to_none=True)
            loss, actual = backward_token_budget(model, self.state, batches)
            if actual != n:
                raise ValueError('Target count mismatch')
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1., error_if_nonfinite=True)
            opt.step()
            if any(not bool(torch.isfinite(p).all()) for p in model.trainable_parameters()):
                raise ValueError('Nonfinite model after update')
            if next(model.parameters()).is_cuda:
                torch.cuda.synchronize()
            losses.append(float(loss)); seconds.append(time.perf_counter()-started)
        self.steps = step; self.targets += n; self.spans.extend(spans)
        self.trace.append({'sha256': _digest_batches(batches), 'targets': n, 'spans': spans})
        self.logs.append({'step': step, 'loss': losses, 'seconds': seconds, 'lr': s['lr']*fraction})
        self.check_frozen()
        self._dirty = False
        return self.logs[-1]

    def evaluate(self, development):
        self.check_frozen()
        values = [evaluate(m, self.state, development, self.spec['length']) for m in self.models]
        self.check_frozen()
        return {'step': self.steps, 'arms': values, 'difference': values[0]['nll']-values[1]['nll']}

    def save(self, path):
        if self._dirty:
            raise ValueError('Cannot save an incomplete paired update')
        self.check_frozen()
        payload = dict(format='spora-paired-v1', spec=self.spec, provenance=self.provenance,
                       frozen=self.before, models=[interface_state(m) for m in self.models],
                       optimizers=[o.state_dict() for o in self.optimizers], steps=self.steps,
                       targets=self.targets, spans=self.spans, trace=self.trace, logs=self.logs, evaluations=self.evaluations,
                       guard_seconds=self.guard_seconds,
                       sampler_rng=self.rng.getstate(), torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if next(self.models[0].parameters()).is_cuda else [])
        buffer = io.BytesIO(); torch.save(payload, buffer); data = buffer.getvalue()
        atomic_save({'sha256': hashlib.sha256(data).hexdigest(), 'payload': data}, path)

    def restore(self, path):
        envelope = torch.load(path, map_location='cpu', weights_only=True)
        data = envelope['payload']
        if hashlib.sha256(data).hexdigest() != envelope['sha256']:
            raise ValueError('Checkpoint integrity failure')
        p = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
        if p['format'] != 'spora-paired-v1' or p['spec'] != self.spec or p['provenance'] != self.provenance or p['frozen'] != self.before:
            raise ValueError('Checkpoint provenance mismatch')
        if type(p['steps']) is not int or not 0 <= p['steps'] <= self.spec['schedule_steps'] or len(p['trace']) != p['steps'] or len(p['logs']) != p['steps'] or p['targets'] != sum(row['targets'] for row in p['trace']):
            raise ValueError('Invalid checkpoint counters')
        if p['spans'] != [span for row in p['trace'] for span in row['spans']] or p['targets'] != sum(end-start for _, start, end in p['spans']):
            raise ValueError('Invalid checkpoint coverage')
        for model, values in zip(self.models, p['models'], strict=True):
            expected = interface_state(model)
            if values.keys() != expected.keys():
                raise ValueError('Invalid model tensors')
            for name, value in values.items():
                if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape or value.dtype != expected[name].dtype or not bool(torch.isfinite(value).all()):
                    raise ValueError('Invalid model tensor: '+name)
            full = model.state_dict(); full.update(values); model.load_state_dict(full)
        for optimizer, value in zip(self.optimizers, p['optimizers'], strict=True):
            optimizer.load_state_dict(value)
            for state in optimizer.state.values():
                for tensor in state.values():
                    if isinstance(tensor, torch.Tensor) and not bool(torch.isfinite(tensor).all()):
                        raise ValueError('Nonfinite optimizer state')
        self.rng.setstate(p['sampler_rng']); torch.set_rng_state(p['torch_rng'])
        if p['cuda_rng']:
            if not next(self.models[0].parameters()).is_cuda:
                raise ValueError('Device changed')
            torch.cuda.set_rng_state_all(p['cuda_rng'])
        self.steps, self.targets, self.spans, self.trace, self.logs = (p[k] for k in ('steps','targets','spans','trace','logs'))
        self.evaluations = p['evaluations']
        self.guard_seconds = p['guard_seconds']
        self.check_frozen()
