"""Strict byte-interface snapshots and resumable finite adapter updates."""
import os
import copy
import hashlib
import json
from pathlib import Path
import torch
from .language_bridge import causal_loss, PAD


def frozen_digest(model, state):
    from .state_codec import flatten
    schema, tensors = flatten(model, state)
    digest = hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(',', ':')).encode())
    for name, tensor in sorted(tensors.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def interface_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith('organism.')}


def load_interface(model, payload, parent_sha256):
    if payload.get('format') not in ('spora-byte-interface-v1', 'spora-byte-interface-v2', 'spora-byte-training-v2',
                                     'spora-codepoint-interface-v1', 'spora-codepoint-training-v1',
                                     'spora-hybrid-training-v1'):
        raise ValueError('Unsupported byte interface')
    hybrid_model = getattr(model, 'hybrid_vocab_sha256', None)
    hybrid_payload = payload.get('format') == 'spora-hybrid-training-v1'
    if hybrid_payload != (hybrid_model is not None):
        raise ValueError('Hybrid interface format mismatch')
    if hybrid_payload and payload.get('provenance', {}).get('vocab_sha256') != hybrid_model:
        raise ValueError('Hybrid vocabulary digest mismatch')
    if payload.get('organism_sha256') != parent_sha256:
        raise ValueError('Interface belongs to a different individual')
    expected_context_layers = len(getattr(model, 'contextualizer', ()))
    if payload.get('context_layers', 0) != expected_context_layers:
        raise ValueError('Interface context layer count does not match model')
    if payload.get('vocab_size', 259) != getattr(model, 'vocab_size', 259):
        raise ValueError('Interface vocabulary size does not match model')
    expected = interface_state(model)
    actual = payload.get('state', {})
    if set(actual) != set(expected):
        raise ValueError('Interface must contain exactly adapter tensors; organism replacement forbidden')
    for name, tensor in actual.items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype or not bool(torch.isfinite(tensor).all()):
            raise ValueError('Invalid interface tensor: '+name)
    merged = model.state_dict()
    merged.update(actual)
    model.load_state_dict(merged)


def atomic_save(payload, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix+'.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


class ByteTrainer:
    def __init__(self, model, state, parent_sha256, provenance, seed, format_name='spora-byte-training-v2'):
        if any(p.requires_grad for p in model.organism.parameters()):
            raise ValueError('The entire organism must stay frozen')
        self.model, self.state = model, state
        self.parent_sha256, self.provenance = parent_sha256, dict(provenance)
        self.format_name = format_name
        self.generator = torch.Generator().manual_seed(seed)
        self.optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=3e-4, weight_decay=.01)
        self.steps = self.tokens_seen = 0

    def step(self, examples, batch_size):
        if examples.ndim != 2 or examples.shape[0] == 0 or type(batch_size) is not int or not 1 <= batch_size <= 16:
            raise ValueError('Nonempty finite batch required')
        ids = torch.randint(len(examples), (batch_size,), generator=self.generator)
        return self.step_batch(examples[ids])

    def step_batch(self, batch):
        if batch.ndim != 2 or batch.dtype != torch.long or not 1 <= batch.shape[0] <= 16:
            raise ValueError('Finite batch of integer tokens required')
        batch = batch.to(next(self.model.parameters()).device)
        self.optimizer.zero_grad(set_to_none=True)
        loss = causal_loss(self.model(batch, self.state), batch)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite byte loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), 1., error_if_nonfinite=True)
        self.optimizer.step()
        self.steps += 1
        self.tokens_seen += int((batch[:, 1:] != PAD).sum())
        return float(loss.detach())

    def snapshot(self):
        on_cuda = next(self.model.parameters()).is_cuda
        return {'format': self.format_name, 'context_layers': len(getattr(self.model, 'contextualizer', ())),
            'vocab_size': getattr(self.model, 'vocab_size', 259),
            'organism_sha256': self.parent_sha256,
            'state': interface_state(self.model), 'provenance': self.provenance,
            'optimizer': copy.deepcopy(self.optimizer.state_dict()), 'steps': self.steps,
            'trained_tokens': self.tokens_seen, 'sampler_rng': self.generator.get_state(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if on_cuda else []}

    def restore(self, payload):
        if 'covering_sampler' in payload and not hasattr(self, 'sampler'):
            raise ValueError('Covering curriculum requires its own resumable trainer')
        if payload.get('format') != self.format_name or payload.get('provenance') != self.provenance:
            raise ValueError('Resume requires identical data, seed, batch, context and device')
        if type(payload.get('steps')) is not int or not 0 <= payload['steps'] <= 10000 or type(payload.get('trained_tokens')) is not int or payload['trained_tokens'] < 0:
            raise ValueError('Invalid training counters')
        load_interface(self.model, payload, self.parent_sha256)
        self.optimizer.load_state_dict(payload['optimizer'])
        self.generator.set_state(payload['sampler_rng'])
        torch.set_rng_state(payload['torch_rng'])
        if next(self.model.parameters()).is_cuda:
            torch.cuda.set_rng_state_all(payload['cuda_rng'])
        elif payload['cuda_rng']:
            raise ValueError('Cannot silently resume a CUDA run on CPU')
        self.steps, self.tokens_seen = payload['steps'], payload['trained_tokens']
