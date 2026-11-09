"""Aligned-label hybrid training primitive; source admission is external."""
import random

import torch
import torch.nn.functional as F

from .language_bridge import PAD
from .language_training import ByteTrainer
from .packed_coverage import PackedCoverage


class PackedHybridTrainer(ByteTrainer):
    def __init__(self, model, state, parent_sha256, provenance, seed):
        required = ('tokens_sha256', 'index_sha256', 'vocab_sha256', 'verification_sha256',
                    'admission_sha256', 'batch', 'length', 'device')
        if not isinstance(provenance, dict) or any(not provenance.get(key) for key in required):
            raise ValueError('Packed input, admission and schedule provenance required')
        if model.hybrid_vocab_sha256 != provenance['vocab_sha256']:
            raise ValueError('Hybrid vocabulary provenance mismatch')
        super().__init__(model, state, parent_sha256, provenance, seed,
                         format_name='spora-hybrid-training-v1')
        self.packed_rng = random.Random(seed)
        self.coverage = None
        self.pending_coverage = None

    def step(self, stream):
        if self.coverage is None:
            if self.pending_coverage is None and self.tokens_seen:
                raise ValueError('Cannot start coverage after untracked updates')
            self.coverage = (PackedCoverage.restore(stream, self.pending_coverage)
                             if self.pending_coverage is not None else PackedCoverage(stream))
            self.pending_coverage = None
        if self.coverage.stream is not stream:
            raise ValueError('Cannot change packed stream during training')
        inputs, labels, spans = stream.batch(self.packed_rng, self.provenance['batch'],
                                             self.provenance['length'], next(self.model.parameters()).device,
                                             return_spans=True)
        loss = self.step_aligned(inputs, labels)
        self.coverage.add(spans)
        return loss

    def step_aligned(self, inputs, labels):
        if (inputs.ndim != 2 or labels.shape != inputs.shape or inputs.dtype != torch.long
                or labels.dtype != torch.long or not 1 <= inputs.shape[0] <= 16 or inputs.shape[1] < 1):
            raise ValueError('Expected aligned nonempty integer inputs and labels')
        device = next(self.model.parameters()).device
        inputs, labels = inputs.to(device), labels.to(device)
        valid = labels != -100
        if (not bool(valid.any()) or bool(((inputs < 0) | (inputs >= self.model.vocab_size)).any())
                or bool(((labels[valid] < 0) | (labels[valid] >= self.model.vocab_size)).any())
                or bool((labels[valid] == PAD).any()) or bool((inputs[valid] == PAD).any())):
            raise ValueError('Invalid or empty supervised targets')
        if inputs.shape[1] > 1:
            adjacent = valid[:, :-1] & valid[:, 1:]
            if bool((labels[:, :-1][adjacent] != inputs[:, 1:][adjacent]).any()):
                raise ValueError('Inputs and labels are not next-token aligned')
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(inputs, self.state)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite packed loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), 1., error_if_nonfinite=True)
        self.optimizer.step()
        self.steps += 1
        self.tokens_seen += int(valid.sum())
        return float(loss.detach())

    def snapshot(self):
        if self.coverage is not None and self.coverage.presentations != self.tokens_seen:
            raise ValueError('Cannot snapshot mixed tracked and untracked updates')
        payload = super().snapshot()
        payload['packed_sampler_rng'] = self.packed_rng.getstate()
        if self.coverage is not None:
            payload['packed_coverage'] = self.coverage.snapshot()
        elif self.pending_coverage is not None:
            payload['packed_coverage'] = self.pending_coverage
        return payload

    def restore(self, payload):
        if 'packed_sampler_rng' not in payload:
            raise ValueError('Packed sampler state required')
        coverage = payload.get('packed_coverage')
        if coverage is not None and (not isinstance(coverage, dict)
                                     or coverage.get('presentations') != payload.get('trained_tokens')):
            raise ValueError('Coverage and training counters disagree')
        candidate = random.Random()
        candidate.setstate(payload['packed_sampler_rng'])
        super().restore(payload)
        self.packed_rng = candidate
        self.coverage = None
        self.pending_coverage = coverage
