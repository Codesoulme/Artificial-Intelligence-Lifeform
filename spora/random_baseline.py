"""Untrained causal control and paired initialization for P91.

This module does not provide training, checkpoint restoration, data auditing or
an experiment runner. Pairing only establishes equal initial backbone tensors;
it does not establish matched batches, costs, or experimental results.
"""
from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .conversation import ConversationBlock, ConversationBridge, ConversationConfig, RMSNorm


class _CausalPast(list):
    """Incremental attention values bound to the creating model and its weights."""
    def __init__(self, model):
        super().__init__([None] * model.config.depth)
        self.owner = model
        self.sources = tuple((p, p._version) for p in model.parameters())

    def validate(self, model):
        params = tuple(model.parameters())
        if (self.owner is not model or len(self) != model.config.depth
                or len(params) != len(self.sources)
                or any(p is not old or p._version != version
                       for p, (old, version) in zip(params, self.sources))):
            raise ValueError('Weights changed; rebuild the complete conversation prefix')


class StandardCausalModel(nn.Module):
    """ConversationBridge backbone with tied embeddings and no memory input.

    ``state`` is accepted for call compatibility and is never inspected. Past
    values are transient attention values, not organism state. As with the
    original binding, direct .data writes and foreign writes are unsupported.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = deepcopy(config) if config is not None else ConversationConfig()
        self.config.validate()
        c = self.config
        self.embedding = nn.Embedding(c.vocabulary, c.width)
        self.blocks = nn.ModuleList([ConversationBlock(c) for _ in range(c.depth)])
        self.norm = RMSNorm(c.width)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=.02)
        for block in self.blocks:
            nn.init.normal_(block.out.weight, std=.02 / math.sqrt(2 * c.depth))
            nn.init.normal_(block.down.weight, std=.02 / math.sqrt(2 * c.depth))

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, tokens, state=None, past=None, use_past=False):
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError('Integer tokens expected')
        if past is not None:
            if not isinstance(past, _CausalPast):
                raise ValueError('Unbound incremental state; rebuild the prefix')
            past.validate(self)
        binding = _CausalPast(self) if use_past else None
        offset = 0 if past is None else past[0][0].shape[-2]
        if not 0 < tokens.shape[1] + offset <= self.config.context:
            raise ValueError('Conversation exceeds context')
        x = self.embedding(tokens)
        updated = []
        for i, block in enumerate(self.blocks):
            x, p = block(x, None if past is None else past[i], use_past)
            if use_past:
                updated.append(p)
        logits = F.linear(self.norm(x), self.embedding.weight)
        if past is not None:
            past.validate(self)
        if use_past:
            binding.validate(self)
            binding[:] = updated
            return logits, binding
        return logits


def make_paired_models(organism, config=None, *, seed):
    """Construct two fresh CPU models and copy every common backbone tensor.

    The organism is passed to ConversationBridge, which freezes its parameters;
    callers must provide an appropriately protected individual. No language
    checkpoint is accepted. CPU RNG state is restored on return; data sampling
    should nevertheless use an independent generator. Move models after pairing.
    """
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('seed must be an integer in [0, 2**64)')
    config = deepcopy(config) if config is not None else ConversationConfig()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        bridge = ConversationBridge(organism, deepcopy(config))
        standard = StandardCausalModel(config)
    source = bridge.state_dict()
    target = standard.state_dict()
    backbone = {name: value for name, value in source.items()
                if name.startswith(('embedding.', 'blocks.', 'norm.'))}
    if backbone.keys() != target.keys():
        raise RuntimeError('Backbone tensor names differ')
    standard.load_state_dict(backbone, strict=True)
    if any(not torch.equal(value, standard.state_dict()[name])
           for name, value in backbone.items()):
        raise RuntimeError('Backbone copy failed')
    return bridge, standard
