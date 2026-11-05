"""Causal backbone with differentiable, persistent fast neural weights."""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 512
    width: int = 128
    layers: int = 4
    heads: int = 4
    memory_width: int = 64
    expert_rank: int = 16
    key_count: int = 64
    value_count: int = 16
    key_start: int = 16
    value_start: int = 128
    max_length: int = 256
    memory_layers: tuple = (1, 3)
    memory_kind: str = 'mlp'

    def validate(self):
        if self.memory_kind not in ('mlp','delta'):
            raise ValueError('Unsupported neural memory kind')
        if self.width % self.heads or self.layers < 1:
            raise ValueError('Invalid backbone dimensions')
        if any(x < 0 or x >= self.layers for x in self.memory_layers):
            raise ValueError('Memory attachment outside backbone')
        if self.key_start + self.key_count > self.value_start:
            raise ValueError('Key and value token ranges overlap')
        if self.value_start + self.value_count > self.vocab_size:
            raise ValueError('Vocabulary too small')


class CausalBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.width
        self.heads = config.heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3*d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, d//self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(d//self.heads)
        mask = torch.ones(n, n, device=x.device, dtype=torch.bool).tril()
        weights = scores.masked_fill(~mask, float('-inf')).softmax(-1)
        x = x + self.out((weights @ v).transpose(1, 2).reshape(b, n, d))
        return x + self.ff(self.norm2(x))


def fast_forward(parameters, keys):
    w1, b1, w2, b2 = parameters
    hidden = F.gelu(torch.bmm(w1, keys.unsqueeze(-1)).squeeze(-1) + b1)
    return torch.bmm(w2, hidden.unsqueeze(-1)).squeeze(-1) + b2


class MemorySite(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, m, r = config.width, config.memory_width, config.expert_rank
        self.key = nn.Linear(d, m)
        self.key_norm = nn.LayerNorm(m)
        self.value = nn.Linear(d, m)
        self.value_decoder = nn.Linear(m, config.value_count)
        self.initial = nn.ParameterList([
            nn.Parameter(torch.randn(m, m) / math.sqrt(m)),
            nn.Parameter(torch.zeros(m)),
            nn.Parameter(torch.randn(m, m) * .02),
            nn.Parameter(torch.zeros(m)),
        ])
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(m, r), nn.GELU(), nn.Linear(r, m))
            for _ in range(2)
        ])
        self.router = nn.Linear(d, 2)
        self.read_gate = nn.Linear(d, 1)
        self.read_out = nn.Linear(m, d)
        self.log_lr = nn.Parameter(torch.tensor(-3.0))
        self.momentum = .5
        self.decay = 1e-4
        self.clip_norm = 10.0

    def address(self, key_embedding):
        return self.key_norm(self.key(key_embedding))

    def target(self, value_embedding):
        return F.normalize(self.value(value_embedding), dim=-1) * math.sqrt(self.initial[0].shape[0])

    def read(self, x, query, fast):
        recent = self.retrieve(fast, query)
        slow = torch.stack([expert(query) for expert in self.experts], dim=1)
        routing = self.router(x[:, -1]).softmax(-1).unsqueeze(-1)
        signal = recent + (routing * slow).sum(1)
        residual = torch.sigmoid(self.read_gate(x[:, -1])) * self.read_out(signal)
        return torch.cat([x[:, :-1], (x[:, -1]+residual).unsqueeze(1)], dim=1)

    def retrieve(self, fast, query):
        return fast_forward(fast, query)

    def write(self, fast, velocity, key, target, gate, differentiable):
        # Each row is an independent world; sum retains batch-independent step size.
        loss = .5 * (fast_forward(fast, key)-target.detach()).square().sum()
        grads = torch.autograd.grad(loss, fast, create_graph=differentiable)
        norm = torch.sqrt(sum(g.flatten(1).square().sum(1) for g in grads)+1e-12)
        factor = (self.clip_norm/norm).clamp(max=1.)
        rate = F.softplus(self.log_lr).clamp(max=.5)
        updated, moments = [], []
        for p, u, g, init in zip(fast, velocity, grads, self.initial):
            shape = (-1,) + (1,)*(p.ndim-1)
            mask = gate.reshape(shape)
            candidate_u = self.momentum*u + g*factor.reshape(shape)
            # gate=0 is a true no-write operation, including decay and momentum.
            new_u = mask*candidate_u+(1-mask)*u
            new_p = p-rate*mask*(candidate_u+self.decay*(p-init.unsqueeze(0)))
            if not differentiable:
                new_p = new_p.detach().requires_grad_(True)
                new_u = new_u.detach()
            updated.append(new_p)
            moments.append(new_u)
        return updated, moments


class DeltaMemorySite(MemorySite):
    """A normalized-key delta reference, not a Gated DeltaNet replication."""
    def __init__(self, config):
        super().__init__(config)
        m=config.memory_width
        self.initial=nn.ParameterList([nn.Parameter(torch.randn(m,m)*.02)])
        with torch.no_grad(): self.log_lr.fill_(2.1972246)

    def address(self, key_embedding):
        return F.normalize(super().address(key_embedding),dim=-1)

    def retrieve(self, fast, query):
        return torch.bmm(fast[0],query.unsqueeze(-1)).squeeze(-1)

    def write(self, fast, velocity, key, target, gate, differentiable):
        residual=target.detach()-self.retrieve(fast,key)
        delta=residual.unsqueeze(-1)*key.unsqueeze(1)
        updated=fast[0]+gate[:,None,None]*torch.sigmoid(self.log_lr)*delta
        if not differentiable: updated=updated.detach().requires_grad_(True)
        return [updated],[velocity[0]]


class SPORA(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        self.config.validate()
        c = self.config
        self.embedding = nn.Embedding(c.vocab_size, c.width)
        self.position = nn.Embedding(c.max_length, c.width)
        self.blocks = nn.ModuleList([CausalBlock(c) for _ in range(c.layers)])
        site_type=MemorySite if c.memory_kind=='mlp' else DeltaMemorySite
        self.memories = nn.ModuleList([site_type(c) for _ in c.memory_layers])
        self.final_norm = nn.LayerNorm(c.width)
        nn.init.normal_(self.embedding.weight, std=.1)
        nn.init.normal_(self.position.weight, std=.02)

    def initial_state(self, batch_size, differentiable=False):
        fast, velocity = [], []
        for site in self.memories:
            params = [p.unsqueeze(0).expand(batch_size, *p.shape).clone() for p in site.initial]
            if not differentiable:
                params = [p.detach().requires_grad_(True) for p in params]
            fast.append(params)
            velocity.append([torch.zeros_like(p) for p in params])
        return fast, velocity

    def query_tokens(self, keys):
        if keys.ndim != 1 or bool(((keys < 0) | (keys >= self.config.key_count)).any()):
            raise ValueError('Keys must be a vector within the symbolic vocabulary')
        return torch.stack([torch.ones_like(keys), torch.full_like(keys, 3),
                            keys+self.config.key_start], dim=1)

    def forward(self, keys, fast):
        tokens = self.query_tokens(keys)
        x = self.embedding(tokens)+self.position(torch.arange(tokens.shape[1], device=keys.device))
        key_embedding = self.embedding(tokens[:, -1])
        site_index = 0
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index in self.config.memory_layers:
                site = self.memories[site_index]
                x = site.read(x, site.address(key_embedding), fast[site_index])
                site_index += 1
        c = self.config
        return F.linear(self.final_norm(x[:, -1]),
                        self.embedding.weight[c.value_start:c.value_start+c.value_count])

    def predict_many(self, keys, fast):
        b, n = keys.shape
        expanded = [[p.repeat_interleave(n, dim=0) for p in site] for site in fast]
        return self(keys.flatten(), expanded).reshape(b, n, -1)

    def write_event(self, fast, velocity, keys, values, differentiable=False, trusted=None):
        if bool(((values < 0) | (values >= self.config.value_count)).any()):
            raise ValueError('Values outside vocabulary')
        self.query_tokens(keys)
        gate = torch.ones_like(keys, dtype=self.embedding.weight.dtype) if trusted is None else trusted.to(
            device=keys.device,dtype=self.embedding.weight.dtype)
        if gate.shape != keys.shape or not bool(torch.isfinite(gate).all()) or bool(((gate < 0) | (gate > 1)).any()):
            raise ValueError('Invalid write gate')
        key_embed = self.embedding(keys+self.config.key_start)
        value_embed = self.embedding(values+self.config.value_start)
        new_fast, new_velocity = [], []
        for site, params, momentum in zip(self.memories, fast, velocity):
            p, u = site.write(params, momentum, site.address(key_embed),
                              site.target(value_embed), gate, differentiable)
            new_fast.append(p)
            new_velocity.append(u)
        return new_fast, new_velocity

    def auxiliary_loss(self):
        c = self.config
        labels = torch.arange(c.value_count, device=self.embedding.weight.device)
        embedding = self.embedding(labels+c.value_start)
        return sum(F.cross_entropy(site.value_decoder(site.target(embedding)), labels)
                   for site in self.memories) / len(self.memories)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def config_dict(self):
        return asdict(self.config)
