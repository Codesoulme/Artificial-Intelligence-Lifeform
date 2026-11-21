"""Fuse pretrained Qwen2 attention projections for SPORA inference."""

import copy

import torch
from torch import nn
from torch.nn import functional as F
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, apply_rotary_pos_emb

from .integrated_open_base import SPORAGatedFeedForward


class SPORAPositionTable(nn.Module):
    """Trainable per-position mixing coefficients initialized from a static rotary encoder."""

    def __init__(self, source, max_position):
        super().__init__()
        if source.rope_type != 'default' or not 0 < max_position <= source.config.max_position_embeddings:
            raise ValueError('Expected a bounded position table from a static rotary encoder')
        positions = torch.arange(max_position, device=source.inv_freq.device).unsqueeze(0)
        prototype = torch.empty((1, 1, 1), dtype=torch.float32, device=positions.device)
        cosine, sine = source(prototype, positions)
        half = cosine.shape[-1] // 2
        initial = torch.cat((cosine[0, :, :half], sine[0, :, :half]), dim=-1)
        self.coefficients = nn.Embedding.from_pretrained(initial.clone(), freeze=False)
        self.register_buffer('_initial_coefficients', initial, persistent=False)

    def learned_coefficients(self):
        weights = self.coefficients.weight.detach()
        return None if torch.equal(weights, self._initial_coefficients) else weights

    @property
    def max_position(self):
        return self.coefficients.num_embeddings

    def forward(self, hidden_states, position_ids):
        cosine, sine = self.coefficients(position_ids).to(hidden_states.dtype).chunk(2, dim=-1)
        return torch.cat((cosine, cosine), dim=-1), torch.cat((sine, sine), dim=-1)


class SPORAFusedAttention(nn.Module):
    """Use one input projection while preserving rotary positions and attention weights."""

    def __init__(self, source, *, keep_groups=False):
        super().__init__()
        if source.config._attn_implementation != 'sdpa' or source.sliding_window is not None:
            raise ValueError('Expected full Qwen2 SDPA attention')
        self.layer_idx = source.layer_idx
        self.head_dim = source.head_dim
        self.scaling = source.scaling
        self.is_causal = source.is_causal
        groups = source.num_key_value_groups
        self.num_key_value_groups = groups if keep_groups else 1
        query, key, value = source.q_proj, source.k_proj, source.v_proj

        def expand(tensor):
            heads = tensor.detach().unflatten(0, (-1, self.head_dim))
            return heads.repeat_interleave(groups, dim=0).flatten(0, 1)

        if key.out_features * groups != query.out_features or value.out_features != key.out_features:
            raise ValueError('Incompatible attention projection dimensions')
        key_weight = key.weight.detach() if keep_groups else expand(key.weight)
        value_weight = value.weight.detach() if keep_groups else expand(value.weight)
        key_bias = key.bias.detach() if keep_groups else expand(key.bias)
        value_bias = value.bias.detach() if keep_groups else expand(value.bias)
        self.sizes = (query.out_features, key_weight.shape[0], value_weight.shape[0])
        weights = torch.cat((query.weight.detach(), key_weight, value_weight))
        bias = torch.cat((query.bias.detach(), key_bias, value_bias))
        self.qkv = torch.nn.Linear(query.in_features, sum(self.sizes), device='meta')
        self.qkv.weight = torch.nn.Parameter(weights, requires_grad=False)
        self.qkv.bias = torch.nn.Parameter(bias, requires_grad=False)
        self.out = source.o_proj

    def project(self, hidden_states, position_embeddings, position_ids):
        shape = hidden_states.shape[:-1]
        query, key, value = (projection.reshape(*shape, -1, self.head_dim).transpose(1, 2)
                             for projection in self.qkv(hidden_states).split(self.sizes, dim=-1))
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        return query, key, value

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None,
                cache_position=None, position_ids=None, **kwargs):
        dtype = hidden_states.dtype
        shape = hidden_states.shape[:-1]
        query, key, value = self.project(hidden_states.to(self.qkv.weight.dtype), position_embeddings, position_ids)
        if past_key_values is not None:
            cos, sin = position_embeddings
            position = dict(sin=sin, cos=cos, cache_position=cache_position)
            key, value = past_key_values.update(key, value, self.layer_idx, position)
        hidden, weights = sdpa_attention_forward(self, query, key, value, attention_mask,
                                                 dropout=0., scaling=self.scaling, **kwargs)
        return self.out(hidden.reshape(*shape, -1).contiguous()).to(dtype), weights


class SPORAPretrainedBlock(Qwen2DecoderLayer):
    """Fused SPORA projections with the Qwen2 inference and feature capture interface."""

    def __init__(self, source, read_memory=None):
        nn.Module.__init__(self)
        self.hidden_size = source.hidden_size
        self.attention_type = source.attention_type
        self.input_layernorm = source.input_layernorm
        self.self_attn = (source.self_attn if isinstance(source.self_attn, SPORAFusedAttention)
                          else SPORAFusedAttention(source.self_attn, keep_groups=True))
        self.post_attention_layernorm = source.post_attention_layernorm
        self.mlp = (source.mlp if isinstance(source.mlp, SPORAGatedFeedForward)
                    else SPORAGatedFeedForward(source.mlp, read_memory))


class SPORAResidualBlock(nn.Module):
    """A cache-safe Qwen block with a zero-initialized SPORA residual path.

    The frozen native block remains the identity-preserving carrier while the
    small path learns a new structural contribution. It does not introduce a
    second attention cache, so the native cache protocol remains authoritative.
    """

    attention_type = 'full_attention'

    def __init__(self, source, rank=16):
        super().__init__()
        if not isinstance(source, SPORAPretrainedBlock):
            raise ValueError('Expected a Qwen-compatible SPORA block')
        if type(rank) is not int or not 1 <= rank <= source.hidden_size:
            raise ValueError('Residual rank is outside the hidden width')
        self.base = source.requires_grad_(False).eval()
        self.hidden_size = source.hidden_size
        self.rank = rank
        reference = source.self_attn.qkv.weight
        self.down = nn.Linear(self.hidden_size, rank, bias=False,
                              device=reference.device, dtype=torch.float32)
        self.up = nn.Linear(rank, self.hidden_size, bias=False,
                            device=reference.device, dtype=torch.float32)
        nn.init.normal_(self.down.weight, mean=0., std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, *args, **kwargs):
        output = self.base(*args, **kwargs)
        hidden = output[0] if isinstance(output, tuple) else output
        residual = self.up(F.gelu(self.down(hidden.to(self.down.weight.dtype))))
        updated = hidden + residual.to(dtype=hidden.dtype)
        if isinstance(output, tuple):
            return (updated, *output[1:])
        return updated

    def residual_norm(self):
        return float((self.down.weight.detach().float().norm()
                      * self.up.weight.detach().float().norm()).item())


class AbsoluteAttention(SPORAFusedAttention):
    def __init__(self, source, max_position):
        super().__init__(source)
        self.out = copy.deepcopy(self.out)
        self.max_position = max_position
        self.position = torch.nn.Embedding(self.max_position, source.q_proj.in_features,
                                           device=source.q_proj.weight.device, dtype=torch.float32)
        torch.nn.init.zeros_(self.position.weight)
        self.rotary_fraction = 0.

    def project(self, hidden_states, position_embeddings, position_ids):
        if position_ids is None or bool(((position_ids < 0) | (position_ids >= self.max_position)).any()):
            raise ValueError('Position is outside the learned table')
        values = hidden_states + (1 - self.rotary_fraction) * self.position(position_ids)
        shape = values.shape[:-1]
        qkv = self.qkv(values).reshape(*shape, 3, -1, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if self.rotary_fraction:
            cos, sin = (component.to(query.dtype) for component in position_embeddings)
            q_rotated, k_rotated = apply_rotary_pos_emb(query, key, cos, sin)
            query = query.lerp(q_rotated, self.rotary_fraction)
            key = key.lerp(k_rotated, self.rotary_fraction)
        return query, key, value


def load_absolute_attention(path, source, expected_source_sha256, expected_candidate_sha256):
    from hashlib import sha256
    from io import BytesIO

    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected_candidate_sha256:
        raise ValueError('Absolute attention candidate digest mismatch')
    payload = torch.load(BytesIO(raw), map_location='cpu', weights_only=True)
    max_position = payload['max_position']
    if (payload['format'] != 'spora-absolute-attention-v1' or payload['source'] != expected_source_sha256
            or payload['layer'] != source.layer_idx or type(max_position) is not int or not 0 < max_position <= 65536):
        raise ValueError('Absolute attention candidate belongs to a different layer or base checkpoint')
    state = payload['state']
    if any(not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError('Absolute attention requires finite parameter tensors')
    candidate = AbsoluteAttention(source, max_position).to(source.q_proj.weight.dtype).requires_grad_(False).eval()
    candidate.load_state_dict(state, strict=True)
    return candidate


def load_layer_norms(path, layer, expected_source_sha256, expected_candidate_sha256):
    from hashlib import sha256
    from io import BytesIO

    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected_candidate_sha256:
        raise ValueError('LayerNorm candidate digest mismatch')
    payload = torch.load(BytesIO(raw), map_location='cpu', weights_only=True)
    if payload['source_sha256'] != expected_source_sha256 or payload['layer'] != layer.self_attn.layer_idx:
        raise ValueError('LayerNorm candidate belongs to a different layer or base checkpoint')
    modules = []
    for key, reference in (('pre_attention', layer.input_layernorm),
                           ('pre_feedforward', layer.post_attention_layernorm)):
        state = payload[key]
        if any(not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all())
               for value in state.values()):
            raise ValueError('LayerNorm candidate requires finite parameter tensors')
        norm = nn.LayerNorm(reference.weight.numel()).to(reference.weight)
        norm.load_state_dict(state, strict=True)
        modules.append(norm.requires_grad_(False).eval())
    return tuple(modules)
