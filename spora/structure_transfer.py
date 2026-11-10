"""SPORA-native causal block for layer-by-layer teacher distillation."""
import torch
from torch import nn

from .model import CausalBlock, ModelConfig


class SPORALayerReplacement(nn.Module):
    """A distilled SPORA causal block with learned absolute position input.

    It accepts the Qwen2 decoder-layer call shape for full-sequence evaluation.
    Autoregressive KV caching is deliberately rejected until an incremental
    cache implementation is added and parity-tested.
    """

    attention_type = 'full_attention'

    def __init__(self, width, heads, max_length=512):
        super().__init__()
        if (type(width) is not int or width <= 0 or type(heads) is not int or heads <= 0
                or width % heads or type(max_length) is not int or max_length <= 0):
            raise ValueError('Invalid SPORA replacement dimensions')
        config = ModelConfig(width=width, heads=heads, max_length=max_length)
        config.validate()
        self.width, self.max_length = width, max_length
        self.position = nn.Embedding(max_length, width)
        self.block = CausalBlock(config)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
                use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
        if use_cache or past_key_values is not None:
            raise ValueError('SPORA layer replacement does not support KV cache yet')
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3 or hidden_states.shape[-1] != self.width:
            raise ValueError('Expected hidden states with shape [batch, sequence, width]')
        batch, length = hidden_states.shape[:2]
        if not length or length > self.max_length:
            raise ValueError('Sequence length outside replacement position range')
        if attention_mask is not None and attention_mask.ndim >= 2:
            # The prototype only supports ordinary causal full-sequence input.
            if attention_mask.shape[-1] != length:
                raise ValueError('Attention mask and hidden sequence lengths differ')
        if position_ids is None:
            position_ids = torch.arange(length, device=hidden_states.device).expand(batch, -1)
        if (position_ids.shape != (batch, length) or position_ids.dtype != torch.long
                or bool(((position_ids < 0) | (position_ids >= self.max_length)).any())):
            raise ValueError('Invalid absolute position IDs')
        hidden_dtype = hidden_states.dtype
        positions = self.position(position_ids.to(self.position.weight.device))
        values = hidden_states.to(device=positions.device, dtype=positions.dtype) + positions
        return self.block(values).to(device=hidden_states.device, dtype=hidden_dtype)
