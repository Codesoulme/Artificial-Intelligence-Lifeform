"""Install a digest-bound recurrent attention candidate for local greedy dialogue."""

from contextlib import contextmanager
import json
from pathlib import Path

import torch
from torch import nn
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from .pretrained_attention import SPORAFusedAttention, SPORAResidualBlock
from .integrated_checkpoint import _file_sha256
from .research_recurrent_attention import RECURRENT_UPDATE_RULES


class RecurrentDecodeLayer(DynamicLayer):
    """Carry the bounded recurrent state beside the native cache layers."""

    def __init__(self, core):
        super().__init__()
        self.core = core
        self.state = None

    def consume(self, query, key, value, *, include_far, batched_prefill, history=None):
        output, self.state = self.core(query, key, value, state=self.state, include_far=include_far,
                                      batched_prefill=batched_prefill, history=history)
        self.keys, self.values = self.state.key, self.state.value
        self.is_initialized = True
        self.device, self.dtype = self.keys.device, self.keys.dtype
        return output

    def get_seq_length(self):
        return 0 if self.state is None else self.state.length

    def reset(self):
        self.state = self.keys = self.values = None
        self.is_initialized = False

    def update(self, key_states, value_states, cache_kwargs=None):
        raise ValueError('The recurrent layer needs query, key and value together')

    def crop(self, max_length):
        if max_length < 0 or max_length < self.get_seq_length():
            raise ValueError('The recurrent layer cannot undo folded history')

    def reorder_cache(self, beam_idx):
        if beam_idx.shape != (1,) or int(beam_idx[0]) != 0:
            raise ValueError('The recurrent layer supports a single greedy sequence')

    def batch_repeat_interleave(self, repeats):
        if repeats != 1:
            raise ValueError('The recurrent layer supports a single greedy sequence')

    def batch_select_indices(self, indices):
        self.reorder_cache(indices)


class AttentionReplacement(nn.Module):
    def __init__(self, source, core, *, include_far=True, history_projection='rotary'):
        super().__init__()
        self.source = source
        self.core = core
        self.include_far = include_far
        self.history_projection = history_projection
        self.batched_prefill = True
        self.batched_training = False

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None,
                position_ids=None, cache_position=None, **kwargs):
        if hidden_states.shape[0] != 1 or attention_mask is not None:
            raise ValueError('Recurrent attention probe takes one unpadded causal sequence')
        qkv, history = project_with_history(self.source, hidden_states, position_embeddings, position_ids,
                                            self.history_projection)
        if past_key_values is None:
            output, _ = self.core(*qkv, include_far=self.include_far,
                                  batched_prefill=self.batched_training and self.core.training, history=history)
        else:
            if type(past_key_values) is not DynamicCache or past_key_values.offloading:
                raise ValueError('Recurrent attention probe expects a non-offloaded dynamic cache')
            layers, index = past_key_values.layers, self.source.layer_idx
            if index == len(layers):
                layers.append(RecurrentDecodeLayer(self.core))
            layer = layers[index]
            if type(layer) is DynamicLayer and layer.get_seq_length() == 0:
                layer = layers[index] = RecurrentDecodeLayer(self.core)
            if not isinstance(layer, RecurrentDecodeLayer) or layer.core is not self.core:
                raise ValueError('Recurrent layer cache provenance differs from the active attention')
            if cache_position is None or int(cache_position[0]) != layer.get_seq_length():
                raise ValueError('Recurrent layer cache requires append-only positions')
            output = layer.consume(*qkv, include_far=self.include_far, batched_prefill=self.batched_prefill, history=history)
        mixed = output.transpose(1, 2).flatten(start_dim=-2).contiguous()
        return self.source.out(mixed.to(self.source.out.weight.dtype)).to(hidden_states.dtype), None


def projections(source, hidden, rope, position_ids):
    return source.project(hidden.to(source.qkv.weight.dtype), rope, position_ids)


def project_with_history(source, hidden, rope, position_ids, mode):
    if mode == 'rotary':
        return projections(source, hidden, rope, position_ids), None
    if mode != 'unrotated' or type(source) is not SPORAFusedAttention:
        raise ValueError('Unrotated history requires native rotary fused projections')
    query, key, value = source.project_unrotated(hidden.to(source.qkv.weight.dtype))
    local_query, local_key = apply_rotary_pos_emb(query, key, *rope)
    return (local_query, local_key, value), (query, key)


def attention_block(decoder, index):
    layer = decoder.base.model.layers[index]
    block = layer.layer if hasattr(layer, 'layer') else layer
    if isinstance(block, SPORAResidualBlock):
        block = block.base
    if not isinstance(getattr(block, 'self_attn', None), SPORAFusedAttention):
        raise ValueError('Recurrent migration expects the fused SPORA attention projections')
    return block


@contextmanager
def replace(block, replacement):
    original = block.self_attn
    block.self_attn = replacement
    try:
        yield
    finally:
        block.self_attn = original


def load_recurrent_replacements(decoder, directories, checkpoint_sha256, first_layer=0):
    """Validate every candidate before changing a decoder layer."""
    replacements = {}
    for directory in directories:
        path = Path(directory) / 'attention.pt'
        comparison_path = path.with_name('comparison.json')
        comparison = json.loads(comparison_path.read_text(encoding='utf-8'))
        candidate_sha = _file_sha256(path)
        if comparison['candidate_sha256'] != candidate_sha:
            raise ValueError('Recurrent attention weights differ from the comparison')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if (saved['format'] != 'spora-recurrent-attention-v1' or saved['checkpoint_sha256'] != checkpoint_sha256
                or _file_sha256(path) != candidate_sha):
            raise ValueError('Recurrent attention belongs to a different dialogue checkpoint')
        index, width = saved['layer'], saved['block']
        if (type(index) is not int or not first_layer <= index < len(decoder.base.model.layers)
                or index in replacements or index != comparison['layer']
                or type(width) is not int or not 1 <= width <= 256 or width != comparison['block']):
            raise ValueError('Recurrent attention contains an invalid or conflicting layer')
        block = attention_block(decoder, index)
        source = block.self_attn
        rule = saved.get('update_rule', 'sum')
        if rule != comparison.get('update_rule', 'sum') or rule not in RECURRENT_UPDATE_RULES:
            raise ValueError('Recurrent attention update rule differs from its comparison')
        history = saved.get('history_projection', 'rotary')
        if history != comparison.get('history_projection', 'rotary') or history not in ('rotary', 'unrotated'):
            raise ValueError('Recurrent history projection differs from its comparison')
        if history == 'unrotated' and type(source) is not SPORAFusedAttention:
            raise ValueError('Unrotated history requires native rotary fused projections')
        core_type = RECURRENT_UPDATE_RULES[rule]
        core = core_type(source.sizes[0] // source.head_dim, source.head_dim, block=width)
        core.load_state_dict(saved['core'], strict=True)
        if any(not bool(torch.isfinite(parameter).all()) for parameter in core.parameters()):
            raise ValueError('Recurrent attention contains nonfinite weights')
        core.to(decoder.base.device).eval().requires_grad_(False)
        replacements[index] = (block, AttentionReplacement(source, core, history_projection=history).eval())
    return replacements
