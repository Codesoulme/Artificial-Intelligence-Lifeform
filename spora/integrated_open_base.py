"""Experimental decoder-layer integration of a SPORA memory reader.

Construction wraps two layers of the supplied base model in place. The wrapper
owns the reader and passes immutable, per-call memory snapshots via a context
variable; no forward hooks are installed. The caller owns persistence, routing,
and any concurrency policy for the underlying language model.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import math

import torch
from torch import nn

from .model import DeltaMemorySite, ModelConfig
from .memory_output import MemoryOutputAdapter, MemoryOutputLogitsProcessor
from .open_base_memory import OpenBaseMemory
from .sleep_memory import history_site


class ScaledOpenBaseMemory(OpenBaseMemory):
    """Reader whose residual is scaled after the base dtype conversion."""

    def __init__(self, hidden_width, memory_width, strength):
        super().__init__(hidden_width, memory_width)
        if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError('Strength must be finite and in [0, 1]')
        self.strength = float(strength)

    def forward(self, hidden, matrices, site, address=None):
        injected = super().forward(hidden, matrices, site, address)
        return hidden + self.strength * (injected - hidden)


class NativeSPORAMemoryReader(ScaledOpenBaseMemory):
    """Use SPORA's native DeltaMemorySite as the decoder memory reader.

    The supplied memory matrices are the same fast weights used by SPORA.
    ``initialize_from_legacy`` maps the trained open-base projections into the
    native site's readout and gate, with its slow expert path initially zeroed.
    """

    def __init__(self, hidden_width, memory_width, strength):
        super().__init__(hidden_width, memory_width, strength)
        del self.query, self.out, self.gate
        config = ModelConfig(width=hidden_width, heads=1, memory_width=memory_width,
                             expert_rank=min(16, memory_width), memory_kind='delta')
        self.native_sites = nn.ModuleList([DeltaMemorySite(config) for _ in range(2)])
        for site in self.native_sites:
            with torch.no_grad():
                site.read_out.weight.zero_()
                site.read_out.bias.zero_()
                for expert in site.experts:
                    expert[-1].weight.zero_()
                    expert[-1].bias.zero_()

    def initialize_from_legacy(self, legacy):
        if not isinstance(legacy, OpenBaseMemory) or isinstance(legacy, NativeSPORAMemoryReader):
            raise ValueError('Expected the legacy scaled reader')
        if (legacy.hidden_width != self.hidden_width or legacy.memory_width != self.memory_width
                or getattr(legacy, 'strength', self.strength) != self.strength):
            raise ValueError('Legacy reader dimensions or strength do not match')
        with torch.no_grad():
            for index, site in enumerate(self.native_sites):
                site.read_out.weight.copy_(legacy.out[index].weight)
                site.read_out.bias.zero_()
                site.read_gate.weight.zero_()
                site.read_gate.bias.fill_(legacy.gate[index])

    def forward(self, hidden, matrices, site, address=None):
        if type(site) is not int or site not in (0, 1):
            raise ValueError('Site must be 0 or 1')
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[-1] != self.hidden_width:
            raise ValueError('Layer output must have shape [batch, sequence, hidden_width]')
        if not hidden.is_floating_point() or not hidden.shape[0] or not matrices:
            raise ValueError('Layer output and memory matrices must be nonempty floating tensors')
        native_site = self.native_sites[site]
        parameter = native_site.read_out.weight
        x = hidden.to(device=parameter.device, dtype=parameter.dtype)
        batch_size, sequence_length = x.shape[:2]
        if address is None:
            query = native_site.address(x.reshape(batch_size * sequence_length, self.hidden_width))
            query = query.reshape(batch_size, sequence_length, self.memory_width)
        else:
            valid_shapes = ((1, self.memory_width), (hidden.shape[0], self.memory_width))
            if (not isinstance(address, torch.Tensor) or address.ndim != 2
                    or tuple(address.shape) not in valid_shapes
                    or not address.is_floating_point() or not bool(torch.isfinite(address).all())):
                raise ValueError('Address must have shape [1 or batch, memory_width]')
            query = torch.nn.functional.normalize(
                address.detach().to(device=x.device, dtype=x.dtype), dim=-1).expand(hidden.shape[0], -1)
            query = query[:, None, :].expand(-1, sequence_length, -1)
        fixed_matrices = []
        for matrix in matrices:
            if (not isinstance(matrix, torch.Tensor) or matrix.ndim != 3
                    or matrix.shape[1:] != (self.memory_width, self.memory_width)
                    or matrix.shape[0] not in (1, hidden.shape[0]) or not matrix.is_floating_point()):
                raise ValueError('Memory matrices must have shape [1 or batch, memory_width, memory_width]')
            fixed_matrices.append(matrix.detach().to(device=x.device, dtype=x.dtype).expand(hidden.shape[0], -1, -1))
        token_hidden = x.reshape(batch_size * sequence_length, 1, self.hidden_width)
        token_query = query.reshape(batch_size * sequence_length, self.memory_width)

        def read_page(matrix):
            retrieved = torch.bmm(matrix, query.transpose(1, 2)).transpose(1, 2)
            output = native_site.read_retrieved(token_hidden, token_query, retrieved.reshape_as(token_query))
            return retrieved, output.reshape_as(x)

        if len(fixed_matrices) == 1:
            _, injected = read_page(fixed_matrices[0])
        else:
            page_outputs, page_scores = [], []
            for matrix in fixed_matrices:
                retrieved, output = read_page(matrix)
                page_scores.append((retrieved * query).sum(-1))
                page_outputs.append(output)
            weights = torch.stack(page_scores, dim=1).softmax(dim=1)
            residuals = torch.stack([output - x for output in page_outputs], dim=1)
            injected = x + (residuals * weights[:, :, :, None]).sum(1)
        return hidden + self.strength * (injected.to(device=hidden.device, dtype=hidden.dtype) - hidden)


class SPORAGatedFeedForward(nn.Module):
    """Fused pretrained feed-forward projection with a native memory input."""

    def __init__(self, source, read_memory=None):
        super().__init__()
        gate, up, down = source.gate_proj, source.up_proj, source.down_proj
        if gate.weight.shape != up.weight.shape or gate.bias is not None or up.bias is not None:
            raise ValueError('Expected matching bias-free gate and up projections')
        weights = torch.cat((gate.weight.detach(), up.weight.detach()), dim=0)
        self.projection = nn.Linear(gate.in_features, 2 * gate.out_features, bias=False, device='meta')
        self.projection.weight = nn.Parameter(weights, requires_grad=gate.weight.requires_grad or up.weight.requires_grad)
        self.down = down
        self.activation = source.act_fn
        self.read_memory = read_memory
        self.read_memory_at_output = False

    def forward(self, hidden):
        conditioned = hidden
        if self.read_memory is not None:
            conditioned = self.read_memory(hidden)
        projected = hidden if self.read_memory_at_output else conditioned
        gate, value = self.projection(projected).chunk(2, dim=-1)
        output = self.down(self.activation(gate) * value)
        if self.read_memory_at_output and conditioned is not hidden:
            delta = conditioned.float() - hidden.float()
            output = output + delta.to(output.dtype)
        return output


class _MemoryDecoderLayer(nn.Module):
    def __init__(self, layer, reader, site, context):
        super().__init__()
        self.layer = layer
        self.attention_type = layer.attention_type
        self.site = site
        self.feedforward_fraction = 0.
        object.__setattr__(self, '_reader', reader)
        object.__setattr__(self, '_memory_context', context)

    def forward(self, *args, **kwargs):
        output = self.layer(*args, **kwargs)
        if self.feedforward_fraction == 1. or self._memory_context.get() is None:
            return output
        read = self.read_memory(output)
        if self.feedforward_fraction == 0.:
            return read
        return output + (1. - self.feedforward_fraction) * (read - output)

    def read_feedforward_memory(self, hidden):
        if self.feedforward_fraction == 0. or self._memory_context.get() is None:
            return hidden
        read = self.read_memory(hidden)
        if self.feedforward_fraction == 1.:
            return read
        return hidden + self.feedforward_fraction * (read - hidden)

    def read_memory(self, output):
        active = self._memory_context.get()
        if active is None:
            return output
        matrices, addresses, read_mask, memory_slots = active
        address = None if addresses is None else addresses[self.site]
        if read_mask is None and memory_slots is None:
            return self._reader(output, matrices[self.site], self.site, address)
        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            raise ValueError('Layer output must have shape [batch, sequence, hidden_width]')
        batch = output.shape[0]
        if read_mask is not None and read_mask.shape[0] not in (1, batch):
            raise ValueError('Memory read mask must match the decoder batch')
        mask = (torch.ones(batch, dtype=torch.bool, device=output.device) if read_mask is None
                else read_mask.to(device=output.device).expand(batch))
        selected = mask.nonzero(as_tuple=True)[0]
        if not selected.numel():
            return output

        def selected_rows(tensor, name):
            if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1 or tensor.shape[0] not in (1, batch):
                raise ValueError(f'{name} must have a singleton or matching batch')
            return tensor if tensor.shape[0] == 1 else tensor.index_select(0, selected.to(tensor.device))

        site_matrices = matrices[self.site]
        if memory_slots is None:
            selected_matrices = tuple(selected_rows(matrix, 'Memory matrix') for matrix in site_matrices)
        else:
            slots = memory_slots.to(device=output.device).expand(batch).index_select(0, selected)
            if bool(((slots < 0) | (slots >= len(site_matrices))).any()):
                raise ValueError('Memory slot outside the supplied current/history pages')
            expanded_matrices = []
            for matrix in site_matrices:
                if (not isinstance(matrix, torch.Tensor) or matrix.ndim != 3
                        or matrix.shape[0] not in (1, batch)):
                    raise ValueError('Memory matrix must have a singleton or matching batch')
                expanded_matrices.append(matrix.expand(batch, -1, -1))
            stacked_matrices = torch.stack(expanded_matrices, dim=1)
            row_indices = selected.to(stacked_matrices.device)
            row_slots = slots.to(stacked_matrices.device)
            selected_matrix = stacked_matrices[row_indices, row_slots]
            selected_matrices = (selected_matrix,)
        selected_address = None if address is None else selected_rows(address, 'Memory address')
        selected_output = output.index_select(0, selected)
        injected = self._reader(selected_output, selected_matrices, self.site, selected_address)
        return output.index_copy(0, selected, injected)


class IntegratedOpenBase(nn.Module):
    """A language model with two explicit read layers and per-call state.

    Memory is optional so that ordinary base forwards remain possible. The
    reader is applied only when ``memory_state`` is supplied to ``forward`` or
    ``generate``. A model cache must not be reused across memory changes.
    ``memory_read_mask`` optionally selects which batch rows receive the read;
    its boolean vector must have length one or match the decoder batch. The
    caller decides the mask and must account for any generation batch expansion.
    Masked rows skip the reader; state snapshots are still created for the call.
    This is not an access control boundary.
    ``memory_slots`` optionally selects one matrix per row: slot 0 is current
    memory and slots 1..N are history pages in chronological state order. This
    is a caller-supplied temporal address, not inferred from natural language.
    """

    def __init__(self, base, reader, layer_indices, output_adapter=None):
        super().__init__()
        if not isinstance(base, nn.Module) or not isinstance(reader, OpenBaseMemory):
            raise ValueError('Expected a base module and an OpenBaseMemory reader')
        layers = getattr(getattr(base, 'model', None), 'layers', None)
        if not isinstance(layers, nn.ModuleList):
            raise ValueError('Expected base.model.layers as a ModuleList')
        if (not isinstance(layer_indices, (tuple, list)) or len(layer_indices) != 2
                or any(type(i) is not int or not 0 <= i < len(layers) for i in layer_indices)
                or layer_indices[0] == layer_indices[1]):
            raise ValueError('Expected two distinct layer indices')
        selected = [layers[i] for i in layer_indices]
        if any(isinstance(layer, _MemoryDecoderLayer) or not hasattr(layer, 'attention_type') for layer in selected):
            raise ValueError('Layers must be unwrapped tensor-output decoder layers')
        self.base = base
        self.reader = reader
        if output_adapter is not None and not isinstance(output_adapter, MemoryOutputAdapter):
            raise ValueError('Expected a memory output adapter')
        self.output_adapter = output_adapter
        self.layer_indices = tuple(layer_indices)
        self.compatible_block_indices = ()
        self.position_table_max_position = None
        self._memory_context = ContextVar('spora_integrated_memory', default=None)
        for site, index in enumerate(self.layer_indices):
            layers[index] = _MemoryDecoderLayer(layers[index], reader, site, self._memory_context)

    def migrate_pretrained_blocks(self, indices=None):
        """Convert selected Qwen2 blocks while retaining their native functions.

        The migrated module is a SPORA-owned wrapper around the original Qwen2
        attention, position, normalization, and feed-forward computations. It
        is therefore a structural transition step: ordinary forwards remain
        function-preserving, while later experiments may add state pathways.
        """
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        from .pretrained_attention import SPORAPretrainedBlock

        if self.training:
            raise ValueError('Pretrained block migration requires an evaluation model')
        layers = self.base.model.layers
        if indices is None:
            selected = tuple(index for index, layer in enumerate(layers)
                             if type(layer.layer if isinstance(layer, _MemoryDecoderLayer) else layer)
                             is Qwen2DecoderLayer)
        elif (not isinstance(indices, (tuple, list, set))
              or any(type(index) is not int or not 0 <= index < len(layers) for index in indices)):
            raise ValueError('Compatible block indices must be valid layer integers')
        else:
            selected = tuple(sorted(set(indices)))
        migrated = []
        for index in selected:
            layer = layers[index]
            source = layer.layer if isinstance(layer, _MemoryDecoderLayer) else layer
            if isinstance(source, SPORAPretrainedBlock):
                migrated.append(index)
            elif type(source) is Qwen2DecoderLayer:
                if isinstance(layer, _MemoryDecoderLayer):
                    layer.layer = SPORAPretrainedBlock(layer.layer, layer.read_feedforward_memory).eval()
                else:
                    layers[index] = SPORAPretrainedBlock(layer).eval()
                migrated.append(index)
            else:
                raise ValueError(f'Layer {index} is not an original Qwen2 decoder layer')
        self.compatible_block_indices = tuple(sorted(set(self.compatible_block_indices) | set(migrated)))
        return tuple(migrated)

    def migrate_position_encoder(self, max_position):
        """Move shared position coefficients into an editable model parameter table."""
        from .pretrained_attention import SPORAPositionTable

        backbone = self.base.model
        if isinstance(backbone.rotary_emb, SPORAPositionTable):
            if backbone.rotary_emb.max_position != max_position:
                raise ValueError('Position encoder already migrated with a different limit')
        else:
            backbone.rotary_emb = SPORAPositionTable(backbone.rotary_emb, max_position)
        self.position_table_max_position = max_position
        return backbone.rotary_emb

    def use_feedforward_memory(self, layer_index, *, fraction=1., at_output=False):
        """Move a fraction of a memory read into its gated feed-forward network."""
        if type(layer_index) is not int or layer_index not in self.layer_indices:
            raise ValueError('Expected a configured memory layer index')
        if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0. <= fraction <= 1.:
            raise ValueError('Feed-forward memory fraction must be finite and in [0, 1]')
        if type(at_output) is not bool:
            raise ValueError('Feed-forward output placement must be boolean')
        wrapper = self.base.model.layers[layer_index]
        block = wrapper.layer
        source = getattr(block, 'mlp', None)
        if source is None:
            from .pretrained_attention import SPORAResidualBlock
            if isinstance(block, SPORAResidualBlock):
                block = block.base
                source = block.mlp
        if not isinstance(source, SPORAGatedFeedForward):
            if source is None or any(not hasattr(source, name) for name in ('gate_proj', 'up_proj', 'down_proj', 'act_fn')):
                raise ValueError('Memory layer does not expose a compatible gated feed-forward network')
            source = SPORAGatedFeedForward(source, wrapper.read_feedforward_memory).train(source.training)
            block.mlp = source
        source.read_memory_at_output = at_output
        wrapper.feedforward_fraction = float(fraction)

    @contextmanager
    def _snapshot(self, state, addresses, current_only, read_mask, memory_slots,
                  preserve_memory_grad=False):
        if type(current_only) is not bool:
            raise ValueError('current_only must be a boolean')
        if type(preserve_memory_grad) is not bool:
            raise ValueError('preserve_memory_grad must be a boolean')
        if self._memory_context.get() is not None:
            raise ValueError('Nested memory calls are not supported')
        if state is None:
            if addresses is not None or read_mask is not None or memory_slots is not None:
                raise ValueError('Addresses and read controls require a memory state')
            yield
            return
        if addresses is not None and (not isinstance(addresses, (tuple, list)) or len(addresses) != 2
                                      or any(not isinstance(address, torch.Tensor) for address in addresses)):
            raise ValueError('Addresses must contain one tensor per site')
        if read_mask is not None and (not isinstance(read_mask, torch.Tensor) or read_mask.dtype != torch.bool
                                      or read_mask.ndim != 1 or not read_mask.numel()):
            raise ValueError('Memory read mask must be a nonempty boolean vector')
        if memory_slots is not None and (not isinstance(memory_slots, torch.Tensor)
                                         or memory_slots.dtype != torch.long or memory_slots.ndim != 1
                                         or not memory_slots.numel()):
            raise ValueError('Memory slots must be a nonempty integer vector')
        snapshot = (lambda tensor: tensor.clone() if preserve_memory_grad else tensor.detach().clone())
        matrices = tuple(tuple(snapshot(t) for t in
                               [state.current[site]] + ([] if current_only else
                                [history_site(page, site)[0] for page in state.pages]))
                         for site in range(2))
        with torch.no_grad():
            fixed_addresses = None if addresses is None else tuple(address.detach().clone() for address in addresses)
            fixed_mask = None if read_mask is None else read_mask.detach().clone()
            fixed_slots = None if memory_slots is None else memory_slots.detach().clone()
        token = self._memory_context.set((matrices, fixed_addresses, fixed_mask, fixed_slots))
        try:
            yield
        finally:
            self._memory_context.reset(token)

    def forward(self, *args, memory_state=None, memory_addresses=None, memory_read_mask=None, memory_slots=None,
                current_only=False, preserve_memory_grad=False, **kwargs):
        if memory_state is not None and len(args) > 1:
            raise ValueError('Memory calls accept at most one positional base argument')
        if memory_state is not None and kwargs.get('past_key_values') is not None:
            raise ValueError('External key-value caches are not supported with memory state')
        with self._snapshot(memory_state, memory_addresses, current_only, memory_read_mask, memory_slots,
                            preserve_memory_grad):
            return self.base(*args, **kwargs)

    def generate(self, *args, memory_state=None, memory_addresses=None, memory_read_mask=None, memory_slots=None,
                 current_only=False, preserve_memory_grad=False, memory_output_logits=None,
                 memory_output_gain=1.0, **kwargs):
        if memory_state is not None and len(args) > 1:
            raise ValueError('Memory calls accept at most one positional base argument')
        if memory_state is not None and kwargs.get('past_key_values') is not None:
            raise ValueError('External key-value caches are not supported with memory state')
        if memory_output_logits is not None:
            if self.output_adapter is None:
                raise ValueError('Memory output adapter is not attached')
            processor = MemoryOutputLogitsProcessor(self.output_adapter, memory_output_logits, memory_output_gain)
            existing = kwargs.get('logits_processor')
            kwargs['logits_processor'] = ([processor] if existing is None else [processor] + list(existing))
        with self._snapshot(memory_state, memory_addresses, current_only, memory_read_mask, memory_slots,
                            preserve_memory_grad):
            return self.base.generate(*args, **kwargs)
