"""Experimental checkpoint joining an individual, reader and optional SPORA layers.

The base weights remain separate. Their on-disk digest and tokenizer/config
fingerprint are checked, but a caller-supplied in-memory base is not proven to
have been loaded from those files. Load it from the verified directory. The
structural format embeds replacement tensors and checks their declared layout;
it does not serialize unsupported blends or the full frozen base.
"""
import hashlib
import json
import math
from pathlib import Path

import torch

from .integrated_open_base import (IntegratedOpenBase, NativeSPORAMemoryReader,
                                   ScaledOpenBaseMemory)
from .memory_output import MemoryOutputAdapter
from .memory_reply import MemoryReplyAdapter
from .language_training import atomic_save
from .named_address import feature_files_sha256, structural_feature_sha256
from .structure_transfer import SPORALayerReplacement, SPORALayerBlend
from .value_position_checkpoint import (QUERY_KEY_POSITION_PREFIX, VALUE_POSITION_PREFIX,
                                        decode_query_key_positions, decode_value_positions,
                                        query_key_position_bundle, value_position_bundle)
from .state_codec import flatten, restore
from .internal_memory import observe_sourced as observe_spora_sourced

FORMAT = 'spora-integrated-open-base-v2'
LEGACY_FORMAT = 'spora-integrated-open-base-v1'
STRUCTURAL_FORMAT = 'spora-integrated-open-base-v3'
OUTPUT_ADAPTER_FORMAT = 'spora-integrated-open-base-v4'


class IntegratedSPORARuntime:
    """Own one decoder, one SPORA individual and its durable evolving state."""

    def __init__(self, decoder, organism, state, model_dir, checkpoint_path=None):
        if not isinstance(decoder, IntegratedOpenBase):
            raise ValueError('Expected an integrated open-base decoder')
        if (not hasattr(organism, 'write_control') or state.internal is None
                or type(decoder.reader) is not NativeSPORAMemoryReader):
            raise ValueError('Runtime requires native SPORA sites and controlled individual memory')
        self.decoder = decoder
        self.organism = organism
        self.state = state
        self.model_dir = Path(model_dir)
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)

    def generate(self, *args, memory_key=None, memory_mode=0, memory_time=0,
                 memory_output=False, use_event_focus=False, **kwargs):
        if 'memory_state' in kwargs or 'memory_output_logits' in kwargs:
            raise ValueError('Runtime owns memory state and output evidence')
        if type(memory_output) is not bool:
            raise ValueError('memory_output must be a boolean')
        if type(use_event_focus) is not bool:
            raise ValueError('use_event_focus must be a boolean')
        if use_event_focus:
            if (memory_key is not None or type(memory_mode) is not int or memory_mode != 0
                    or type(memory_time) is not int or memory_time != 0
                    or any(k in kwargs for k in ('memory_addresses', 'memory_slots', 'current_only', 'memory_read_mask'))):
                raise ValueError('Event focus cannot be overridden by explicit memory controls')
            from .event_focus import focused_memory_args
            controls, selection = focused_memory_args(self.organism, self.state)
            kwargs.update(controls)
            memory_key, memory_mode, memory_time = selection['key'], 1, selection['event_time']
        if memory_output:
            if self.decoder.output_adapter is None:
                raise ValueError('Runtime has no memory output adapter')
            if type(memory_key) is not int or not 0 <= memory_key < self.organism.config.key_count:
                raise ValueError('A valid memory key is required for output conditioning')
            if type(memory_mode) is not int or memory_mode not in (0, 1) or type(memory_time) is not int:
                raise ValueError('Memory mode and time must be valid integers')
            if (memory_mode == 0 and memory_time != 0) or (memory_mode == 1 and not 0 <= memory_time < self.state.cursor):
                raise ValueError('Memory time is outside the observed history')
            from .growing import query
            device = next(self.organism.parameters()).device
            key = torch.tensor([memory_key], dtype=torch.long, device=device)
            mode = torch.tensor([memory_mode], dtype=torch.long, device=device)
            time = torch.tensor([memory_time], dtype=torch.long, device=device)
            memory_output_logits = query(self.organism, self.state, key, mode, time)
            if int(memory_output_logits.argmax(-1)[0]) == self.organism.config.unknown:
                raise ValueError('Cannot condition output on unknown memory')
        else:
            memory_output_logits = None
        return self.decoder.generate(*args, memory_state=self.state,
                                     memory_output_logits=memory_output_logits, **kwargs)

    def observe_sourced(self, key, value, source, quality, valid_at=None, *, checkpoint_path=None,
                       update_event_focus=False):
        """Apply one guarded SPORA write; publish the new state only after save."""
        if type(update_event_focus) is not bool:
            raise ValueError('update_event_focus must be a boolean')
        from .event_focus import observe_focused
        observe = observe_focused if update_event_focus else observe_spora_sourced
        candidate, decision = observe(
            self.organism, self.state, key, value, source, quality, valid_at)
        destination = self.checkpoint_path if checkpoint_path is None else Path(checkpoint_path)
        if destination is not None:
            save_integrated(destination, self.organism, candidate, self.decoder.reader,
                            self.model_dir, self.decoder.layer_indices, replacements=self._replacements(),
                            residuals=self._residuals(),
                            value_positions=value_position_modules(self.decoder),
                            query_key_positions=query_key_position_modules(self.decoder),
                            output_adapter=self.decoder.output_adapter,
                            feedforward_fractions=self._feedforward_fractions(),
                            feedforward_output_layers=self._feedforward_output_layers(),
                            gelu_gate_layers=self._gelu_gate_layers(),
                            compatible_layers=self.decoder.compatible_block_indices,
                            position_table_max_position=self.decoder.position_table_max_position,
                            position_coefficients=self._position_coefficients())
        self.state = candidate
        return decision

    def save(self, path=None):
        destination = self.checkpoint_path if path is None else Path(path)
        if destination is None:
            raise ValueError('A checkpoint destination is required')
        save_integrated(destination, self.organism, self.state, self.decoder.reader,
                        self.model_dir, self.decoder.layer_indices, replacements=self._replacements(),
                        residuals=self._residuals(),
                        value_positions=value_position_modules(self.decoder),
                        query_key_positions=query_key_position_modules(self.decoder),
                        output_adapter=self.decoder.output_adapter,
                        feedforward_fractions=self._feedforward_fractions(),
                        feedforward_output_layers=self._feedforward_output_layers(),
                        gelu_gate_layers=self._gelu_gate_layers(),
                        compatible_layers=self.decoder.compatible_block_indices,
                        position_table_max_position=self.decoder.position_table_max_position,
                        position_coefficients=self._position_coefficients())
        self.checkpoint_path = destination
        return destination

    def _replacements(self):
        layers = self.decoder.base.model.layers
        if any(isinstance(layer, SPORALayerBlend) for layer in layers):
            raise ValueError('Cannot persist an unsupported blended structural layer')
        return {index: layer for index, layer in enumerate(layers) if isinstance(layer, SPORALayerReplacement)}

    def _residuals(self):
        if not self.decoder.compatible_block_indices:
            return {}
        from .pretrained_attention import SPORAResidualBlock

        layers = self.decoder.base.model.layers
        residuals = {}
        for index in self.decoder.compatible_block_indices:
            layer = layers[index]
            candidate = layer.layer if index in self.decoder.layer_indices else layer
            if isinstance(candidate, SPORAResidualBlock):
                residuals[index] = candidate
        return residuals

    def _feedforward_fractions(self):
        """Return nonzero in-layer memory strengths for durable checkpoints."""
        layers = self.decoder.base.model.layers
        return {index: float(layer.feedforward_fraction)
                for index, layer in enumerate(layers)
                if index in self.decoder.layer_indices
                and float(getattr(layer, 'feedforward_fraction', 0.)) != 0.}

    def _feedforward_output_layers(self):
        layers = self.decoder.base.model.layers
        output_layers = []
        residuals = self._residuals()
        for index in self._feedforward_fractions():
            block = layers[index].layer
            if index in residuals:
                block = block.base
            if block.mlp.read_memory_at_output:
                output_layers.append(index)
        return output_layers

    def _position_coefficients(self):
        if self.decoder.position_table_max_position is None:
            return None
        return self.decoder.base.model.rotary_emb.learned_coefficients()

    def _gelu_gate_layers(self):
        if not self.decoder.compatible_block_indices:
            return []
        from .pretrained_attention import SPORAGeluGate

        return [index for index, block in _compatible_blocks(self.decoder)
                if isinstance(block.mlp.activation, SPORAGeluGate)]


def _compatible_blocks(decoder):
    from .pretrained_attention import SPORAResidualBlock

    layers = decoder.base.model.layers
    for index in decoder.compatible_block_indices:
        block = layers[index]
        if index in decoder.layer_indices:
            block = block.layer
        yield index, block.base if isinstance(block, SPORAResidualBlock) else block

def value_position_modules(decoder):
    if not decoder.compatible_block_indices:
        return {}
    from .pretrained_attention import ValuePositionAttention
    from .research_partial_query_key import PartialQueryKeyPositionAttention

    result = {}
    for index, block in _compatible_blocks(decoder):
        attention = block.self_attn
        if isinstance(attention, PartialQueryKeyPositionAttention):
            attention = attention.parent
        if isinstance(attention, ValuePositionAttention):
            result[index] = attention
    return result


def query_key_position_modules(decoder):
    if not decoder.compatible_block_indices:
        return {}
    from .research_partial_query_key import PartialQueryKeyPositionAttention

    return {index: block.self_attn for index, block in _compatible_blocks(decoder)
            if isinstance(block.self_attn, PartialQueryKeyPositionAttention)}


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(metadata, tensors):
    digest = hashlib.sha256()
    encoded = json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    digest.update(len(encoded).to_bytes(8, 'big'))
    digest.update(encoded)
    for name, tensor in sorted(tensors.items()):
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError('Expected named tensors')
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError('Nonfinite checkpoint tensor')
        item = tensor.detach().cpu().contiguous()
        header = json.dumps([name, str(item.dtype), list(item.shape)], separators=(',', ':')).encode('utf-8')
        payload = item.reshape(-1).view(torch.uint8).numpy().tobytes()
        for part in (header, payload):
            digest.update(len(part).to_bytes(8, 'big'))
            digest.update(part)
    return digest.hexdigest()


def _validate_binding(model, base_sha, config_sha, expected_variant=None):
    binding = getattr(model, 'named_address_binding', None)
    if binding is not None:
        binding.validate(model)
        if binding.feature_model_sha256 != base_sha or binding.feature_config_sha256 != config_sha:
            raise ValueError('Individual binding does not match base files')
        if binding.feature_variant_sha256 != expected_variant:
            raise ValueError('Individual binding does not match the structural feature variant')


def _replacement_bundle(replacements):
    if not isinstance(replacements, dict):
        raise ValueError('Structural replacements must be an indexed mapping')
    if any(type(index) is not int or index < 0 for index in replacements):
        raise ValueError('Invalid structural replacement index')
    specs, tensors = [], {}
    for index, layer in sorted(replacements.items()):
        if (type(index) is not int or index < 0 or type(layer) is not SPORALayerReplacement
                or layer.layer_idx != index or not 0 < layer.max_length <= 32768):
            raise ValueError('Invalid structural replacement')
        spec = {'index': index, 'width': layer.width, 'heads': layer.block.heads,
                'max_length': layer.max_length}
        state = layer.state_dict()
        with torch.random.fork_rng(devices=[]):
            expected = SPORALayerReplacement(layer.width, layer.block.heads,
                                              layer.max_length, layer_idx=index).state_dict()
        if (set(state) != set(expected) or any(state[k].shape != v.shape or state[k].dtype != v.dtype
                                               for k, v in expected.items())):
            raise ValueError('Invalid structural replacement tensors')
        spec['sha256'] = _digest(spec, state)
        specs.append(spec)
        tensors.update({f'replacement/{index}/{name}': tensor.detach().cpu().clone() for name, tensor in state.items()})
    return specs, tensors


def _residual_bundle(residuals):
    if not isinstance(residuals, dict):
        raise ValueError('Structural residuals must be an indexed mapping')
    specs, tensors = [], {}
    if not residuals:
        return specs, tensors
    from .pretrained_attention import SPORAResidualBlock

    for index, block in sorted(residuals.items()):
        if (type(index) is not int or index < 0 or type(block) is not SPORAResidualBlock
                or type(block.rank) is not int or block.rank < 1):
            raise ValueError('Invalid structural residual')
        state = {'down': block.down.weight.detach(), 'up': block.up.weight.detach()}
        if (state['down'].shape != (block.rank, block.hidden_size)
                or state['up'].shape != (block.hidden_size, block.rank)
                or any(not value.is_floating_point() or not bool(torch.isfinite(value).all())
                       for value in state.values())):
            raise ValueError('Invalid structural residual tensors')
        spec = {'index': index, 'rank': block.rank}
        if block.memory_only:
            spec['memory_only'] = True
        spec['sha256'] = _digest(spec, state)
        specs.append(spec)
        tensors.update({f'residual/{index}/{name}': value.detach().cpu().clone()
                        for name, value in state.items()})
    return specs, tensors


def _output_adapter_bundle(adapter, expected_vocab_size):
    if adapter is None:
        return None, {}
    if not isinstance(adapter, (MemoryOutputAdapter, MemoryReplyAdapter)):
        raise ValueError('Expected a supported memory output adapter')
    config = adapter.config_dict()
    if config['vocab_size'] != expected_vocab_size:
        raise ValueError('Memory output vocabulary does not match base configuration')
    state = adapter.state_dict()
    if not state or any(not bool(t.isfinite().all()) for t in state.values()):
        raise ValueError('Nonfinite memory output adapter')
    return config, {'output/' + name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def replacement_feature_variant_sha256(model_dir, replacements):
    """Fingerprint supplied replacement tensors; caller owns the loaded base's provenance."""
    if not replacements:
        raise ValueError('At least one structural replacement is required')
    specs, _ = _replacement_bundle(replacements)
    return structural_feature_sha256(_file_sha256(Path(model_dir)/'model.safetensors'),
                                     feature_files_sha256(model_dir),
                                     {x['index']: x['sha256'] for x in specs},
                                     {x['index']: x['max_length'] for x in specs})


def _validate_base_config(directory, hidden_width, layers):
    with (Path(directory) / 'config.json').open(encoding='utf-8') as stream:
        config = json.load(stream)
    count = config.get('num_hidden_layers')
    if (type(config.get('hidden_size')) is not int or config['hidden_size'] != hidden_width
            or type(count) is not int or count <= 0 or not isinstance(layers, (tuple, list))
            or len(layers) != 2 or any(type(i) is not int or not 0 <= i < count for i in layers)
            or layers[0] == layers[1]):
        raise ValueError('Reader does not match base configuration')


def save_integrated(path, organism, state, reader, model_dir, layer_indices, *, replacements=None,
                    residuals=None, value_positions=None, query_key_positions=None, output_adapter=None,
                    feedforward_fractions=None, compatible_layers=None,
                    feedforward_output_layers=(), gelu_gate_layers=(), position_table_max_position=None,
                    position_coefficients=None):
    if type(reader) not in (ScaledOpenBaseMemory, NativeSPORAMemoryReader):
        raise ValueError('Expected a supported scaled SPORA memory reader')
    if (not isinstance(layer_indices, (tuple, list)) or len(layer_indices) != 2
            or any(type(i) is not int or i < 0 for i in layer_indices) or layer_indices[0] == layer_indices[1]):
        raise ValueError('Expected two distinct nonnegative layer indices')
    if reader.memory_width != organism.config.memory_width:
        raise ValueError('Reader and individual memory widths differ')
    directory = Path(model_dir)
    protected = {(directory / name).resolve() for name in
                 ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json')}
    if Path(path).resolve() in protected:
        raise ValueError('Checkpoint destination overlaps a protected base file')
    _validate_base_config(directory, reader.hidden_width, layer_indices)
    base_sha = _file_sha256(directory / 'model.safetensors')
    config_sha = feature_files_sha256(directory)
    replacements = {} if replacements is None else replacements
    specs, replacement_tensors = _replacement_bundle(replacements)
    residuals = {} if residuals is None else residuals
    residual_specs, residual_tensors = _residual_bundle(residuals)
    with (directory/'config.json').open(encoding='utf-8') as stream:
        base_config = json.load(stream)
    replacement_indices = {spec['index'] for spec in specs}
    residual_indices = {spec['index'] for spec in residual_specs}
    if any(spec['index'] >= base_config['num_hidden_layers'] or spec['index'] in layer_indices
           or spec['width'] != base_config['hidden_size'] or spec['heads'] != base_config.get('num_attention_heads')
           for spec in specs):
        raise ValueError('Structural replacement does not match base configuration')
    if any(spec['index'] >= base_config['num_hidden_layers']
           or spec['index'] in replacement_indices for spec in residual_specs):
        raise ValueError('Structural residual does not match base configuration')
    if compatible_layers is None:
        compatible_layers = ()
    if (not isinstance(compatible_layers, (tuple, list, set))
            or any(type(index) is not int or not 0 <= index < base_config['num_hidden_layers']
                   for index in compatible_layers)):
        raise ValueError('Invalid compatible Qwen block indices')
    compatible_layers = tuple(sorted(set(compatible_layers)))
    if set(compatible_layers) & replacement_indices:
        raise ValueError('A layer cannot be both compatible and fully replaced')
    if not residual_indices <= set(compatible_layers):
        raise ValueError('Structural residuals require compatible Qwen blocks')
    _validate_gelu_gate_layers(gelu_gate_layers, compatible_layers)
    if (position_table_max_position is not None
            and (type(position_table_max_position) is not int
                 or not 0 < position_table_max_position <= base_config.get('max_position_embeddings', 0))):
        raise ValueError('Invalid position table length')
    if position_coefficients is not None:
        _validate_position_coefficients(position_coefficients, position_table_max_position, base_config)
    value_specs, value_tensors = value_position_bundle(
        value_positions, compatible_layers, base_config, position_table_max_position,
    )
    query_key_specs, query_key_tensors = query_key_position_bundle(query_key_positions, value_positions or {}, base_config)
    variant = (structural_feature_sha256(base_sha, config_sha, {x['index']: x['sha256'] for x in specs},
                                         {x['index']: x['max_length'] for x in specs}) if specs else None)
    _validate_binding(organism, base_sha, config_sha, variant)
    output_config, output_tensors = _output_adapter_bundle(output_adapter, base_config.get('vocab_size'))
    if feedforward_fractions is None:
        feedforward_fractions = {}
    if (not isinstance(feedforward_fractions, dict)
            or any(type(index) is not int or index not in layer_indices
                   or type(value) not in (int, float) or not math.isfinite(value)
                   or not 0. <= value <= 1.
                   for index, value in feedforward_fractions.items())):
        raise ValueError('Invalid feed-forward memory fractions')
    if (not isinstance(feedforward_output_layers, (tuple, list))
            or any(type(index) is not int or feedforward_fractions.get(index, 0.) == 0.
                   for index in feedforward_output_layers)
            or len(set(feedforward_output_layers)) != len(feedforward_output_layers)):
        raise ValueError('Output memory requires a nonzero feed-forward fraction')
    schema, individual = flatten(organism, state)
    reader_tensors = reader.state_dict()
    with torch.random.fork_rng(devices=[]):
        reader_type = type(reader)
        reference = reader_type(reader.hidden_width, reader.memory_width, reader.strength).state_dict()
    if (set(reference) != set(reader_tensors)
            or any(reader_tensors[k].shape != v.shape or reader_tensors[k].dtype != v.dtype
                   for k, v in reference.items())):
        raise ValueError('Invalid reader tensor set, shape or dtype')
    tensors = {**{'individual/' + k: v.detach().cpu().clone() for k, v in individual.items()},
               **{'reader/' + k: v.detach().cpu().clone() for k, v in reader_tensors.items()},
               **replacement_tensors, **residual_tensors, **output_tensors, **value_tensors, **query_key_tensors}
    metadata = {'format': (OUTPUT_ADAPTER_FORMAT if output_config is not None
                           else STRUCTURAL_FORMAT if specs else FORMAT),
                'base_sha256': base_sha, 'feature_config_sha256': config_sha,
                'individual_schema': schema, 'reader': {'hidden_width': reader.hidden_width,
                'memory_width': reader.memory_width, 'strength': reader.strength, 'layers': list(layer_indices),
                'kind': 'native-spora-delta-site' if type(reader) is NativeSPORAMemoryReader else 'scaled-open-base'}}
    if specs:
        metadata['replacements'] = specs
        metadata['feature_variant_sha256'] = variant
    if residual_specs:
        metadata['residuals'] = residual_specs
    if compatible_layers:
        metadata['compatible_layers'] = list(compatible_layers)
    if gelu_gate_layers:
        metadata['gelu_gate_layers'] = sorted(gelu_gate_layers)
    if value_specs:
        metadata['value_position_layers'] = value_specs
    if query_key_specs:
        metadata['query_key_position_layers'] = query_key_specs
    if position_table_max_position is not None:
        metadata['position_table_max_position'] = position_table_max_position
    if position_coefficients is not None:
        tensors['position/coefficients'] = position_coefficients.detach().cpu().clone()
    if output_config is not None:
        metadata['output_adapter'] = output_config
    if feedforward_fractions:
        metadata['feedforward_fractions'] = {
            str(index): float(value) for index, value in sorted(feedforward_fractions.items())
        }
    if feedforward_output_layers:
        metadata['feedforward_output_layers'] = sorted(feedforward_output_layers)
    integrity = _digest(metadata, tensors)
    atomic_save({'metadata': metadata, 'tensors': tensors, 'integrity_sha256': integrity}, path)


def _validate_gelu_gate_layers(indices, compatible):
    if (not isinstance(indices, (list, tuple))
            or any(type(index) is not int or index not in compatible for index in indices)
            or len(indices) != len(set(indices))):
        raise ValueError('GELU activation requires unique compatible decoder layers')


def _validate_position_coefficients(tensor, length, config):
    heads = config.get('num_attention_heads')
    width = config.get('head_dim')
    if width is None and type(heads) is int and heads > 0:
        width = config['hidden_size'] // heads
    if (length is None or type(width) is not int or width < 2 or width % 2
            or not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
            or tensor.shape != (length, width) or not bool(torch.isfinite(tensor).all())):
        raise ValueError('Invalid learned position coefficients')


def load_integrated(path, base, model_dir, *, individual_device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {'metadata', 'tensors', 'integrity_sha256'}:
        raise ValueError('Invalid integrated checkpoint')
    metadata, tensors = payload['metadata'], payload['tensors']
    if (not isinstance(metadata, dict) or metadata.get('format') not in
            (FORMAT, LEGACY_FORMAT, STRUCTURAL_FORMAT, OUTPUT_ADAPTER_FORMAT)
            or not isinstance(tensors, dict)):
        raise ValueError('Unsupported integrated checkpoint')
    if _digest(metadata, tensors) != payload['integrity_sha256']:
        raise ValueError('Integrated checkpoint integrity mismatch')
    directory = Path(model_dir)
    base_sha, config_sha = _file_sha256(directory / 'model.safetensors'), feature_files_sha256(directory)
    with (directory/'config.json').open(encoding='utf-8') as stream:
        base_config = json.load(stream)
    if metadata.get('base_sha256') != base_sha or metadata.get('feature_config_sha256') != config_sha:
        raise ValueError('Base weight or feature configuration mismatch')
    config = metadata.get('reader')
    old_config = {'hidden_width', 'memory_width', 'strength', 'layers'}
    new_config = old_config | {'kind'}
    if not isinstance(config, dict) or set(config) not in (old_config, new_config):
        raise ValueError('Invalid reader configuration')
    _validate_base_config(directory, config['hidden_width'], config['layers'])
    if getattr(getattr(base, 'config', None), 'hidden_size', None) != config['hidden_width']:
        raise ValueError('Base hidden width mismatch')
    individual_tensors = {k[len('individual/'):]: v for k, v in tensors.items() if k.startswith('individual/')}
    reader_tensors = {k[len('reader/'):]: v for k, v in tensors.items() if k.startswith('reader/')}
    replacement_tensors = {k: v for k, v in tensors.items() if k.startswith('replacement/')}
    residual_tensors = {k: v for k, v in tensors.items() if k.startswith('residual/')}
    output_tensors = {k[len('output/'):]: v for k, v in tensors.items() if k.startswith('output/')}
    value_tensors = {k: v for k, v in tensors.items() if k.startswith(VALUE_POSITION_PREFIX)}
    query_key_tensors = {k: v for k, v in tensors.items() if k.startswith(QUERY_KEY_POSITION_PREFIX)}
    has_position = 'position/coefficients' in tensors
    position_coefficients = tensors.get('position/coefficients')
    if (len(individual_tensors) + len(reader_tensors) + len(replacement_tensors)
            + len(residual_tensors) + len(output_tensors) + len(value_tensors) + len(query_key_tensors)
            + has_position != len(tensors)):
        raise ValueError('Unexpected integrated checkpoint tensor')
    format_name = metadata['format']
    if format_name in (LEGACY_FORMAT, FORMAT):
        if 'replacements' in metadata or 'output_adapter' in metadata:
            raise ValueError('Legacy checkpoint contains newer-format components')
    elif format_name == STRUCTURAL_FORMAT:
        if 'replacements' not in metadata or 'output_adapter' in metadata:
            raise ValueError('Structural checkpoint requires replacements and no output adapter')
    elif format_name == OUTPUT_ADAPTER_FORMAT and 'output_adapter' not in metadata:
        raise ValueError('Output-adapter checkpoint requires an output adapter')
    layers = getattr(getattr(base, 'model', None), 'layers', None)
    if not isinstance(layers, torch.nn.ModuleList):
        raise ValueError('Expected base.model.layers as a ModuleList')
    loaded_replacements, hashes, lengths, consumed = {}, {}, {}, set()
    if 'replacements' in metadata:
        specs = metadata.get('replacements')
        if not isinstance(specs, list) or not specs:
            raise ValueError('Invalid structural replacement metadata')
        for spec in specs:
            if not isinstance(spec, dict) or set(spec) != {'index', 'width', 'heads', 'max_length', 'sha256'}:
                raise ValueError('Invalid structural replacement metadata')
            index, width, heads, length = spec['index'], spec['width'], spec['heads'], spec['max_length']
            if (type(index) is not int or index < 0 or index >= len(layers) or index in loaded_replacements
                    or index in config['layers'] or type(width) is not int or width != config['hidden_width']
                    or type(heads) is not int or heads != getattr(base.config, 'num_attention_heads', None)
                    or type(length) is not int or not 0 < length <= 32768):
                raise ValueError('Invalid structural replacement configuration')
            prefix = f'replacement/{index}/'
            state_dict = {key[len(prefix):]: value for key, value in replacement_tensors.items() if key.startswith(prefix)}
            consumed.update(key for key in replacement_tensors if key.startswith(prefix))
            identity = {k: spec[k] for k in ('index', 'width', 'heads', 'max_length')}
            if _digest(identity, state_dict) != spec['sha256']:
                raise ValueError('Structural replacement digest mismatch')
            with torch.random.fork_rng(devices=[]):
                block = SPORALayerReplacement(width, heads, length, layer_idx=index)
            expected_state = block.state_dict()
            if (set(state_dict) != set(expected_state) or any(state_dict[k].shape != value.shape
                    or state_dict[k].dtype != value.dtype for k, value in expected_state.items())):
                raise ValueError('Invalid structural replacement tensors')
            block.load_state_dict(state_dict, strict=True)
            loaded_replacements[index] = block
            hashes[index], lengths[index] = spec['sha256'], length
        if consumed != set(replacement_tensors):
            raise ValueError('Unexpected structural replacement tensor')
        variant = structural_feature_sha256(base_sha, config_sha, hashes, lengths)
        if metadata.get('feature_variant_sha256') != variant:
            raise ValueError('Structural replacement variant mismatch')
    else:
        if replacement_tensors or 'feature_variant_sha256' in metadata:
            raise ValueError('Structural replacements require a structural checkpoint')
        variant = None
    residual_specs = metadata.get('residuals', [])
    if not isinstance(residual_specs, list):
        raise ValueError('Invalid structural residual metadata')
    loaded_residuals = {}
    consumed_residuals = set()
    for spec in residual_specs:
        fields = {'index', 'rank', 'sha256'}
        if not isinstance(spec, dict) or set(spec) not in (fields, fields | {'memory_only'}):
            raise ValueError('Invalid structural residual metadata')
        index, rank = spec['index'], spec['rank']
        if (type(index) is not int or index < 0 or index >= len(layers)
                or index in loaded_replacements or type(rank) is not int or not 1 <= rank <= base.config.hidden_size):
            raise ValueError('Invalid structural residual configuration')
        prefix = f'residual/{index}/'
        state = {key[len(prefix):]: value for key, value in residual_tensors.items() if key.startswith(prefix)}
        consumed_residuals.update(key for key in residual_tensors if key.startswith(prefix))
        if set(state) != {'down', 'up'}:
            raise ValueError('Invalid structural residual tensor set')
        identity = {'index': index, 'rank': rank}
        if 'memory_only' in spec:
            if spec['memory_only'] is not True:
                raise ValueError('Invalid residual memory activation mode')
            identity['memory_only'] = True
        if _digest(identity, state) != spec['sha256']:
            raise ValueError('Structural residual digest mismatch')
        if (state['down'].shape != (rank, base.config.hidden_size)
                or state['up'].shape != (base.config.hidden_size, rank)
                or any(not value.is_floating_point() or not bool(torch.isfinite(value).all())
                       for value in state.values())):
            raise ValueError('Invalid structural residual tensor shapes')
        loaded_residuals[index] = (rank, state, spec.get('memory_only', False))
    if consumed_residuals != set(residual_tensors):
        raise ValueError('Unexpected structural residual tensor')
    compatible_layers = metadata.get('compatible_layers', [])
    if (not isinstance(compatible_layers, list)
            or any(type(index) is not int or not 0 <= index < len(layers)
                   for index in compatible_layers)
            or len(set(compatible_layers)) != len(compatible_layers)):
        raise ValueError('Invalid compatible Qwen block metadata')
    if set(compatible_layers) & set(loaded_replacements):
        raise ValueError('A layer cannot be both compatible and fully replaced')
    if not set(loaded_residuals) <= set(compatible_layers):
        raise ValueError('Structural residuals require compatible Qwen blocks')
    gelu_gate_layers = metadata.get('gelu_gate_layers', [])
    _validate_gelu_gate_layers(gelu_gate_layers, compatible_layers)
    position_table_max_position = metadata.get('position_table_max_position')
    if (position_table_max_position is not None
            and (type(position_table_max_position) is not int
                 or not 0 < position_table_max_position <= base_config.get('max_position_embeddings', 0))):
        raise ValueError('Invalid position table metadata')
    if has_position:
        _validate_position_coefficients(position_coefficients, position_table_max_position, base_config)
    loaded_value_positions = decode_value_positions(
        metadata.get('value_position_layers', {}), value_tensors, compatible_layers, base_config,
        position_table_max_position,
    )
    loaded_query_key_positions = decode_query_key_positions(
        metadata.get('query_key_position_layers', {}), query_key_tensors, loaded_value_positions, base_config,
    )
    output_config = metadata.get('output_adapter')
    if output_config is None:
        if output_tensors:
            raise ValueError('Output adapter tensors require output adapter metadata')
        output_adapter = None
    else:
        if not isinstance(output_config, dict) or output_config.get('vocab_size') != base_config.get('vocab_size'):
            raise ValueError('Invalid memory output adapter configuration')
        keys = {'value_count', 'vocab_size', 'rank', 'max_scale'}
        if set(output_config) == keys:
            output_adapter = MemoryOutputAdapter(**output_config)
        elif set(output_config) == keys | {'format', 'steps'} and output_config['format'] == MemoryReplyAdapter.format:
            output_adapter = MemoryReplyAdapter(**{key: value for key, value in output_config.items() if key != 'format'})
        else:
            raise ValueError('Invalid memory output adapter configuration')
        expected_output = output_adapter.state_dict()
        if (set(expected_output) != set(output_tensors)
                or any(output_tensors[k].shape != v.shape or output_tensors[k].dtype != v.dtype
                       for k, v in expected_output.items())):
            raise ValueError('Invalid memory output adapter tensors')
        output_adapter.load_state_dict(output_tensors, strict=True)
    fractions = metadata.get('feedforward_fractions', {})
    if (not isinstance(fractions, dict)
            or any((not isinstance(key, str) or not key.isdigit()
                    or int(key) not in config['layers']
                    or type(value) not in (int, float) or not math.isfinite(value)
                    or not 0. <= value <= 1.) for key, value in fractions.items())):
        raise ValueError('Invalid feed-forward memory fraction metadata')
    feedforward_output_layers = metadata.get('feedforward_output_layers', [])
    if (not isinstance(feedforward_output_layers, list)
            or any(type(index) is not int or fractions.get(str(index), 0.) == 0.
                   for index in feedforward_output_layers)
            or len(set(feedforward_output_layers)) != len(feedforward_output_layers)):
        raise ValueError('Invalid feed-forward output layer metadata')
    organism, state = restore(metadata['individual_schema'], individual_tensors, individual_device)
    if organism.config.memory_width != config['memory_width']:
        raise ValueError('Reader and individual memory widths differ')
    if output_adapter is not None and output_adapter.value_count != organism.config.value_count:
        raise ValueError('Memory output value count does not match individual')
    _validate_binding(organism, base_sha, config_sha, variant)
    kind = config.get('kind', 'scaled-open-base')
    if kind == 'scaled-open-base':
        reader_type = ScaledOpenBaseMemory
    elif kind == 'native-spora-delta-site' and metadata['format'] in (FORMAT, STRUCTURAL_FORMAT, OUTPUT_ADAPTER_FORMAT):
        reader_type = NativeSPORAMemoryReader
    else:
        raise ValueError('Unsupported integrated reader kind')
    reader = reader_type(config['hidden_width'], config['memory_width'], config['strength'])
    expected = reader.state_dict()
    if set(expected) != set(reader_tensors) or any(reader_tensors[k].shape != v.shape or reader_tensors[k].dtype != v.dtype
                                                    for k, v in expected.items()):
        raise ValueError('Invalid reader tensor set, shape or dtype')
    reader.load_state_dict(reader_tensors, strict=True)
    try:
        device = next(base.parameters()).device
    except StopIteration:
        device = torch.device('cpu')
    reader.to(device)
    for block in loaded_replacements.values():
        block.to(device).eval()
    previous = {index: layers[index] for index in
                set(loaded_replacements) | set(config['layers']) | set(compatible_layers)}
    prior_rotary = getattr(base.model, 'rotary_emb', None)
    try:
        for index, block in loaded_replacements.items():
            layers[index] = block
        if output_adapter is not None:
            output_adapter.to(device).eval()
        integrated = IntegratedOpenBase(base, reader, config['layers'], output_adapter=output_adapter)
        if position_table_max_position is not None:
            integrated.eval()
            table = integrated.migrate_position_encoder(position_table_max_position)
        if compatible_layers:
            integrated.eval()
            integrated.migrate_pretrained_blocks(compatible_layers)
        if loaded_residuals:
            from .pretrained_attention import SPORAPretrainedBlock, SPORAResidualBlock
        for index, fraction in ((int(key), value) for key, value in fractions.items()):
            integrated.use_feedforward_memory(index, fraction=fraction,
                                              at_output=index in feedforward_output_layers)
        for index, (rank, residual_state, memory_only) in loaded_residuals.items():
            container = layers[index]
            on_memory_layer = index in integrated.layer_indices
            source = container.layer if on_memory_layer else container
            if not isinstance(source, SPORAPretrainedBlock):
                raise ValueError('Structural residual requires a Qwen-compatible block')
            context = integrated._memory_context if memory_only else None
            block = SPORAResidualBlock(source, rank=rank, memory_context=context).to(device).eval()
            with torch.no_grad():
                block.down.weight.copy_(residual_state['down'].to(device=device, dtype=block.down.weight.dtype))
                block.up.weight.copy_(residual_state['up'].to(device=device, dtype=block.up.weight.dtype))
            if on_memory_layer:
                container.layer = block
            else:
                layers[index] = block
        if has_position:
            with torch.no_grad():
                table.coefficients.weight.copy_(position_coefficients)
        if gelu_gate_layers:
            from .pretrained_attention import SPORAGeluGate

            for index, block in _compatible_blocks(integrated):
                if index in gelu_gate_layers:
                    block.mlp.activation = SPORAGeluGate().eval()
        if loaded_value_positions:
            from .pretrained_attention import ValuePositionAttention
            from .research_partial_query_key import PartialQueryKeyPositionAttention

            for index, block in _compatible_blocks(integrated):
                if index in loaded_value_positions:
                    attention = ValuePositionAttention(block.self_attn, *loaded_value_positions[index])
                    if index in loaded_query_key_positions:
                        attention = PartialQueryKeyPositionAttention(attention, *loaded_query_key_positions[index])
                    block.self_attn = attention.eval()
    except Exception:
        for index, layer in previous.items():
            layers[index] = layer
        if prior_rotary is not None:
            base.model.rotary_emb = prior_rotary
        raise
    return integrated, organism, state


def load_integrated_runtime(path, base, model_dir, *, individual_device='cpu'):
    decoder, organism, state = load_integrated(path, base, model_dir,
                                                individual_device=individual_device)
    if type(decoder.reader) is not NativeSPORAMemoryReader:
        raise ValueError('Integrated runtime requires native SPORA DeltaMemorySite reader')
    if not hasattr(organism, 'write_control') or state.internal is None:
        raise ValueError('Integrated checkpoint does not contain guarded write controls')
    return IntegratedSPORARuntime(decoder, organism, state, model_dir, path)
