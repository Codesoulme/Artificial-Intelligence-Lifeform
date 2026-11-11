"""Experimental checkpoint joining an individual, reader and optional SPORA layers.

The base weights remain separate. Their on-disk digest and tokenizer/config
fingerprint are checked, but a caller-supplied in-memory base is not proven to
have been loaded from those files. Load it from the verified directory. The
structural format embeds replacement tensors and checks their declared layout;
it does not serialize unsupported blends or the full frozen base.
"""
import hashlib
import json
from pathlib import Path

import torch

from .integrated_open_base import (IntegratedOpenBase, NativeSPORAMemoryReader,
                                   ScaledOpenBaseMemory)
from .memory_output import MemoryOutputAdapter
from .language_training import atomic_save
from .named_address import feature_files_sha256, structural_feature_sha256
from .structure_transfer import SPORALayerReplacement, SPORALayerBlend
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
                 memory_output=False, **kwargs):
        if 'memory_state' in kwargs or 'memory_output_logits' in kwargs:
            raise ValueError('Runtime owns memory state and output evidence')
        if type(memory_output) is not bool:
            raise ValueError('memory_output must be a boolean')
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

    def observe_sourced(self, key, value, source, quality, valid_at=None, *, checkpoint_path=None):
        """Apply one guarded SPORA write; publish the new state only after save."""
        candidate, decision = observe_spora_sourced(
            self.organism, self.state, key, value, source, quality, valid_at)
        destination = self.checkpoint_path if checkpoint_path is None else Path(checkpoint_path)
        if destination is not None:
            save_integrated(destination, self.organism, candidate, self.decoder.reader,
                            self.model_dir, self.decoder.layer_indices, replacements=self._replacements(),
                            output_adapter=self.decoder.output_adapter)
        self.state = candidate
        return decision

    def save(self, path=None):
        destination = self.checkpoint_path if path is None else Path(path)
        if destination is None:
            raise ValueError('A checkpoint destination is required')
        save_integrated(destination, self.organism, self.state, self.decoder.reader,
                        self.model_dir, self.decoder.layer_indices, replacements=self._replacements(),
                        output_adapter=self.decoder.output_adapter)
        self.checkpoint_path = destination
        return destination

    def _replacements(self):
        layers = self.decoder.base.model.layers
        if any(isinstance(layer, SPORALayerBlend) for layer in layers):
            raise ValueError('Cannot persist an unsupported blended structural layer')
        return {index: layer for index, layer in enumerate(layers) if isinstance(layer, SPORALayerReplacement)}


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


def _output_adapter_bundle(adapter, expected_vocab_size):
    if adapter is None:
        return None, {}
    if not isinstance(adapter, MemoryOutputAdapter):
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
                    output_adapter=None):
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
    with (directory/'config.json').open(encoding='utf-8') as stream:
        base_config = json.load(stream)
    if any(spec['index'] >= base_config['num_hidden_layers'] or spec['index'] in layer_indices
           or spec['width'] != base_config['hidden_size'] or spec['heads'] != base_config.get('num_attention_heads')
           for spec in specs):
        raise ValueError('Structural replacement does not match base configuration')
    variant = (structural_feature_sha256(base_sha, config_sha, {x['index']: x['sha256'] for x in specs},
                                         {x['index']: x['max_length'] for x in specs}) if specs else None)
    _validate_binding(organism, base_sha, config_sha, variant)
    output_config, output_tensors = _output_adapter_bundle(output_adapter, base_config.get('vocab_size'))
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
               **replacement_tensors, **output_tensors}
    metadata = {'format': (OUTPUT_ADAPTER_FORMAT if output_config is not None
                           else STRUCTURAL_FORMAT if specs else FORMAT),
                'base_sha256': base_sha, 'feature_config_sha256': config_sha,
                'individual_schema': schema, 'reader': {'hidden_width': reader.hidden_width,
                'memory_width': reader.memory_width, 'strength': reader.strength, 'layers': list(layer_indices),
                'kind': 'native-spora-delta-site' if type(reader) is NativeSPORAMemoryReader else 'scaled-open-base'}}
    if specs:
        metadata['replacements'] = specs
        metadata['feature_variant_sha256'] = variant
    if output_config is not None:
        metadata['output_adapter'] = output_config
    integrity = _digest(metadata, tensors)
    atomic_save({'metadata': metadata, 'tensors': tensors, 'integrity_sha256': integrity}, path)


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
    output_tensors = {k[len('output/'):]: v for k, v in tensors.items() if k.startswith('output/')}
    if len(individual_tensors) + len(reader_tensors) + len(replacement_tensors) + len(output_tensors) != len(tensors):
        raise ValueError('Unexpected integrated checkpoint tensor')
    format_name = metadata['format']
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
    output_config = metadata.get('output_adapter')
    if output_config is None:
        if output_tensors:
            raise ValueError('Output adapter tensors require output adapter metadata')
        output_adapter = None
    else:
        if (not isinstance(output_config, dict)
                or set(output_config) != {'value_count', 'vocab_size', 'rank', 'max_scale'}
                or output_config.get('vocab_size') != base_config.get('vocab_size')):
            raise ValueError('Invalid memory output adapter configuration')
        output_adapter = MemoryOutputAdapter(**output_config)
        expected_output = output_adapter.state_dict()
        if (set(expected_output) != set(output_tensors)
                or any(output_tensors[k].shape != v.shape or output_tensors[k].dtype != v.dtype
                       for k, v in expected_output.items())):
            raise ValueError('Invalid memory output adapter tensors')
        output_adapter.load_state_dict(output_tensors, strict=True)
    organism, state = restore(metadata['individual_schema'], individual_tensors, individual_device)
    if organism.config.memory_width != config['memory_width']:
        raise ValueError('Reader and individual memory widths differ')
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
    previous = {index: layers[index] for index in set(loaded_replacements) | set(config['layers'])}
    try:
        for index, block in loaded_replacements.items():
            layers[index] = block
        if output_adapter is not None:
            output_adapter.to(device).eval()
        integrated = IntegratedOpenBase(base, reader, config['layers'], output_adapter=output_adapter)
    except Exception:
        for index, layer in previous.items():
            layers[index] = layer
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
