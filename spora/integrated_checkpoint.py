"""Experimental checkpoint joining an individual and its open-base reader.

The base weights remain separate. Their on-disk digest and tokenizer/config
fingerprint are checked, but a caller-supplied in-memory base is not proven to
have been loaded from those files. Load it from the verified directory.
"""
import hashlib
import json
from pathlib import Path

import torch

from .integrated_open_base import IntegratedOpenBase, NativeSPORAMemoryReader, ScaledOpenBaseMemory
from .language_training import atomic_save
from .named_address import feature_files_sha256
from .state_codec import flatten, restore

FORMAT = 'spora-integrated-open-base-v2'
LEGACY_FORMAT = 'spora-integrated-open-base-v1'


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


def _validate_binding(model, base_sha, config_sha):
    binding = getattr(model, 'named_address_binding', None)
    if binding is not None:
        binding.validate(model)
        if binding.feature_model_sha256 != base_sha or binding.feature_config_sha256 != config_sha:
            raise ValueError('Individual binding does not match base files')


def _validate_base_config(directory, hidden_width, layers):
    with (Path(directory) / 'config.json').open(encoding='utf-8') as stream:
        config = json.load(stream)
    count = config.get('num_hidden_layers')
    if (type(config.get('hidden_size')) is not int or config['hidden_size'] != hidden_width
            or type(count) is not int or count <= 0 or not isinstance(layers, (tuple, list))
            or len(layers) != 2 or any(type(i) is not int or not 0 <= i < count for i in layers)
            or layers[0] == layers[1]):
        raise ValueError('Reader does not match base configuration')


def save_integrated(path, organism, state, reader, model_dir, layer_indices):
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
    _validate_binding(organism, base_sha, config_sha)
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
               **{'reader/' + k: v.detach().cpu().clone() for k, v in reader_tensors.items()}}
    metadata = {'format': FORMAT, 'base_sha256': base_sha, 'feature_config_sha256': config_sha,
                'individual_schema': schema, 'reader': {'hidden_width': reader.hidden_width,
                'memory_width': reader.memory_width, 'strength': reader.strength, 'layers': list(layer_indices),
                'kind': 'native-spora-delta-site' if type(reader) is NativeSPORAMemoryReader else 'scaled-open-base'}}
    integrity = _digest(metadata, tensors)
    atomic_save({'metadata': metadata, 'tensors': tensors, 'integrity_sha256': integrity}, path)


def load_integrated(path, base, model_dir, *, individual_device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {'metadata', 'tensors', 'integrity_sha256'}:
        raise ValueError('Invalid integrated checkpoint')
    metadata, tensors = payload['metadata'], payload['tensors']
    if (not isinstance(metadata, dict) or metadata.get('format') not in (FORMAT, LEGACY_FORMAT)
            or not isinstance(tensors, dict)):
        raise ValueError('Unsupported integrated checkpoint')
    if _digest(metadata, tensors) != payload['integrity_sha256']:
        raise ValueError('Integrated checkpoint integrity mismatch')
    directory = Path(model_dir)
    base_sha, config_sha = _file_sha256(directory / 'model.safetensors'), feature_files_sha256(directory)
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
    if len(individual_tensors) + len(reader_tensors) != len(tensors):
        raise ValueError('Unexpected integrated checkpoint tensor')
    organism, state = restore(metadata['individual_schema'], individual_tensors, individual_device)
    if organism.config.memory_width != config['memory_width']:
        raise ValueError('Reader and individual memory widths differ')
    _validate_binding(organism, base_sha, config_sha)
    kind = config.get('kind', 'scaled-open-base')
    if kind == 'scaled-open-base':
        reader_type = ScaledOpenBaseMemory
    elif kind == 'native-spora-delta-site' and metadata['format'] == FORMAT:
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
        reader.to(next(base.parameters()).device)
    except StopIteration:
        pass
    integrated = IntegratedOpenBase(base, reader, config['layers'])
    return integrated, organism, state
