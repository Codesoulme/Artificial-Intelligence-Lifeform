"""Experimental incremental individual state bound to a structural bootstrap.

The immutable v3/v4 checkpoint holds the reader and replacement layers; TensorStore
holds the evolving neural individual. The large DeepSeek base remains separate.
Only a single writer with explicit revision checks is supported. Bootstrap hashes
are verified on open; a cheap file-stat guard and tensor version counters detect
ordinary in-session changes but are not an authentication or tamper boundary.
"""
from pathlib import Path
from dataclasses import replace

import torch

from .durable import TensorStore
from .integrated_checkpoint import (STRUCTURAL_FORMAT, OUTPUT_ADAPTER_FORMAT, _file_sha256, _validate_binding,
                                    load_integrated)
from .integrated_open_base import NativeSPORAMemoryReader
from .internal_memory import observe_sourced
from .named_address import address_basis_sha256
from .structured_text_memory import StructuredTextMemoryState

STORE_FORMAT = 'spora-integrated-durable-v1'


def _stat(path):
    info = Path(path).stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _bootstrap_metadata(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get('metadata'), dict):
        raise ValueError('Invalid structural bootstrap')
    meta = payload['metadata']
    if meta.get('format') not in (STRUCTURAL_FORMAT, OUTPUT_ADAPTER_FORMAT) or not meta.get('replacements'):
        raise ValueError('A structural v3 or v4 bootstrap is required')
    return {'format': STORE_FORMAT, 'bootstrap_sha256': _file_sha256(path),
            'base_sha256': meta.get('base_sha256'), 'feature_config_sha256': meta.get('feature_config_sha256'),
            'feature_variant_sha256': meta.get('feature_variant_sha256')}


def _decoder_signature(decoder):
    def tensors(module):
        return tuple((name, id(tensor), tensor._version, tuple(tensor.shape), str(tensor.dtype), str(tensor.device))
                     for name, tensor in sorted(module.state_dict(keep_vars=True).items()))
    layers = decoder.base.model.layers
    adapter = decoder.output_adapter
    shape = tuple((i, id(layer), getattr(layer, 'max_length', None),
                   getattr(layer, 'layer_idx', None), layer.training) for i, layer in enumerate(layers))
    return (id(decoder.base), decoder.base.training, id(decoder.reader), decoder.reader.training,
            getattr(decoder.reader, 'strength', None), tuple(decoder.layer_indices), shape,
            tensors(decoder.base), tensors(decoder.reader),
            None if adapter is None else (id(adapter), adapter.training,
                                          tuple(sorted(adapter.config_dict().items())), tensors(adapter)))


def _require_runtime(decoder, organism, state):
    if (type(decoder.reader) is not NativeSPORAMemoryReader or not hasattr(organism, 'write_control')
            or state.internal is None):
        raise ValueError('Durable runtime requires a native reader and controlled individual')


class IntegratedDurableRuntime:
    def __init__(self, decoder, organism, state, store, bootstrap):
        _require_runtime(decoder, organism, state)
        self.decoder, self.organism, self.state, self.store = decoder, organism, state, store
        self.bootstrap = Path(bootstrap).resolve()
        self._bootstrap_stat = _stat(self.bootstrap)
        self._signature = _decoder_signature(decoder)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.store.close()

    def _check(self):
        try:
            unchanged = _stat(self.bootstrap) == self._bootstrap_stat
        except OSError as error:
            raise ValueError('Structural bootstrap is unavailable') from error
        if not unchanged:
            raise ValueError('Structural bootstrap changed during runtime')
        if _decoder_signature(self.decoder) != self._signature:
            raise ValueError('Decoder changed since durable state was bound')

    def _commit_state(self, candidate, *, hook=None):
        previous = self.store.revision
        try:
            stats = self.store.commit(self.organism, candidate,
                                      expected_revision=previous, hook=hook)
        except BaseException:
            # A hook can run after COMMIT. Keep the in-process snapshot aligned
            # with the durable revision before propagating the injected error.
            if self.store.revision == previous + 1:
                self.state = candidate
            raise
        self.state = candidate
        return stats

    def generate(self, *args, **kwargs):
        if 'memory_state' in kwargs:
            raise ValueError('Runtime owns the memory state')
        self._check()
        return self.decoder.generate(*args, memory_state=self.state, **kwargs)

    def observe_sourced(self, key, value, source, quality, valid_at=None, *, hook=None):
        self._check()
        candidate, decision = observe_sourced(self.organism, self.state, key, value, source, quality, valid_at)
        stats = self._commit_state(candidate, hook=hook)
        return decision, stats

    def register_structured_text(self, codec_config, *, hook=None):
        """Persist an empty structured-memory schema for this open-base runtime."""
        self._check()
        config = dict(codec_config) if isinstance(codec_config, dict) else None
        if config is None:
            raise ValueError('Structured text codec configuration must be a mapping')
        current = self.state.structured_text
        if current is not None:
            if current.codec_config != config:
                raise ValueError('Structured text codec configuration mismatch')
            return {'registered': False, 'revision': current.revision}
        structured = StructuredTextMemoryState(config)
        structured.validate(self.organism.config.memory_width, self.organism.config.key_count)
        candidate = replace(self.state, structured_text=structured)
        stats = self._commit_state(candidate, hook=hook)
        return {'registered': True, 'revision': structured.revision, 'stats': stats}

    def write_structured_text(self, key, packet, metadata, *, codec_config=None, hook=None):
        """Atomically append or replace one validated identity/fact record."""
        self._check()
        current = self.state.structured_text
        if current is None:
            if not isinstance(codec_config, dict):
                raise ValueError('Codec configuration is required for first structured write')
            current = StructuredTextMemoryState(codec_config)
        elif codec_config is not None and current.codec_config != dict(codec_config):
            raise ValueError('Structured text codec configuration mismatch')
        updated = current.put(key, packet, metadata, self.organism.config.memory_width,
                              self.organism.config.key_count)
        candidate = replace(self.state, structured_text=updated)
        stats = self._commit_state(candidate, hook=hook)
        return {'key': key, 'record_revision': updated.records[key].revision,
                'state_revision': updated.revision, 'stats': stats}

    def read_structured_text(self, key):
        """Read one durable structured record without exposing a mutable tensor."""
        self._check()
        structured = self.state.structured_text
        if structured is None or key not in structured.records:
            return None
        record = structured.records[key]
        return {'packet': record.packet.detach().clone(),
                'metadata': dict(record.metadata), 'revision': record.revision}

    def write_structured_relation(self, entity, relation, packet, metadata, *,
                                  codec_config=None, replace_existing=False, hook=None):
        """Commit an explicitly supplied entity/relation binding and packet together."""
        self._check()
        current = self.state.structured_text
        if current is None:
            if not isinstance(codec_config, dict):
                raise ValueError('Codec configuration is required for first structured write')
            current = StructuredTextMemoryState(codec_config)
        elif codec_config is not None and current.codec_config != dict(codec_config):
            raise ValueError('Structured text codec configuration mismatch')
        updated = current.put_relation(entity, relation, packet, metadata,
                                       self.organism.config.memory_width,
                                       self.organism.config.key_count,
                                       replace_existing=replace_existing)
        stats = self._commit_state(replace(self.state, structured_text=updated), hook=hook)
        slot = updated.relations[(entity, relation)]
        return {'slot': slot, 'record_revision': updated.records[slot].revision,
                'state_revision': updated.revision, 'stats': stats}

    def read_structured_relation(self, entity, relation):
        """Read an explicit relation; unknown bindings return no record."""
        self._check()
        if (type(entity) is not int or not 0 <= entity < self.organism.config.key_count
                or not isinstance(relation, str) or not relation.strip()
                or relation != relation.strip() or len(relation) > 128):
            raise ValueError('Invalid entity or relation')
        structured = self.state.structured_text
        slot = None if structured is None else structured.relations.get((entity, relation))
        return None if slot is None else self.read_structured_text(slot)


def create_integrated_durable(database, bootstrap, base, model_dir, *, individual_device='cpu'):
    bootstrap = Path(bootstrap).resolve()
    before = _stat(bootstrap)
    metadata = _bootstrap_metadata(bootstrap)
    previous_layers = list(base.model.layers)
    store = None
    try:
        decoder, organism, state = load_integrated(bootstrap, base, model_dir, individual_device=individual_device)
        if _stat(bootstrap) != before:
            raise ValueError('Structural bootstrap changed during load')
        _require_runtime(decoder, organism, state)
        store = TensorStore.create(database, organism, state, metadata)
        return IntegratedDurableRuntime(decoder, organism, state, store, bootstrap)
    except BaseException:
        if store is not None:
            store.close()
        for index, layer in enumerate(previous_layers):
            base.model.layers[index] = layer
        raise


def load_integrated_durable(database, bootstrap, base, model_dir, *, individual_device='cpu'):
    bootstrap = Path(bootstrap).resolve()
    before = _stat(bootstrap)
    expected = _bootstrap_metadata(bootstrap)
    previous_layers = list(base.model.layers)
    store = TensorStore(database)
    try:
        organism, state, metadata, _ = store.read(device=individual_device)
        if metadata != expected:
            raise ValueError('Durable state and structural bootstrap do not match')
        decoder, reference, _ = load_integrated(bootstrap, base, model_dir, individual_device=individual_device)
        if _stat(bootstrap) != before:
            raise ValueError('Structural bootstrap changed during load')
        if organism.config != reference.config or address_basis_sha256(organism) != address_basis_sha256(reference):
            raise ValueError('Durable individual is incompatible with the structural bootstrap')
        _validate_binding(organism, expected['base_sha256'], expected['feature_config_sha256'],
                          expected['feature_variant_sha256'])
        _require_runtime(decoder, organism, state)
        return IntegratedDurableRuntime(decoder, organism, state, store, bootstrap)
    except BaseException:
        store.close()
        for index, layer in enumerate(previous_layers):
            base.model.layers[index] = layer
        raise
