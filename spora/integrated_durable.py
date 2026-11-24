"""Experimental incremental individual state bound to a structural bootstrap.

The immutable checkpoint holds the reader and the layer migration settings.
TensorStore holds the evolving neural individual and the base weights are loaded separately.
Only a single writer with explicit revision checks is supported. Bootstrap hashes
are verified on open; a cheap file-stat guard and tensor version counters detect
ordinary in-session changes but are not an authentication or tamper boundary.
"""
from copy import deepcopy
from pathlib import Path
from dataclasses import replace
from datetime import date

import torch

from .calendar_history import ObservationCalendar
from .durable import RevisionConflict, TensorStore
from .growing import capacity, grow, load, query
from .identity_fingerprints import IdentityFingerprints
from .integrated_checkpoint import (FORMAT, STRUCTURAL_FORMAT, OUTPUT_ADAPTER_FORMAT, _file_sha256, _validate_binding,
                                    load_integrated, query_key_position_modules, value_position_modules)
from .integrated_open_base import NativeSPORAMemoryReader
from .internal_memory import observe_sourced
from .named_address import address_basis_sha256
from .open_base_memory import addressed_memory_controls
from .registered_name_route import resolve_confirmed_name, verify_detected_name
from .sleep_memory import behavior_gate, indexed_behavior_gate, propose
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
    if (meta.get('format') not in (FORMAT, STRUCTURAL_FORMAT, OUTPUT_ADAPTER_FORMAT)
            or not (meta.get('replacements') or meta.get('compatible_layers'))):
        raise ValueError('A structural bootstrap with migrated decoder layers is required')
    return {'format': STORE_FORMAT, 'bootstrap_sha256': _file_sha256(path),
            'base_sha256': meta.get('base_sha256'), 'feature_config_sha256': meta.get('feature_config_sha256'),
            'feature_variant_sha256': meta.get('feature_variant_sha256')}


def _load_dialogue_decoder(bootstrap, dialogue_checkpoint, base, model_dir, individual_device, late_gate_start):
    if dialogue_checkpoint is None:
        return load_integrated(bootstrap, base, model_dir, individual_device=individual_device), None

    dialogue_checkpoint = Path(dialogue_checkpoint).resolve()
    before = _stat(dialogue_checkpoint)
    original = torch.load(bootstrap, map_location='cpu', weights_only=True)
    candidate = torch.load(dialogue_checkpoint, map_location='cpu', weights_only=True)
    if (not isinstance(candidate, dict) or not isinstance(candidate.get('metadata'), dict)
            or not isinstance(candidate.get('tensors'), dict)):
        raise ValueError('Invalid dialogue checkpoint')
    original_meta, candidate_meta = (item['metadata'].copy() for item in (original, candidate))
    original_specs = original_meta.pop('residuals', [])
    candidate_specs = candidate_meta.pop('residuals', None)
    original_gates = original_meta.pop('gelu_gate_layers', [])
    candidate_gates = candidate_meta.pop('gelu_gate_layers', [])
    if candidate_gates != original_gates:
        if (type(late_gate_start) is not int or late_gate_start < 0 or not isinstance(candidate_gates, list)
                or any(type(index) is not int for index in candidate_gates)
                or not set(original_gates) <= set(candidate_gates)):
            raise ValueError('Dialogue checkpoint changed existing feed-forward gates')
        new_gates = set(candidate_gates) - set(original_gates)
        protected_end = max(late_gate_start, max(original_meta['reader']['layers']) + 1)
        if not new_gates or min(new_gates) < protected_end:
            raise ValueError('Dialogue checkpoint changed a gate before memory feature extraction ends')
    if (original_meta != candidate_meta or not isinstance(candidate_specs, list)
            or any(not isinstance(spec, dict) or type(spec.get('index')) is not int
                   for spec in candidate_specs)):
        raise ValueError('Dialogue checkpoint changed structural metadata')

    existing = {spec['index']: spec for spec in original_specs}
    retained = {spec['index']: spec for spec in candidate_specs if spec['index'] in existing}
    added = [spec for spec in candidate_specs if spec['index'] not in existing]
    if retained != existing or len(candidate_specs) != len(existing) + len(added) or not added or any(
            spec.get('ordinary_only') is not True for spec in added):
        raise ValueError('Dialogue checkpoint may only add ordinary-generation residuals')

    original_tensors, candidate_tensors = (item['tensors'] for item in (original, candidate))
    added_keys = {f"residual/{spec['index']}/{direction}" for spec in added for direction in ('down', 'up')}
    if (set(candidate_tensors) != set(original_tensors) | added_keys
            or any(not isinstance(candidate_tensors[key], torch.Tensor)
                   or value.dtype != candidate_tensors[key].dtype or not torch.equal(value, candidate_tensors[key])
                   for key, value in original_tensors.items())):
        raise ValueError('Dialogue checkpoint changed existing tensors')
    del original, candidate, original_tensors, candidate_tensors
    loaded = load_integrated(dialogue_checkpoint, base, model_dir, individual_device=individual_device)
    if _stat(dialogue_checkpoint) != before:
        raise ValueError('Dialogue checkpoint changed during load')
    return loaded, (dialogue_checkpoint, before)


def _decoder_signature(decoder):
    def tensors(module):
        return tuple((name, id(tensor), tensor._version, tuple(tensor.shape), str(tensor.dtype), str(tensor.device))
                     for name, tensor in sorted(module.state_dict(keep_vars=True).items()))
    layers = decoder.base.model.layers
    adapter = decoder.output_adapter
    shape = tuple((i, id(layer), getattr(layer, 'max_length', None),
                   getattr(layer, 'layer_idx', None), getattr(layer, 'feedforward_fraction', None),
                   layer.training) for i, layer in enumerate(layers))
    placement = tuple((name, module.read_memory_at_output) for name, module in decoder.base.named_modules()
                      if hasattr(module, 'read_memory_at_output'))
    residual_modes = tuple((name, id(module.memory_context), module.ordinary_only)
                           for name, module in decoder.base.named_modules()
                           if hasattr(module, 'memory_only'))
    activations = tuple((name, id(module.activation)) for name, module in decoder.base.named_modules()
                        if hasattr(module, 'read_memory_at_output'))
    value_groups = tuple((index, id(module), module.group) for index, module in value_position_modules(decoder).items())
    query_key_pairs = tuple((index, id(component), component.group, component.pairs,
                             id(component._coordinates), component._coordinates._version)
                            for index, module in query_key_position_modules(decoder).items()
                            for component in module.migrated_groups())
    return (id(decoder.base), decoder.base.training, id(decoder.reader), decoder.reader.training,
            getattr(decoder.reader, 'strength', None), tuple(decoder.layer_indices), shape,
            tuple(decoder.compatible_block_indices), decoder.position_table_max_position,
            decoder.anchored_attention_layout,
            placement, residual_modes, activations, value_groups, query_key_pairs,
            tensors(decoder.base), tensors(decoder.reader),
            None if adapter is None else (id(adapter), adapter.training,
                                          tuple(sorted(adapter.config_dict().items())), tensors(adapter)))


def _require_runtime(decoder, organism, state):
    if (type(decoder.reader) is not NativeSPORAMemoryReader or not hasattr(organism, 'write_control')
            or state.internal is None):
        raise ValueError('Durable runtime requires a native reader and controlled individual')


def _bind_recurrent_layers(decoder, directories, parent, first_layer):
    if not directories:
        return {}
    from .research_recurrent_decode import load_recurrent_replacements

    directories = tuple(Path(directory).resolve() for directory in directories)
    paths = [directory / name for directory in directories for name in ('attention.pt', 'comparison.json')]
    sources = {path: _stat(path) for path in paths}
    replacements = load_recurrent_replacements(decoder, directories, _file_sha256(parent), first_layer)
    if any(_stat(path) != stamp for path, stamp in sources.items()):
        raise ValueError('A recurrent attention candidate changed during startup')
    for block, replacement in replacements.values():
        block.self_attn = replacement
    return sources


def _observation_calendar(metadata, state):
    if 'observation_calendar' not in metadata:
        return None
    if state.internal is None or state.internal.start_cursor != 0 or state.internal.event_keys is None:
        raise ValueError('A calendar requires a complete observation index')
    return ObservationCalendar.from_payload(metadata['observation_calendar'], state.cursor)


class IntegratedDurableRuntime:
    def __init__(self, decoder, organism, state, store, bootstrap, dialogue_source=None, attention_sources=None):
        _require_runtime(decoder, organism, state)
        self.decoder, self.organism, self.state, self.store = decoder, organism, state, store
        self.bootstrap = Path(bootstrap).resolve()
        self._bootstrap_stat = _stat(self.bootstrap)
        self._dialogue_source = dialogue_source
        self._attention_sources = {} if attention_sources is None else attention_sources
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
        if self._dialogue_source is not None:
            path, stat = self._dialogue_source
            try:
                unchanged = _stat(path) == stat
            except OSError as error:
                raise ValueError('Dialogue checkpoint is unavailable') from error
            if not unchanged:
                raise ValueError('Dialogue checkpoint changed during runtime')
        for path, stamp in self._attention_sources.items():
            try:
                unchanged = _stat(path) == stamp
            except OSError as error:
                raise ValueError('Recurrent attention candidate is unavailable') from error
            if not unchanged:
                raise ValueError('Recurrent attention candidate changed during runtime')
        if _decoder_signature(self.decoder) != self._signature:
            raise ValueError('Decoder changed since durable state was bound')

    @property
    def observation_calendar(self):
        return _observation_calendar(self.store.metadata, self.state)

    def _commit_state(self, candidate, *, metadata=None, hook=None):
        _observation_calendar(self.store.metadata if metadata is None else metadata, candidate)
        previous = self.store.revision
        try:
            stats = self.store.commit(self.organism, candidate,
                                      expected_revision=previous, metadata=metadata, hook=hook)
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

    def generate_ordinary(self, *args, **kwargs):
        """Run the ordinary decoder path without opening an individual snapshot."""
        if any(name in kwargs for name in ('memory_state', 'memory_addresses', 'memory_read_mask',
                                          'memory_slots', 'memory_output_logits')):
            raise ValueError('Ordinary generation does not accept memory controls')
        self._check()
        return self.decoder.generate(*args, **kwargs)

    @torch.no_grad()
    def generate_addressed_memory(self, *args, key, event_time=None, **kwargs):
        """Query the committed neural state and generate from that same selection."""
        reserved = ('memory_state', 'memory_addresses', 'memory_slots', 'memory_read_mask',
                    'current_only', 'preserve_memory_grad', 'memory_output_logits')
        if any(name in kwargs for name in reserved):
            raise ValueError('Addressed generation owns the memory evidence and decoder controls')
        self._check()
        organism, state = self.organism, self.state
        controls, indices = addressed_memory_controls(organism, state, key, event_time)
        evidence = query(organism, state, *indices)
        if evidence.shape != (1, organism.config.value_count) or not bool(torch.isfinite(evidence).all()):
            raise ValueError('Invalid addressed memory evidence')
        value = int(evidence.argmax(-1)[0])
        if value == organism.config.unknown:
            return value, None
        if self.decoder.output_adapter is None:
            raise ValueError('Addressed generation requires a memory output adapter')
        output = self.decoder.generate(*args, **kwargs, **controls, memory_state=state,
                                       memory_output_logits=evidence)
        return value, output

    def enroll_name(self, key, full_name, *, hook=None):
        """Add an explicit identity binding in its own durable transaction."""
        self._check()
        candidate = deepcopy(self.organism)
        if not hasattr(candidate, 'identity_fingerprints'):
            device = next(candidate.parameters()).device
            candidate.identity_fingerprints = IdentityFingerprints(candidate.config.key_count).to(device)
        candidate.identity_fingerprints.bind(key, full_name)
        revision = self.store.revision
        try:
            stats = self.store.commit(candidate, self.state, expected_revision=revision, hook=hook)
        finally:
            if self.store.revision == revision + 1:
                self.organism = candidate
        return stats

    def resolve_confirmed_name(self, full_name):
        self._check()
        return resolve_confirmed_name(getattr(self.organism, 'identity_fingerprints', None), full_name)

    def verify_detected_name(self, question, spans):
        self._check()
        return verify_detected_name(getattr(self.organism, 'identity_fingerprints', None), question, spans)

    def observe_sourced(self, key, value, source, quality, valid_at=None, *, observed_on=None,
                        effective_on=None, hook=None):
        """Commit a controlled observation, extending a full calendar history in the same transaction."""
        self._check()
        calendar = self.observation_calendar
        working = self.state
        metadata = None
        if calendar is None:
            if observed_on is not None or effective_on is not None:
                raise ValueError('This individual has no observation calendar')
        else:
            observed_on = date.today() if observed_on is None else observed_on
            extended = calendar.append(observed_on)
            if effective_on is not None:
                if type(effective_on) is not date or valid_at is not None:
                    raise ValueError('Provide either a calendar day or an effective observation step')
                if effective_on != observed_on:
                    valid_at = calendar.locate(effective_on)
                    if valid_at is None or calendar.days[valid_at] != effective_on:
                        raise ValueError('The earlier effective date must exist in the observation calendar')
            metadata = dict(self.store.metadata, observation_calendar=extended.payload())
            if working.cursor == capacity(self.organism, working):
                working = grow(self.organism, working)
        candidate, decision = observe_sourced(self.organism, working, key, value, source, quality, valid_at)
        stats = self._commit_state(candidate, metadata=metadata, hook=hook)
        return decision, stats

    def consolidate_history(self, page_index, *, rank=16, steps=600, method='distill', verify=None, hook=None):
        """Seal one full calendar history page after neural and optional decoder checks."""
        self._check()
        if self.observation_calendar is None:
            raise ValueError('Durable consolidation requires an indexed observation calendar')
        parent, revision = self.state, self.store.revision
        candidate, training = propose(self.organism, parent, page_index, rank=rank, steps=steps, method=method)
        general = behavior_gate(self.organism, parent, candidate, page_index, minimum=1.)
        indexed = indexed_behavior_gate(self.organism, parent, candidate, page_index)
        accepted = general['passed'] and indexed['passed'] and training['state_reduction'] >= .2
        verified = None
        if accepted and verify is not None:
            verified = verify(candidate)
            if type(verified) is not bool:
                raise ValueError('The decoder verification must return a boolean')
            accepted = verified
        result = {'accepted': accepted, 'training': training, 'gate': general, 'indexed_gate': indexed,
                  'decoder_verified': verified}
        if accepted:
            self._check()
            if self.state is not parent or self.store.revision != revision:
                raise RevisionConflict('History changed during consolidation verification')
            result['commit'] = self._commit_state(candidate, hook=hook)
        return result

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
                                  codec_config=None, replace_existing=False, retain_previous=False, hook=None):
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
                                       replace_existing=replace_existing, retain_previous=retain_previous)
        stats = self._commit_state(replace(self.state, structured_text=updated), hook=hook)
        slot = updated.relations[(entity, relation)]
        return {'slot': slot, 'record_revision': updated.records[slot].revision,
                'state_revision': updated.revision, 'stats': stats}

    def read_structured_relation(self, entity, relation, *, as_of_revision=None):
        """Read the latest retained value at or before a structured memory revision."""
        self._check()
        if (type(entity) is not int or not 0 <= entity < self.organism.config.key_count
                or not isinstance(relation, str) or not relation.strip()
                or relation != relation.strip() or len(relation) > 128):
            raise ValueError('Invalid entity or relation')
        structured = self.state.structured_text
        if as_of_revision is not None and (type(as_of_revision) is not int or as_of_revision < 0):
            raise ValueError('Structured memory revision must be a nonnegative integer')
        slot = None if structured is None else structured.relation_slot(entity, relation, as_of_revision)
        return None if slot is None else self.read_structured_text(slot)

    @torch.no_grad()
    def generate_structured_relation(self, *, entity, relation, codec, input_ids, as_of_revision=None,
                                     attention_mask=None, pad_token_id=None):
        """Generate a bounded factual reply from the selected vector packet."""
        from .memory_output import MemorySequenceLogitsProcessor

        record = self.read_structured_relation(entity, relation, as_of_revision=as_of_revision)
        if record is None:
            return None
        if (codec.eos_token_id is None or self.state.structured_text.codec_config != codec.config_dict()
                or not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1):
            raise ValueError('Structured generation requires one query and the recorded codec')
        scores = codec.fact_scores(record['packet'], record['metadata'])
        top = scores.topk(2, dim=-1)
        if (not bool((top.values[:, 0] >= .9999).all())
                or not bool((top.values[:, 0] > top.values[:, 1]).all())
                or int(top.indices[-1, 0]) != codec.eos_token_id
                or bool((top.indices[:-1, 0] == codec.eos_token_id).any())):
            raise ValueError('无法可靠恢复这条事实。')
        length = input_ids.shape[1]
        processor = MemorySequenceLogitsProcessor(scores * 1024., length)
        output = self.generate_ordinary(input_ids=input_ids, attention_mask=attention_mask, use_cache=True,
                                        do_sample=False, temperature=None, top_p=None, top_k=None,
                                        pad_token_id=pad_token_id, logits_processor=[processor],
                                        eos_token_id=codec.eos_token_id, max_new_tokens=len(scores))
        if output[0, length:].tolist() != top.indices[:, 0].tolist():
            raise ValueError('无法可靠生成这条事实。')
        return output


def create_integrated_durable(database, bootstrap, base, model_dir, *, individual_device='cpu',
                              calendar_individual=None, initial_state_factory=None, dialogue_checkpoint=None,
                              late_gate_start=None, recurrent_attention=()):
    if initial_state_factory is not None and (not callable(initial_state_factory) or calendar_individual is not None):
        raise ValueError('A fresh state factory requires a bootstrap without a separate calendar seed')
    bootstrap = Path(bootstrap).resolve()
    before = _stat(bootstrap)
    metadata = _bootstrap_metadata(bootstrap)
    previous_layers = list(base.model.layers)
    previous_rotary = getattr(base.model, 'rotary_emb', None)
    store = None
    try:
        (decoder, organism, state), dialogue_source = _load_dialogue_decoder(
            bootstrap, dialogue_checkpoint, base, model_dir, individual_device, late_gate_start)
        if _stat(bootstrap) != before:
            raise ValueError('Structural bootstrap changed during load')
        if calendar_individual is not None:
            imported, imported_state, archive = load(calendar_individual, individual_device)
            if (not isinstance(archive, dict) or archive.get('decoder_checkpoint_sha256') != metadata['bootstrap_sha256']
                    or imported.config != organism.config
                    or address_basis_sha256(imported) != address_basis_sha256(organism)):
                raise ValueError('Calendar individual is incompatible with the structural bootstrap')
            calendar = _observation_calendar(archive, imported_state)
            if calendar is None:
                raise ValueError('Calendar individual has no saved observation dates')
            _validate_binding(imported, metadata['base_sha256'], metadata['feature_config_sha256'],
                              metadata['feature_variant_sha256'])
            organism, state = imported, imported_state
            metadata['observation_calendar'] = calendar.payload()
        if initial_state_factory is not None:
            state = initial_state_factory(organism)
        _require_runtime(decoder, organism, state)
        attention_sources = _bind_recurrent_layers(decoder, recurrent_attention,
                                                   dialogue_checkpoint or bootstrap, late_gate_start or 0)
        store = TensorStore.create(database, organism, state, metadata)
        return IntegratedDurableRuntime(decoder, organism, state, store, bootstrap, dialogue_source, attention_sources)
    except BaseException:
        if store is not None:
            store.close()
        for index, layer in enumerate(previous_layers):
            base.model.layers[index] = layer
        if previous_rotary is not None:
            base.model.rotary_emb = previous_rotary
        raise


def load_integrated_durable(database, bootstrap, base, model_dir, *, individual_device='cpu', dialogue_checkpoint=None,
                            late_gate_start=None, recurrent_attention=()):
    bootstrap = Path(bootstrap).resolve()
    before = _stat(bootstrap)
    expected = _bootstrap_metadata(bootstrap)
    previous_layers = list(base.model.layers)
    previous_rotary = getattr(base.model, 'rotary_emb', None)
    store = TensorStore(database)
    try:
        organism, state, metadata, _ = store.read(device=individual_device)
        if not isinstance(metadata, dict):
            raise ValueError('Invalid durable state metadata')
        binding = {key: value for key, value in metadata.items() if key != 'observation_calendar'}
        if binding != expected:
            raise ValueError('Durable state and structural bootstrap do not match')
        _observation_calendar(metadata, state)
        (decoder, reference, _), dialogue_source = _load_dialogue_decoder(
            bootstrap, dialogue_checkpoint, base, model_dir, individual_device, late_gate_start)
        if _stat(bootstrap) != before:
            raise ValueError('Structural bootstrap changed during load')
        if organism.config != reference.config or address_basis_sha256(organism) != address_basis_sha256(reference):
            raise ValueError('Durable individual is incompatible with the structural bootstrap')
        _validate_binding(organism, expected['base_sha256'], expected['feature_config_sha256'],
                          expected['feature_variant_sha256'])
        _require_runtime(decoder, organism, state)
        attention_sources = _bind_recurrent_layers(decoder, recurrent_attention,
                                                   dialogue_checkpoint or bootstrap, late_gate_start or 0)
        return IntegratedDurableRuntime(decoder, organism, state, store, bootstrap, dialogue_source, attention_sources)
    except BaseException:
        store.close()
        for index, layer in enumerate(previous_layers):
            base.model.layers[index] = layer
        if previous_rotary is not None:
            base.model.rotary_emb = previous_rotary
        raise
