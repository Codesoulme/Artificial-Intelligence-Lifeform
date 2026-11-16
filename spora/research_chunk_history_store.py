"""Isolated chunk/history transaction experiment; not a production write API.

Stores neural current/history matrices and bounded routing metadata together.
Does not store raw token IDs, authenticate identities, or infer addresses.
Controlled individuals are deliberately rejected: this prototype bypasses the
normal observation/provenance path and must not be applied to real individuals.
"""
import copy
import hashlib
import torch

from spora.durable import TensorStore
from spora.growing import GrowingState, new_state
from spora.research_chunked_text import encode_chunks
from spora.structured_text_memory import StructuredTextMemoryCodec

LEGACY_FORMAT = 'spora-research-chunk-history-v1'
FORMAT = 'spora-research-chunk-history-v2'


def codec_fingerprint(codec):
    digest = hashlib.sha256()
    for name in ('identity_codebook', 'fact_codebook'):
        tensor = getattr(codec, name).detach().cpu().contiguous()
        digest.update(name.encode('ascii'))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _check_model(model, state):
    if (model.config.temporal_kind != 'split' or model.config.history_writer != 'rls'
            or model.config.current_address != 'orthogonal' or state.internal is not None
            or hasattr(model, 'write_control') or state.structured_text is not None
            or len(state.pages) != 1 or not isinstance(state.pages[0], list)):
        raise ValueError('Only uncontrolled, single-page research individuals are supported')


def _validate_metadata(model, state, metadata, codec):
    if (not isinstance(metadata, dict) or metadata.get('format') not in (LEGACY_FORMAT, FORMAT)
            or metadata.get('codec') != codec.config_dict()
            or metadata.get('codec_sha256') != codec_fingerprint(codec)
            or not isinstance(metadata.get('records'), list)
            or not isinstance(metadata.get('events'), list)):
        raise ValueError('Invalid or incompatible experiment metadata')
    events, records = metadata['events'], metadata['records']
    if len(events) != state.cursor or state.cursor > model.config.time_count:
        raise ValueError('Event count and cursor differ')
    used, indexed = set(), set()
    for record in records:
        if (not isinstance(record, dict) or set(record) != {'id', 'keys', 'versions'}
                or type(record['id']) is not int or record['id'] < 0
                or not isinstance(record['keys'], list) or not record['keys']
                or not isinstance(record['versions'], list) or not record['versions']):
            raise ValueError('Invalid record metadata')
        if record['id'] in indexed:
            raise ValueError('Duplicate record')
        indexed.add(record['id'])
        keys = record['keys']
        if (any(type(k) is not int or not 0 <= k < model.config.key_count for k in keys)
                or len(set(keys)) != len(keys) or used.intersection(keys)):
            raise ValueError('Invalid or overlapping addresses')
        used.update(keys)
        for version, times in enumerate(record['versions']):
            if (not isinstance(times, list) or not 1 <= len(times) <= len(keys)
                    or (metadata['format'] == LEGACY_FORMAT and len(times) != len(keys))):
                raise ValueError('Invalid version extent')
            for index, (key, event_time) in enumerate(zip(keys, times)):
                if type(event_time) is not int or not 0 <= event_time < len(events):
                    raise ValueError('Invalid event time')
                event = events[event_time]
                if (not isinstance(event, dict) or set(event) != {'key', 'record', 'version', 'chunk', 'metadata'}
                        or (event['key'], event['record'], event['version'], event['chunk']) !=
                           (key, record['id'], version, index)):
                    raise ValueError('Inconsistent event index')
                bounds = event['metadata']
                if (not isinstance(bounds, dict) or set(bounds) != {'identity_length', 'fact_length'}
                        or type(bounds['identity_length']) is not int or type(bounds['fact_length']) is not int
                        or not 1 <= bounds['identity_length'] <= codec.identity_slots - (codec.eos_token_id is not None)
                        or not 1 <= bounds['fact_length'] <= codec.fact_slots - (codec.eos_token_id is not None)):
                    raise ValueError('Invalid packet boundaries')
    expected = sorted(t for record in records for version in record['versions'] for t in version)
    if expected != list(range(state.cursor)):
        raise ValueError('Events are missing or reused')


class ResearchChunkHistoryStore:
    def __init__(self, store, model, state, metadata, codec):
        _check_model(model, state)
        _validate_metadata(model, state, metadata, codec)
        self.store, self.model, self.state, self.metadata, self.codec = store, model, state, metadata, codec

    @classmethod
    def create(cls, path, model, codec):
        if not isinstance(codec, StructuredTextMemoryCodec) or codec.memory_width != model.config.memory_width:
            raise ValueError('Incompatible codec')
        state = new_state(model)
        _check_model(model, state)
        for index in range(len(state.current)):
            state.current[index] = torch.zeros_like(state.current[index])
            state.pages[0][index][0] = torch.zeros_like(state.pages[0][index][0])
        metadata = {'format': FORMAT, 'codec': codec.config_dict(),
                    'codec_sha256': codec_fingerprint(codec), 'records': [], 'events': []}
        store = TensorStore.create(path, model, state, metadata)
        try:
            return cls(store, model, state, metadata, codec)
        except BaseException:
            store.close()
            raise

    @classmethod
    def open(cls, path, codec, *, device='cpu'):
        store = TensorStore(path)
        try:
            model, state, metadata, _ = store.read(device=device)
            return cls(store, model, state, metadata, codec)
        except BaseException:
            store.close()
            raise

    def close(self):
        self.store.close()

    def write(self, record_id, keys, identity_ids, fact_ids, *, allow_update=False, hook=None):
        _validate_metadata(self.model, self.state, self.metadata, self.codec)
        if (type(record_id) is not int or record_id < 0 or not isinstance(keys, (list, tuple))
                or type(allow_update) is not bool):
            raise ValueError('Explicit record and address list required')
        packets = encode_chunks(self.codec, identity_ids, fact_ids, max_chunks=self.model.config.key_count)
        keys = list(keys)
        if (len(keys) != len(packets) or any(type(k) is not int or not 0 <= k < self.model.config.key_count for k in keys)
                or len(set(keys)) != len(keys) or self.state.cursor + len(keys) > self.model.config.time_count):
            raise ValueError('Invalid address or history capacity')
        metadata = copy.deepcopy(self.metadata)
        # Legacy fixed-width states remain readable; upgrade only in the same
        # transaction as a successful new version.
        metadata['format'] = FORMAT
        record = next((r for r in metadata['records'] if r['id'] == record_id), None)
        if record is None:
            if set(keys).intersection(k for r in metadata['records'] for k in r['keys']):
                raise ValueError('Address belongs to another record')
            record = {'id': record_id, 'keys': keys, 'versions': []}
            metadata['records'].append(record)
        else:
            if not allow_update:
                raise ValueError('Existing record requires explicit update intent')
            old_keys = record['keys']
            common = min(len(old_keys), len(keys))
            if old_keys[:common] != keys[:common]:
                raise ValueError('Existing record address order cannot change')
            extra = keys[len(old_keys):]
            if set(extra).intersection(k for r in metadata['records'] if r['id'] != record_id for k in r['keys']):
                raise ValueError('Address belongs to another record')
            if self.read(record_id)['identity_ids'] != self.codec._ids(identity_ids, 'Identity'):
                raise ValueError('Existing record identity mismatch')
            record['keys'] = old_keys + extra
        version = len(record['versions'])
        current = [x.detach().clone() for x in self.state.current]
        history = [[x.detach().clone() for x in site] for site in self.state.pages[0]]
        fast = [[cur] + page for cur, page in zip(current, history)]
        velocity = [[torch.zeros_like(x) for x in site] for site in fast]
        device = next(self.model.parameters()).device
        times = []
        for index, (key, chunk) in enumerate(zip(keys, packets)):
            event_time = self.state.cursor + index
            key_tensor = torch.tensor([key], dtype=torch.long, device=device)
            time_tensor = torch.full_like(key_tensor, event_time)
            # Durable TensorStore tracks tensor version counters; inference
            # tensors have none, so create normal tensors without gradients.
            with torch.no_grad():
                for mode in (0, 1):
                    mode_tensor = torch.full_like(key_tensor, mode)
                    _, bound = self.model.representation(key_tensor, mode_tensor, time_tensor)
                    for site_index, site in enumerate(self.model.memories):
                        address = self.model.memory_address(site, bound, key_tensor, mode_tensor)
                        fast[site_index], velocity[site_index] = site.write_temporal(
                            fast[site_index], velocity[site_index], address,
                            chunk.packet.reshape(1, -1).to(device), torch.ones(1, device=device),
                            mode_tensor, False)
            metadata['events'].append({'key': key, 'record': record_id, 'version': version,
                                       'chunk': index, 'metadata': dict(chunk.metadata)})
            times.append(event_time)
        record['versions'].append(times)
        candidate = GrowingState([site[0] for site in fast], [[site[1:] for site in fast]],
                                 self.state.cursor + len(keys), self.state.runtime, None, None)
        _validate_metadata(self.model, candidate, metadata, self.codec)
        previous = self.store.revision
        try:
            stats = self.store.commit(self.model, candidate, expected_revision=previous,
                                      metadata=metadata, hook=hook)
        except BaseException:
            if self.store.revision == previous + 1:
                self.state, self.metadata = candidate, metadata
            raise
        self.state, self.metadata = candidate, metadata
        return {'version': version, 'times': times, 'stats': stats}

    def read(self, record_id, version=None):
        _validate_metadata(self.model, self.state, self.metadata, self.codec)
        if type(record_id) is not int or record_id < 0:
            raise ValueError('Invalid record id')
        record = next((r for r in self.metadata['records'] if r['id'] == record_id), None)
        if record is None:
            return None
        if version is not None and (type(version) is not int or not 0 <= version < len(record['versions'])):
            raise ValueError('Unknown version')
        mode = 0 if version is None else 1
        chosen = record['versions'][-1 if version is None else version]
        device = next(self.model.parameters()).device
        identity, fact = None, []
        for key, event_time in zip(record['keys'], chosen):
            keys = torch.tensor([key], dtype=torch.long, device=device)
            modes, times = torch.full_like(keys, mode), torch.full_like(keys, event_time)
            with torch.inference_mode():
                _, bound = self.model.representation(keys, modes, times)
                vectors = [site.retrieve([self.state.current[i] if mode == 0 else self.state.pages[0][i][0]],
                                         self.model.memory_address(site, bound, keys, modes))[0]
                           for i, site in enumerate(self.model.memories)]
                decoded = self.codec.decode(torch.stack(vectors).mean(0), self.metadata['events'][event_time]['metadata'])
            ids = decoded['identity']['token_ids']
            if identity is None:
                identity = ids
            elif identity != ids:
                raise ValueError('Inconsistent identity across chunks')
            fact.extend(decoded['fact']['token_ids'])
        return {'identity_ids': identity, 'fact_ids': fact}
