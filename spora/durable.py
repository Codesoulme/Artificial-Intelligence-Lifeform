"""Incremental, versioned neural tensors committed atomically with the cursor.

Database rows contain weights/state, never symbolic answers or raw experiences.
Only single-writer transactions are supported; stale revisions are rejected.
"""
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import torch
from .state_codec import flatten, restore


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pack(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    if array.dtype.name not in ('float32', 'float64', 'int64', 'int32', 'bool'):
        raise ValueError('Unsupported tensor precision')
    if not bool(np.isfinite(array).all()):
        raise ValueError('Nonfinite state; transaction not committed')
    body = array.tobytes()
    dtype, shape = array.dtype.str, canonical(list(array.shape))
    digest = sha((dtype+shape).encode()+body)
    return dtype, shape, body, digest


def unpack(dtype, shape, body, digest):
    if sha((dtype+shape).encode()+body) != digest:
        raise ValueError('Tensor integrity mismatch')
    kind = np.dtype(dtype)
    dimensions = json.loads(shape)
    if kind.name not in ('float32', 'float64', 'int64', 'int32', 'bool') or not isinstance(dimensions, list):
        raise ValueError('Invalid stored dtype/shape')
    if any(type(x) is not int or x < 0 for x in dimensions):
        raise ValueError('Invalid stored extent')
    array = np.frombuffer(body, dtype=kind).reshape(dimensions).copy()
    if not bool(np.isfinite(array).all()):
        raise ValueError('Nonfinite stored tensor')
    return torch.from_numpy(array)


class RevisionConflict(RuntimeError):
    pass


class TensorStore:
    def __init__(self, path):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.connection = sqlite3.connect(self.path.as_uri()+'?mode=rw', uri=True,
                                          isolation_level=None, timeout=5)
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.execute('PRAGMA wal_autocheckpoint=256')
        self._tracked = {}
        self.revision = None
        self.metadata = None

    @classmethod
    def create(cls, path, model, state, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb'):
            pass
        store = cls(path)
        try:
            store.connection.execute('BEGIN IMMEDIATE')
            store.connection.execute('CREATE TABLE head (singleton INTEGER PRIMARY KEY CHECK(singleton=1), revision INTEGER NOT NULL, manifest TEXT NOT NULL, digest TEXT NOT NULL)')
            store.connection.execute('CREATE TABLE tensors (name TEXT PRIMARY KEY, dtype TEXT NOT NULL, shape TEXT NOT NULL, body BLOB NOT NULL, digest TEXT NOT NULL)')
            store.connection.execute('COMMIT')
            store.commit(model, state, expected_revision=-1, metadata=metadata or {})
            return store
        except BaseException:
            if store.connection.in_transaction:
                store.connection.execute('ROLLBACK')
            store.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.connection.close()

    def read(self, device='cpu'):
        self.connection.execute('BEGIN')
        try:
            revision, manifest, digest = self.connection.execute('SELECT revision,manifest,digest FROM head WHERE singleton=1').fetchone()
            if sha(manifest.encode()) != digest:
                raise ValueError('Manifest integrity mismatch')
            data = json.loads(manifest)
            rows = self.connection.execute('SELECT name,dtype,shape,body,digest FROM tensors').fetchall()
            tensors = {name: unpack(dtype, shape, body, checksum) for name, dtype, shape, body, checksum in rows}
            if data['digests'] != {row[0]: row[4] for row in rows}:
                raise ValueError('Manifest and tensors differ')
            self.connection.execute('COMMIT')
        except BaseException:
            self.connection.execute('ROLLBACK')
            raise
        model, state = restore(data['state'], tensors, device)
        _, live = flatten(model, state)
        self._tracked = {name: (tensor, tensor._version, data['digests'][name]) for name, tensor in live.items()}
        self.revision, self.metadata = revision, data['metadata']
        return model, state, self.metadata, revision

    def commit(self, model, state, expected_revision, metadata=None, hook=None):
        hook = hook or (lambda stage: None)
        if type(expected_revision) is not int:
            raise ValueError('Explicit integer parent revision required')
        if expected_revision != (self.revision if self.revision is not None else -1):
            raise RevisionConflict('Parent revision must match the snapshot read by this store')
        schema, tensors = flatten(model, state)
        if type(state.cursor) is not int or not 1 <= len(state.pages) <= 16 or not 0 <= state.cursor <= len(state.pages)*model.config.time_count:
            raise ValueError('Invalid cursor/capacity')
        staged, tracked, digests = {}, {}, {}
        for name, tensor in tensors.items():
            old = self._tracked.get(name)
            if old and old[0] is tensor and old[1] == tensor._version:
                checksum = old[2]
            else:
                record = pack(tensor)
                checksum = record[3]
                if not old or checksum != old[2]:
                    staged[name] = record
            tracked[name] = (tensor, tensor._version, checksum)
            digests[name] = checksum
        metadata = self.metadata if metadata is None else metadata
        manifest = canonical({'state': schema, 'metadata': metadata or {}, 'digests': digests})
        hook('before_begin')
        self.connection.execute('BEGIN IMMEDIATE')
        try:
            found = self.connection.execute('SELECT revision FROM head WHERE singleton=1').fetchone()
            actual = found[0] if found else -1
            if actual != expected_revision:
                raise RevisionConflict(f'Expected revision {expected_revision}, found {actual}')
            hook('after_begin')
            for index, (name, record) in enumerate(staged.items()):
                self.connection.execute('INSERT OR REPLACE INTO tensors VALUES(?,?,?,?,?)', (name, *record))
                if index == 0:
                    hook('after_first_tensor')
            for (name,) in self.connection.execute('SELECT name FROM tensors').fetchall():
                if name not in tensors:
                    self.connection.execute('DELETE FROM tensors WHERE name=?', (name,))
            hook('after_tensors')
            self.connection.execute('INSERT OR REPLACE INTO head VALUES(1,?,?,?)',
                                    (actual+1, manifest, sha(manifest.encode())))
            hook('after_manifest')
            hook('before_commit')
            self.connection.execute('COMMIT')
        except BaseException:
            if self.connection.in_transaction:
                self.connection.execute('ROLLBACK')
            raise
        self._tracked = tracked
        self.revision, self.metadata = actual+1, metadata or {}
        hook('after_commit')
        return {'revision': self.revision, 'cursor': state.cursor, 'changed_tensors': len(staged),
                'tensor_bytes_written': sum(len(record[2]) for record in staged.values()),
                'manifest_bytes': len(manifest.encode())}

    def backup(self, destination):
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb'):
            pass
        target = sqlite3.connect(path)
        try:
            self.connection.backup(target)
        finally:
            target.close()

    def checkpoint(self):
        return self.connection.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()

    def compact(self):
        """Explicit maintenance; freed SQLite pages otherwise remain reusable."""
        self.connection.execute('VACUUM')
        return self.checkpoint()


def adopt(source, destination):
    from .growing import load
    model, state, metadata = load(source)
    metadata = dict(metadata, durable_parent_sha256=sha(Path(source).read_bytes()))
    return TensorStore.create(destination, model, state, metadata)
