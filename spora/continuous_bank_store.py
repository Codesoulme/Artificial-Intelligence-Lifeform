"""Transactional storage for continuous-memory bank tensors."""

import json
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from .durable import RevisionConflict, canonical, pack, sha, unpack


def _tensors(payload):
    banks = {(0, field, version): bank for (field, version), bank in payload['banks'].items()}
    for index, page in enumerate(payload.get('pages', ()), start=1):
        banks.update({(index, field, 1): bank for field, bank in page.items()})
    return banks


def bank_revision(path):
    if not path.exists():
        return None
    with closing(sqlite3.connect(path.as_uri() + '?mode=rw', uri=True)) as conn:
        row = conn.execute('SELECT revision FROM bank_head WHERE id=1').fetchone()
        return row[0] if row else None


def _read_head(conn):
    row = conn.execute('SELECT revision, manifest, digest FROM bank_head WHERE id=1').fetchone()
    if row is None or sha(row[1].encode()) != row[2]:
        raise ValueError('Bank manifest is missing or damaged')
    return row[0], json.loads(row[1])


def read_banks(path, *, load_tensors=True):
    # Opening read-write allows SQLite to recover an interrupted transaction on load.
    with closing(sqlite3.connect(path.as_uri() + '?mode=rw', uri=True)) as conn, conn:
        conn.execute('BEGIN')
        revision, data = _read_head(conn)
        selection = 'dtype, shape, body, digest' if load_tensors else 'digest'
        records = conn.execute(f'SELECT page, field, version, {selection} FROM bank_tensors '
                               'ORDER BY page, field, version').fetchall()
        actual = [[*record[:3], record[-1]] for record in records]
        if data['digests'] != actual:
            raise ValueError('Bank tensors differ from the manifest')
        payload = data['metadata']
        payload['banks'] = {}
        if 'page_count' in payload:
            payload['pages'] = [{} for _ in range(payload.pop('page_count'))]
        for page, field, version, *record in records:
            bank = unpack(*record) if load_tensors else None
            if page == 0:
                payload['banks'][field, version] = bank
            elif 0 < page <= len(payload.get('pages', ())) and version == 1:
                payload['pages'][page - 1][field] = bank
            else:
                raise ValueError('Bank tensor is outside its page layout')
    return payload, revision, {tuple(row[:3]): row[3] for row in actual}


def read_bank_tensor(path, key, revision, digest):
    with closing(sqlite3.connect(path.as_uri() + '?mode=rw', uri=True)) as conn, conn:
        conn.execute('BEGIN')
        found = conn.execute('SELECT revision FROM bank_head WHERE id=1').fetchone()
        if found is None or found[0] != revision:
            raise RevisionConflict('Continuous memory changed; reload before reading')
        record = conn.execute('SELECT dtype, shape, body, digest FROM bank_tensors '
                              'WHERE page=? AND field=? AND version=?', key).fetchone()
        if record is None or record[-1] != digest:
            raise ValueError('Bank tensor differs from the session manifest')
        return unpack(*record)


def read_archived_slots(path, entity, field, revision, back):
    if type(back) is not int or not 1 <= back <= 2**63 - 1:
        raise ValueError('History offset must be a positive SQLite integer')
    with closing(sqlite3.connect(path.as_uri() + '?mode=rw', uri=True)) as conn, conn:
        conn.execute('BEGIN')
        found = conn.execute('SELECT revision FROM bank_head WHERE id=1').fetchone()
        if found is None or found[0] != revision:
            raise RevisionConflict('Continuous memory changed; reload before reading history')
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='bank_history'").fetchone() is None:
            return None
        record = conn.execute('SELECT dtype, shape, body, digest FROM bank_history WHERE entity=? AND field=? '
                              'ORDER BY replaced_at DESC LIMIT 1 OFFSET ?', (entity, field, back - 1)).fetchone()
        return unpack(*record) if record is not None else None


def read_slots_as_of(path, entity, field, revision, cutoff_us):
    with closing(sqlite3.connect(path.as_uri() + '?mode=rw', uri=True)) as conn, conn:
        conn.execute('BEGIN')
        found = conn.execute('SELECT revision FROM bank_head WHERE id=1').fetchone()
        if found is None or found[0] != revision:
            raise RevisionConflict('Continuous memory changed; reload before reading history')
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='bank_timeline'").fetchone() is None:
            return None
        record = conn.execute('SELECT dtype, shape, body, digest FROM bank_timeline '
                              'WHERE entity=? AND field=? AND recorded_us<=? '
                              'ORDER BY recorded_us DESC, revision DESC LIMIT 1', (entity, field, cutoff_us)).fetchone()
        return unpack(*record) if record is not None else None


def read_recent_snapshots(path, revision, limit):
    """Read a bounded suffix of the append-only timeline, newest per field."""
    if type(limit) is not int or not 1 <= limit <= 256:
        raise ValueError('Replay budget must be between 1 and 256 writes')
    with closing(sqlite3.connect(path.as_uri() + '?mode=rw', uri=True)) as conn, conn:
        conn.execute('BEGIN')
        active_revision, _ = _read_head(conn)
        if active_revision != revision:
            raise RevisionConflict('Continuous memory changed; reload before replay')
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='bank_timeline'").fetchone() is None:
            return []
        rows = conn.execute('SELECT entity, field, revision, dtype, shape, body, digest FROM bank_timeline '
                            'ORDER BY rowid DESC LIMIT ?', (limit,))
        latest = {}
        for entity, field, write_revision, *record in rows:
            key = entity, field
            if key not in latest:
                latest[key] = (entity, field, write_revision, unpack(*record))
    return list(latest.values())


def commit_banks(path, payload, revision, digests, *, changed=(), archive=None, recorded=None, repairs=(), hook=None):
    tensors = _tensors(payload)
    if not set(digests) <= tensors.keys():
        raise ValueError('Bank tensor removal is unsupported')
    staged = {}
    next_digests = dict(digests)
    for key in (tensors.keys() - digests.keys()) | set(changed):
        record = pack(tensors[key])
        if digests.get(key) != record[-1]:
            staged[key] = record
            next_digests[key] = record[-1]
    metadata = {key: value for key, value in payload.items() if key not in ('banks', 'pages')}
    if 'pages' in payload:
        metadata['page_count'] = len(payload['pages'])
    manifest = canonical(dict(metadata=metadata, digests=[[*key, next_digests[key]] for key in sorted(tensors)]))
    previous = None if archive is None else (archive[0], archive[1], *pack(archive[2]))
    snapshot = None if recorded is None else (recorded[0], recorded[1], *pack(recorded[2]))
    if repairs and (revision is None or not staged or previous is not None or snapshot is not None):
        raise ValueError('Repairs require an existing revision, changed tensors and no new fact')
    if revision is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent, prefix='.bank-init-') as folder:
            temporary = Path(folder) / 'banks.sqlite3'
            next_revision = _commit(temporary, manifest, staged, revision, previous, snapshot, repairs, hook)
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise RevisionConflict('Continuous memory was created by another writer') from error
    else:
        next_revision = _commit(path, manifest, staged, revision, previous, snapshot, repairs, hook)
    return next_revision, next_digests, sum(len(record[2]) for record in staged.values())


def _commit(path, manifest, staged, revision, previous, snapshot, repairs, hook):
    mode = 'rwc' if revision is None else 'rw'
    with closing(sqlite3.connect(path.as_uri() + '?mode=' + mode, uri=True, timeout=5)) as conn, conn:
        conn.execute('PRAGMA synchronous=FULL')
        conn.execute('BEGIN IMMEDIATE')
        conn.execute('CREATE TABLE IF NOT EXISTS bank_head (id INTEGER PRIMARY KEY CHECK(id=1), '
                     'revision INTEGER NOT NULL, manifest TEXT NOT NULL, digest TEXT NOT NULL)')
        conn.execute('CREATE TABLE IF NOT EXISTS bank_tensors (page INTEGER, field TEXT, version INTEGER, '
                     'dtype TEXT NOT NULL, shape TEXT NOT NULL, body BLOB NOT NULL, digest TEXT NOT NULL, '
                     'PRIMARY KEY(page, field, version))')
        found = conn.execute('SELECT revision FROM bank_head WHERE id=1').fetchone()
        if (found[0] if found else None) != revision:
            raise RevisionConflict('Continuous memory changed; reload before writing')
        conn.executemany('INSERT OR REPLACE INTO bank_tensors VALUES (?,?,?,?,?,?,?)',
                         [(*key, *record) for key, record in staged.items()])
        next_revision = 0 if revision is None else revision + 1
        if previous is not None:
            conn.execute('CREATE TABLE IF NOT EXISTS bank_history (entity INTEGER NOT NULL, field TEXT NOT NULL, '
                         'replaced_at INTEGER NOT NULL, dtype TEXT NOT NULL, shape TEXT NOT NULL, body BLOB NOT NULL, '
                         'digest TEXT NOT NULL, PRIMARY KEY(entity, field, replaced_at))')
            conn.execute('INSERT INTO bank_history VALUES (?,?,?,?,?,?,?)', (*previous[:2], next_revision, *previous[2:]))
        if snapshot is not None:
            conn.execute('CREATE TABLE IF NOT EXISTS bank_timeline (entity INTEGER NOT NULL, field TEXT NOT NULL, '
                         'recorded_us INTEGER NOT NULL, revision INTEGER NOT NULL, dtype TEXT NOT NULL, shape TEXT NOT NULL, '
                         'body BLOB NOT NULL, digest TEXT NOT NULL, PRIMARY KEY(entity, field, recorded_us, revision))')
            latest = conn.execute('SELECT MAX(recorded_us) FROM bank_timeline WHERE entity=? AND field=?', snapshot[:2]).fetchone()[0]
            recorded_us = time.time_ns() // 1000
            if latest is not None and recorded_us < latest:
                raise ValueError('System clock moved behind this record’s previous write time')
            conn.execute('INSERT INTO bank_timeline VALUES (?,?,?,?,?,?,?,?)',
                         (*snapshot[:2], recorded_us, next_revision, *snapshot[2:]))
        if repairs:
            conn.execute('CREATE TABLE IF NOT EXISTS bank_repairs (entity INTEGER NOT NULL, field TEXT NOT NULL, '
                         'revision INTEGER NOT NULL, source_revision INTEGER NOT NULL, '
                         'PRIMARY KEY(entity, field, revision))')
            for entity, field, source_revision in repairs:
                latest = conn.execute('SELECT MAX(revision) FROM bank_timeline WHERE entity=? AND field=?',
                                      (entity, field)).fetchone()[0]
                if latest is None or latest != source_revision or source_revision > revision:
                    raise ValueError('Repair source is not the newest recorded value')
                conn.execute('INSERT INTO bank_repairs VALUES (?,?,?,?)', (entity, field, next_revision, source_revision))
        conn.execute('INSERT OR REPLACE INTO bank_head VALUES (1,?,?,?)',
                     (next_revision, manifest, sha(manifest.encode())))
        if hook is not None:
            hook()
    return next_revision
