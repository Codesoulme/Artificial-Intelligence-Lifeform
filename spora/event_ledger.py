"""Append-only, hash-chained events for replayable individual state."""
from __future__ import annotations
import hashlib
import json
import sqlite3
import time
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class EventLedger:
    """Single-writer event log with idempotent append and chain verification."""

    def __init__(self, path, connection):
        self.path = Path(path)
        self.connection = connection

    @classmethod
    def create(cls, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('PRAGMA synchronous=FULL')
        connection.executescript('''
            CREATE TABLE ledger_meta (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                head_hash TEXT NOT NULL
            );
            CREATE TABLE events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                parent_hash TEXT NOT NULL,
                kind TEXT NOT NULL,
                monotonic_ns INTEGER NOT NULL,
                wall_ns INTEGER NOT NULL,
                source TEXT NOT NULL,
                confidence REAL NOT NULL,
                state_version INTEGER NOT NULL,
                idempotency_key TEXT NOT NULL UNIQUE,
                payload TEXT NOT NULL,
                event_hash TEXT NOT NULL UNIQUE
            );
            INSERT INTO ledger_meta(singleton, head_hash) VALUES(1, '');
        ''')
        connection.commit()
        return cls(path, connection)

    @classmethod
    def open(cls, path):
        path = Path(path)
        connection = sqlite3.connect(path)
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('PRAGMA synchronous=FULL')
        if connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='events'").fetchone() is None:
            connection.close()
            raise ValueError('Not an event ledger')
        return cls(path, connection)

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def append(self, kind, payload, *, event_id, source='system', confidence=1.0,
               state_version=0, idempotency_key=None, monotonic_ns=None, wall_ns=None):
        if not isinstance(kind, str) or not kind:
            raise ValueError('Event kind required')
        if not isinstance(event_id, str) or not event_id:
            raise ValueError('Event id required')
        if not isinstance(source, str) or not source:
            raise ValueError('Event source required')
        if not isinstance(state_version, int) or state_version < 0:
            raise ValueError('Nonnegative state version required')
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError('Confidence must be between zero and one')
        key = idempotency_key or event_id
        mono = time.perf_counter_ns() if monotonic_ns is None else monotonic_ns
        wall = time.time_ns() if wall_ns is None else wall_ns
        if not isinstance(mono, int) or not isinstance(wall, int):
            raise ValueError('Integer timestamps required')
        body = canonical(payload)
        self.connection.execute('BEGIN IMMEDIATE')
        try:
            existing = self.connection.execute(
                'SELECT seq,event_hash FROM events WHERE idempotency_key=?', (key,)).fetchone()
            if existing:
                self.connection.execute('COMMIT')
                return {'seq': existing[0], 'event_hash': existing[1], 'duplicate': True}
            parent = self.connection.execute('SELECT head_hash FROM ledger_meta WHERE singleton=1').fetchone()[0]
            envelope = canonical(dict(event_id=event_id, parent_hash=parent, kind=kind,
                                     monotonic_ns=mono, wall_ns=wall, source=source,
                                     confidence=float(confidence), state_version=state_version,
                                     idempotency_key=key, payload=json.loads(body)))
            event_hash = digest((parent + envelope).encode('utf-8'))
            cursor = self.connection.execute('''
                INSERT INTO events(event_id,parent_hash,kind,monotonic_ns,wall_ns,source,
                    confidence,state_version,idempotency_key,payload,event_hash)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
            ''', (event_id, parent, kind, mono, wall, source, float(confidence),
                  state_version, key, body, event_hash))
            seq = cursor.lastrowid
            self.connection.execute('UPDATE ledger_meta SET head_hash=? WHERE singleton=1', (event_hash,))
            self.connection.execute('COMMIT')
            return {'seq': seq, 'event_hash': event_hash, 'duplicate': False}
        except BaseException:
            if self.connection.in_transaction:
                self.connection.execute('ROLLBACK')
            raise

    def events(self):
        rows = self.connection.execute('''
            SELECT seq,event_id,parent_hash,kind,monotonic_ns,wall_ns,source,confidence,
                   state_version,idempotency_key,payload,event_hash
            FROM events ORDER BY seq
        ''').fetchall()
        return [dict(seq=row[0], event_id=row[1], parent_hash=row[2], kind=row[3],
                     monotonic_ns=row[4], wall_ns=row[5], source=row[6], confidence=row[7],
                     state_version=row[8], idempotency_key=row[9], payload=json.loads(row[10]),
                     event_hash=row[11]) for row in rows]

    def verify(self):
        parent = ''
        for event in self.events():
            if event['parent_hash'] != parent:
                raise ValueError('Event chain parent mismatch at seq '+str(event['seq']))
            envelope = canonical(dict(event_id=event['event_id'], parent_hash=parent,
                                     kind=event['kind'], monotonic_ns=event['monotonic_ns'],
                                     wall_ns=event['wall_ns'], source=event['source'],
                                     confidence=float(event['confidence']),
                                     state_version=event['state_version'],
                                     idempotency_key=event['idempotency_key'],
                                     payload=event['payload']))
            expected = digest((parent + envelope).encode('utf-8'))
            if event['event_hash'] != expected:
                raise ValueError('Event hash mismatch at seq '+str(event['seq']))
            parent = expected
        head = self.connection.execute('SELECT head_hash FROM ledger_meta WHERE singleton=1').fetchone()[0]
        if head != parent:
            raise ValueError('Ledger head mismatch')
        return {'events': len(self.events()), 'head_hash': head}
