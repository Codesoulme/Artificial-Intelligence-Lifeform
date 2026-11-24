"""Persist versioned fast weights and their temporal coordinates in SQLite."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import sqlite3

from .durable import pack, unpack
from .research_ttt_ntp import LowRankFastWeight


@dataclass(frozen=True, order=True)
class MemoryAddress:
    name: str
    field: str
    condition: str = ''


@dataclass(frozen=True)
class MemoryVersion:
    number: int
    effective: date
    observed: date


class VersionStore:
    def __init__(self, path, binding, *, create=False):
        path = Path(path).resolve()
        if create:
            with path.open('xb'):
                pass
        self._connection = sqlite3.connect(path.as_uri() + '?mode=rw', uri=True, isolation_level=None, timeout=5)
        connection = self._connection
        try:
            if create:
                with connection:
                    connection.execute('BEGIN IMMEDIATE')
                    connection.execute('CREATE TABLE binding (identity TEXT NOT NULL)')
                    connection.execute('INSERT INTO binding VALUES (?)', (binding,))
                    connection.execute('''CREATE TABLE versions (
                        number INTEGER PRIMARY KEY, event TEXT UNIQUE NOT NULL,
                        name TEXT NOT NULL, field TEXT NOT NULL, condition TEXT NOT NULL,
                        effective TEXT NOT NULL, observed TEXT NOT NULL,
                        k_dtype TEXT NOT NULL, k_shape TEXT NOT NULL, k_tensor BLOB NOT NULL, k_hash TEXT NOT NULL,
                        v_dtype TEXT NOT NULL, v_shape TEXT NOT NULL, v_tensor BLOB NOT NULL, v_hash TEXT NOT NULL)''')
                    connection.execute('CREATE INDEX version_address ON versions (name, field, condition, number)')
            if connection.execute('SELECT identity FROM binding').fetchall() != [(binding,)]:
                raise ValueError('Version store belongs to a different model or write configuration')
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute('PRAGMA synchronous=FULL')
        except BaseException:
            connection.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._connection.close()

    def append(self, event, address, effective, observed, state):
        if not event or not address.name or not address.field or not 1 <= len(state.keys) <= 4:
            raise ValueError('A version needs an address, event identity and bounded tensor rank')
        values = (address.name, address.field, address.condition, effective.isoformat(), observed.isoformat(),
                  *pack(state.keys), *pack(state.values))
        with self._connection as connection:
            connection.execute('BEGIN IMMEDIATE')
            previous = connection.execute('''SELECT number,name,field,condition,effective,observed,k_dtype,k_shape,k_tensor,k_hash,
                                              v_dtype,v_shape,v_tensor,v_hash FROM versions WHERE event=?''', (event,)).fetchone()
            if previous is not None:
                if previous[1:] != values:
                    raise ValueError('An existing event cannot be reused for a different write')
                return previous[0]
            return connection.execute('INSERT INTO versions VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', (event, *values)).lastrowid

    def catalog(self):
        """Return only addresses and temporal coordinates."""
        data = {}
        for index, name, field, condition, effective, observed in self._connection.execute(
                'SELECT number,name,field,condition,effective,observed FROM versions ORDER BY number'):
            address = MemoryAddress(name, field, condition)
            data.setdefault(address, []).append(MemoryVersion(index, date.fromisoformat(effective), date.fromisoformat(observed)))
        return data

    def read(self, number):
        row = self._connection.execute('''SELECT k_dtype,k_shape,k_tensor,k_hash,v_dtype,v_shape,v_tensor,v_hash
                                          FROM versions WHERE number=?''', (number,)).fetchone()
        if row is None:
            raise KeyError(number)
        return LowRankFastWeight(unpack(*row[:4]), unpack(*row[4:]))
