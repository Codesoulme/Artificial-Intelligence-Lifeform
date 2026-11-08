"""Deterministic matched-dialogue ordering; no training or file access."""
import copy
import hashlib
import json
from collections import Counter


def _hash(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


class PairedOrder:
    """One shared order for two arms. Commit only at a joint update boundary."""

    def __init__(self, context, last_user, *, seed, epochs, length):
        if not _integer(seed) or not _integer(epochs, 1) or not _integer(length, 1):
            raise ValueError('Invalid seed, epochs or length')
        self.seed, self.epochs, self.length = seed, epochs, length
        arms = []
        for source in (context, last_user):
            records = {}
            for row in source:
                if not isinstance(row, dict):
                    raise ValueError('Record must be a mapping')
                key, group, tokens, mask = (row.get(k) for k in ('pair_id', 'group', 'tokens', 'mask'))
                if not isinstance(key, str) or not key or not isinstance(group, str) or not group or key in records:
                    raise ValueError('Invalid or duplicate pair_id or group')
                if (not isinstance(tokens, list) or not isinstance(mask, list) or
                    len(tokens) != len(mask) or not 2 <= len(tokens) <= length + 1 or
                    any(not _integer(t) for t in tokens) or any(type(m) is not bool for m in mask) or
                    mask[0] or not any(mask[1:])):
                    raise ValueError('Invalid tokens, mask or record length')
                records[key] = {'pair_id': key, 'group': group, 'tokens': tokens[:], 'mask': mask[:]}
            arms.append(records)
        if not arms[0] or arms[0].keys() != arms[1].keys():
            raise ValueError('Pair sets must be nonempty and identical')
        self._arms = arms
        self._ids = sorted(arms[0])
        for key in self._ids:
            a, b = (arm[key] for arm in arms)
            if a['group'] != b['group'] or self._target(a) != self._target(b):
                raise ValueError('Pair group or supervised targets differ')
        self.fingerprint = _hash([[arm[k] for k in self._ids] for arm in arms])
        self.cursor = 0
        self._pending = None
        self._trace = []
        self._counts = Counter()
        self._order_epoch = None
        self._order = None

    @staticmethod
    def _target(row):
        return [t for t, m in zip(row['tokens'][1:], row['mask'][1:]) if m]

    def _key_at(self, position):
        epoch, offset = divmod(position, len(self._ids))
        # A specified SHA-256 ordering is independent of Python's RNG version.
        if epoch != self._order_epoch:
            self._order = sorted(self._ids, key=lambda k: (_hash(['paired-order-v1', self.seed, epoch, k]), k))
            self._order_epoch = epoch
        return self._order[offset]

    @property
    def total(self):
        return len(self._ids) * self.epochs

    def prepare(self, batch_size):
        """Return both aligned arms and events; repeated calls before commit match."""
        if not _integer(batch_size, 1):
            raise ValueError('batch_size must be positive')
        if self._pending is not None:
            if self._pending['requested'] != batch_size:
                raise ValueError('A different batch is already pending')
            return copy.deepcopy(self._pending['batch'])
        if self.cursor == self.total:
            raise StopIteration
        counts = self._counts.copy()
        events, rows = [], [[], []]
        for position in range(self.cursor, min(self.total, self.cursor + batch_size)):
            key = self._key_at(position)
            counts[key] += 1
            record = self._arms[0][key]
            events.append({'position': position, 'epoch': position // len(self._ids),
                           'pair_id': key, 'group': record['group'],
                           'exposure': counts[key], 'target_count': len(self._target(record)),
                           'target_sha256': _hash(self._target(record))})
            for arm in range(2):
                rows[arm].append(copy.deepcopy(self._arms[arm][key]))
        batch = {'events': events, 'context': rows[0], 'last_user': rows[1]}
        self._pending = {'requested': batch_size, 'batch': batch}
        return copy.deepcopy(batch)

    def commit(self):
        if self._pending is None:
            raise ValueError('No batch pending')
        events = self._pending['batch']['events']
        for event in events:
            self._counts[event['pair_id']] += 1
        self._trace.extend(copy.deepcopy(events))
        self.cursor += len(events)
        self._pending = None

    def state_dict(self):
        if self._pending is not None:
            raise ValueError('Cannot checkpoint a pending batch')
        return {'format': 'paired-order-v1', 'seed': self.seed, 'epochs': self.epochs,
                'length': self.length, 'fingerprint': self.fingerprint,
                'cursor': self.cursor, 'events': copy.deepcopy(self._trace)}

    def load_state_dict(self, state):
        if self._pending is not None:
            raise ValueError('Cannot restore over a pending batch')
        if not isinstance(state, dict) or set(state) != {'format', 'seed', 'epochs', 'length', 'fingerprint', 'cursor', 'events'}:
            raise ValueError('Invalid state')
        if (state['format'] != 'paired-order-v1' or type(state['seed']) is not int or state['seed'] != self.seed or
            type(state['epochs']) is not int or state['epochs'] != self.epochs or
            type(state['length']) is not int or state['length'] != self.length or state['fingerprint'] != self.fingerprint or
            not _integer(state['cursor']) or state['cursor'] > self.total or
            not isinstance(state['events'], list) or len(state['events']) != state['cursor']):
            raise ValueError('State configuration or counters differ')
        # Rebuild and compare every committed event before mutating this instance.
        check = PairedOrder(list(self._arms[0].values()), list(self._arms[1].values()),
                            seed=self.seed, epochs=self.epochs, length=self.length)
        if state['cursor']:
            check.prepare(state['cursor'])
            check.commit()
        if check._trace != state['events'] or any(type(e.get(k)) is not int for e in state['events'] if isinstance(e, dict) for k in ('position', 'epoch', 'exposure', 'target_count')):
            raise ValueError('Invalid exposure history')
        self.cursor, self._trace, self._counts = check.cursor, check._trace, check._counts


def padded(rows, length, pad):
    """Build lists compatible with the existing shifted next-token objective."""
    if not _integer(length, 1) or not _integer(pad) or not rows:
        raise ValueError('Invalid padding arguments')
    inputs, labels = [], []
    for row in rows:
        tokens, mask = row['tokens'], row['mask']
        if (not isinstance(tokens, list) or not isinstance(mask, list) or
            len(tokens) != len(mask) or not 2 <= len(tokens) <= length + 1 or
            any(not _integer(t) for t in tokens) or any(type(m) is not bool for m in mask) or
            mask[0] or not any(mask[1:])):
            raise ValueError('Invalid row')
        inputs.append(tokens[:-1] + [pad] * (length + 1 - len(tokens)))
        labels.append([t if m else -100 for t, m in zip(tokens[1:], mask[1:])] + [-100] * (length + 1 - len(tokens)))
    return inputs, labels
