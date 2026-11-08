"""Finite, deterministic target-budget planning; no model or file access."""

import copy
import hashlib
import json


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


class MixedTargetBudget:
    """Plan exact source quotas from finite ordered records, one update at a time.

    Mask positions refer to tokens, with position zero always unsupervised. A
    partially consumed record retains all tokens; its unused targets resume in
    the next update that uses that source. No data is sampled or repeated.
    """

    def __init__(self, sources, plan, *, microbatch_size):
        if not isinstance(sources, dict) or not sources or not _integer(microbatch_size, 1):
            raise ValueError('Invalid sources or microbatch size')
        if any(not isinstance(k, str) or not k for k in sources):
            raise ValueError('Invalid source name')
        self._names = sorted(sources)
        self._sources = {}
        for name in self._names:
            rows = sources[name]
            if not isinstance(rows, list) or not rows:
                raise ValueError('Every source must contain records')
            ids, clean = set(), []
            for row in rows:
                if not isinstance(row, dict) or set(row) != {'id', 'tokens', 'mask'}:
                    raise ValueError('Invalid record fields')
                key, tokens, mask = row['id'], row['tokens'], row['mask']
                if not isinstance(key, str) or not key or key in ids:
                    raise ValueError('Invalid or duplicate record id')
                if (not isinstance(tokens, list) or not isinstance(mask, list)
                        or len(tokens) != len(mask) or len(tokens) < 2
                        or any(not _integer(t) for t in tokens)
                        or any(type(m) is not bool for m in mask)
                        or mask[0] or not any(mask[1:])):
                    raise ValueError('Invalid tokens or mask')
                ids.add(key)
                clean.append({'id': key, 'tokens': tokens[:], 'mask': mask[:]})
            self._sources[name] = clean
        if not isinstance(plan, list) or not plan:
            raise ValueError('Plan must contain updates')
        self._plan = []
        for quotas in plan:
            if (not isinstance(quotas, dict) or set(quotas) != set(self._names)
                    or any(not _integer(v) for v in quotas.values()) or not sum(quotas.values())):
                raise ValueError('Each update needs nonnegative integer source quotas')
            self._plan.append({name: quotas[name] for name in self._names})
        for name in self._names:
            available = sum(sum(row['mask']) for row in self._sources[name])
            required = sum(step[name] for step in self._plan)
            if required > available:
                raise ValueError('Insufficient targets for source: ' + name)
        self.microbatch_size = microbatch_size
        self.fingerprint = _digest({'sources': self._sources, 'plan': self._plan,
                                    'microbatch_size': microbatch_size})
        self._update = 0
        self._positions = {name: [0, 0] for name in self._names}
        self._totals = {name: 0 for name in self._names}
        self._pending = None

    def prepare(self):
        """Return a copy of the pending update, or raise StopIteration."""
        if self._pending is not None:
            return copy.deepcopy(self._pending['batch'])
        if self._update == len(self._plan):
            raise StopIteration
        quotas = self._plan[self._update]
        positions = copy.deepcopy(self._positions)
        pieces = []
        for name in self._names:
            remaining = quotas[name]
            while remaining:
                row_index, offset = positions[name]
                row = self._sources[name][row_index]
                indices = [i for i, enabled in enumerate(row['mask']) if enabled]
                count = min(remaining, len(indices) - offset)
                selected = indices[offset:offset + count]
                mask = [False] * len(row['mask'])
                for index in selected:
                    mask[index] = True
                pieces.append({'source': name, 'id': row['id'], 'tokens': row['tokens'][:],
                               'mask': mask, 'target_start': offset, 'target_count': count})
                remaining -= count
                offset += count
                positions[name] = ([row_index + 1, 0] if offset == len(indices)
                                   else [row_index, offset])
        denominator = sum(quotas.values())
        microbatches = []
        for i in range(0, len(pieces), self.microbatch_size):
            rows = pieces[i:i + self.microbatch_size]
            microbatches.append({'rows': rows, 'target_count': sum(r['target_count'] for r in rows),
                                 'denominator': denominator})
        batch = {'update': self._update, 'quotas': quotas.copy(), 'denominator': denominator,
                 'microbatches': microbatches}
        self._pending = {'positions': positions, 'batch': batch}
        return copy.deepcopy(batch)

    def commit(self):
        """Advance only after the caller has completed the whole update."""
        if self._pending is None:
            raise ValueError('No update pending')
        self._positions = self._pending['positions']
        for name, value in self._plan[self._update].items():
            self._totals[name] += value
        self._update += 1
        self._pending = None

    def state_dict(self):
        if self._pending is not None:
            raise ValueError('Cannot checkpoint a pending update')
        return {'format': 'mixed-target-budget-v1', 'fingerprint': self.fingerprint,
                'update': self._update, 'positions': copy.deepcopy(self._positions),
                'totals': self._totals.copy()}

    def load_state_dict(self, state):
        if self._pending is not None:
            raise ValueError('Cannot restore over a pending update')
        if (not isinstance(state, dict)
                or set(state) != {'format', 'fingerprint', 'update', 'positions', 'totals'}
                or state['format'] != 'mixed-target-budget-v1'
                or state['fingerprint'] != self.fingerprint
                or not _integer(state['update']) or state['update'] > len(self._plan)):
            raise ValueError('Invalid state or fingerprint')
        # Replay committed updates; validate before mutating this instance.
        check = MixedTargetBudget(self._sources, self._plan, microbatch_size=self.microbatch_size)
        for _ in range(state['update']):
            check.prepare()
            check.commit()
        expected = check.state_dict()
        # Equality alone would accept bool as int in Python.
        positions, totals = state['positions'], state['totals']
        if (not isinstance(positions, dict) or not isinstance(totals, dict)
                or set(positions) != set(self._names) or set(totals) != set(self._names)
                or any(not isinstance(positions[n], list) or len(positions[n]) != 2
                       or any(not _integer(v) for v in positions[n]) for n in self._names)
                or any(not _integer(totals[n]) for n in self._names)
                or state != expected):
            raise ValueError('Invalid cursor or target totals')
        self._update = check._update
        self._positions = copy.deepcopy(check._positions)
        self._totals = check._totals.copy()


def normalized_loss(loss_sum, *, target_count, denominator):
    """Scale a microbatch's SUM of target losses by the update-wide count.

    The caller must verify the summed loss uses exactly target_count labels.
    Works with Python numbers and differentiable tensor scalars.
    """
    if (not _integer(target_count, 1) or not _integer(denominator, 1)
            or target_count > denominator):
        raise ValueError('Invalid target counts')
    return loss_sum / denominator
