"""CPU-only accounting of observed dialogue targets, without retaining dialogue.

Register each record with its immutable source SHA-256, pair_id and full mask.
Positions are indices into the original mask; position zero cannot be a target.
Only call add for presentations that actually occurred. A snapshot checksum
detects accidental changes, but is not authentication against a malicious editor.
"""
import hashlib
import json
import re


def _digest(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def _identity(source, pair_id):
    if not isinstance(source, str) or re.fullmatch(r'[0-9a-fA-F]{64}', source) is None:
        raise ValueError('source must be a SHA-256 digest')
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError('pair_id must be a nonempty string')
    return source.lower(), pair_id


class DialogueReplayCoverage:
    """Count presentations and target positions for a registered set of records.

    records contains mappings with source, pair_id and mask. Extra fields are
    ignored so callers can pass rows with tokens or group without retaining them.
    add accepts an optional nonempty subset of target indices for partial replay.
    """

    def __init__(self, records):
        self._targets = {}
        for row in records:
            if not isinstance(row, dict):
                raise ValueError('record must be a mapping')
            key = _identity(row.get('source'), row.get('pair_id'))
            mask = row.get('mask')
            if (key in self._targets or not isinstance(mask, list) or len(mask) < 2
                    or any(type(x) is not bool for x in mask) or mask[0] or not any(mask)):
                raise ValueError('invalid or duplicate record or mask')
            self._targets[key] = (len(mask), frozenset(i for i, flag in enumerate(mask) if flag))
        if not self._targets:
            raise ValueError('records must not be empty')
        configuration = [[source, pair_id, length, sorted(targets)]
                         for (source, pair_id), (length, targets) in sorted(self._targets.items())]
        self._fingerprint = _digest(configuration)
        self._counts = {key: {} for key in self._targets}
        self._presentations = {key: 0 for key in self._targets}

    def add(self, source, pair_id, positions=None):
        key = _identity(source, pair_id)
        if key not in self._targets:
            raise ValueError('unknown record')
        valid = self._targets[key][1]
        if positions is None:
            positions = sorted(valid)
        if (not isinstance(positions, (list, tuple)) or not positions
                or any(type(p) is not int or p not in valid for p in positions)
                or len(set(positions)) != len(positions)):
            raise ValueError('positions must be distinct valid target indices')
        self._presentations[key] += 1
        for position in positions:
            counts = self._counts[key]
            counts[position] = counts.get(position, 0) + 1

    def summaries(self):
        result = {}
        for key in self._targets:
            counts = self._counts[key]
            total = sum(counts.values())
            result[key] = {'presentations': self._presentations[key],
                           'target_presentations': total,
                           'distinct_target_positions': len(counts),
                           'duplicate_target_positions': total - len(counts)}
        return result

    def to_dict(self):
        payload = {'version': 1, 'fingerprint': self._fingerprint,
                   'records': [{'source': source, 'pair_id': pair_id,
                                'presentations': self._presentations[(source, pair_id)],
                                'positions': [[p, count] for p, count in sorted(self._counts[(source, pair_id)].items())]}
                               for source, pair_id in sorted(self._targets)]}
        return {**payload, 'checksum': _digest(payload)}

    @classmethod
    def from_dict(cls, records, state):
        result = cls(records)
        if (not isinstance(state, dict) or set(state) != {'version', 'fingerprint', 'records', 'checksum'}
                or type(state['version']) is not int or state['version'] != 1
                or state['fingerprint'] != result._fingerprint or not isinstance(state['records'], list)):
            raise ValueError('invalid coverage state or configuration')
        try:
            checksum = _digest({k: state[k] for k in ('version', 'fingerprint', 'records')})
        except (TypeError, ValueError):
            raise ValueError('invalid coverage state') from None
        if state['checksum'] != checksum:
            raise ValueError('coverage checksum mismatch')
        seen = set()
        for row in state['records']:
            if not isinstance(row, dict) or set(row) != {'source', 'pair_id', 'presentations', 'positions'}:
                raise ValueError('invalid coverage record')
            key = _identity(row['source'], row['pair_id'])
            presentations, positions = row['presentations'], row['positions']
            if (key not in result._targets or key in seen or row['source'] != key[0]
                    or type(presentations) is not int or presentations < 0 or not isinstance(positions, list)):
                raise ValueError('invalid coverage identity or presentations')
            seen.add(key)
            counts = {}
            for item in positions:
                if (not isinstance(item, list) or len(item) != 2 or type(item[0]) is not int
                        or item[0] not in result._targets[key][1] or item[0] in counts
                        or type(item[1]) is not int or not 1 <= item[1] <= presentations):
                    raise ValueError('invalid position count')
                counts[item[0]] = item[1]
            if sum(counts.values()) < presentations:
                raise ValueError('inconsistent presentations')
            result._counts[key], result._presentations[key] = counts, presentations
        if seen != result._targets.keys():
            raise ValueError('missing coverage record')
        return result
