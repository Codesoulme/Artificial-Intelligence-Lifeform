"""Select a reproducible per-group subset of already aligned paired records."""
import hashlib
import json
from collections import Counter, defaultdict

from spora.paired_order import PairedOrder


def select(context, last_user, *, seed, max_per_group, length):
    if type(seed) is not int or seed < 0 or type(max_per_group) is not int or max_per_group < 1:
        raise ValueError('Invalid seed or group limit')
    context, last_user = list(context), list(last_user)
    # Validate all pairs before selecting so malformed excluded rows cannot be hidden.
    order = PairedOrder(context, last_user, seed=seed, epochs=1, length=length)
    arms = [dict((row['pair_id'], row) for row in source) for source in (context, last_user)]
    groups = defaultdict(list)
    for row in arms[0].values():
        groups[row['group']].append(row['pair_id'])
    def rank(group, pair):
        content = json.dumps(['paired-group-sample-v1', seed, group, pair], ensure_ascii=False,
                             separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(content).hexdigest(), pair
    selected = []
    for group, ids in sorted(groups.items()):
        selected.extend(sorted(ids, key=lambda pair: rank(group, pair))[:max_per_group])
    # Sort before handing data to the order module; the two arms share this order.
    selected.sort()
    counts = Counter(arms[0][pair]['group'] for pair in selected)
    return ([arms[0][pair] for pair in selected], [arms[1][pair] for pair in selected],
            {'input_fingerprint': order.fingerprint, 'seed': seed, 'max_per_group': max_per_group,
             'input_pairs': len(arms[0]), 'input_groups': len(groups), 'selected_pairs': len(selected),
             'selected_groups': len(counts), 'largest_group_selected': max(counts.values(), default=0),
             'selected_id_sha256': hashlib.sha256('\n'.join(selected).encode('utf-8')).hexdigest()})
