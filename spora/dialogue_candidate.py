"""Validate a reviewed matched-dialogue candidate; this does not approve training."""
import json
from pathlib import Path

from spora.source_review import file_sha256


def validate_candidate(directory, manifest_sha256, tokenizer_sha256, window_length, vocabulary):
    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    if type(window_length) is not int or window_length < 1 or type(vocabulary) is not int or vocabulary < 1:
        raise ValueError('Invalid model dimensions')
    if file_sha256(manifest_path) != manifest_sha256:
        raise ValueError('Candidate manifest changed')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('status') != 'reviewed_candidate_not_approved_for_training'
            or manifest.get('format') != 'matched-dialogue-v1'
            or manifest.get('tokenizer_sha256') != tokenizer_sha256):
        raise ValueError('Candidate format, status or tokenizer mismatch')
    names = ('context.jsonl', 'last-user.jsonl', 'lineage.jsonl')
    files = manifest['files']
    if any(file_sha256(directory / name) != files[name] for name in names):
        raise ValueError('Candidate artifact changed')
    records = []
    for name in names:
        with (directory / name).open(encoding='utf-8') as source:
            records.append([json.loads(line) for line in source])
    context, last_user, lineage = records
    if not context or len(context) != len(last_user) or len(context) != len(lineage):
        raise ValueError('Candidate row counts differ')
    pairs, groups, targets = set(), set(), 0
    for full, last, origin in zip(context, last_user, lineage):
        pair, group = full['pair_id'], full['group']
        if (not isinstance(pair, str) or not pair or pair in pairs or not isinstance(group, str)
                or not group or group in groups or any(r.get('pair_id') != pair or r.get('group') != group
                                                       for r in (last, origin))):
            raise ValueError('Candidate identities differ or repeat')
        pairs.add(pair); groups.add(group)
        selected = []
        for row in (full, last):
            tokens, mask = row['tokens'], row['mask']
            if (not isinstance(tokens, list) or not isinstance(mask, list) or len(tokens) != len(mask)
                    or not 2 <= len(tokens) <= window_length + 1
                    or any(type(t) is not int or not 0 <= t < vocabulary for t in tokens)
                    or any(type(m) is not bool for m in mask) or mask[0]
                    or not any(mask[1:]) or mask != sorted(mask)):
                raise ValueError('Invalid dialogue tokens or mask')
            selected.append([t for t, enabled in zip(tokens, mask) if enabled])
        if selected[0] != selected[1] or len(selected[0]) != origin['target_tokens']:
            raise ValueError('Paired targets or lineage differ')
        targets += len(selected[0])
    if manifest['counts'] != {'pairs': len(pairs), 'groups': len(groups), 'target_tokens': targets}:
        raise ValueError('Candidate counts mismatch')
    if file_sha256(manifest_path) != manifest_sha256 or any(file_sha256(directory / n) != files[n] for n in names):
        raise ValueError('Candidate changed during validation')
    return {'manifest_sha256': manifest_sha256, 'tokenizer_sha256': tokenizer_sha256,
            'files': {name: files[name] for name in names},
            'pairs': len(pairs), 'groups': len(groups), 'target_tokens': targets}
