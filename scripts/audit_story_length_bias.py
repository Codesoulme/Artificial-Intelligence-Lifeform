"""Quantify story length weighting using training data and coverage checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer


def audit(stream_path, tokenizer_path, checkpoints, fraction=.1):
    if not 0 < fraction < 1:
        raise ValueError('Fraction must be between zero and one')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    bos, eos = (tokenizer.token_to_id(t) for t in ('<|bos|>', '<|eos|>'))
    if bos is None or eos is None:
        raise ValueError('Missing boundary tokens')
    stream = np.memmap(stream_path, dtype=np.uint16, mode='r')
    starts = np.flatnonzero(stream == bos)
    if not len(starts) or starts[0] != 0 or stream[-1] != eos or np.count_nonzero(stream == eos) != len(starts) or np.any(stream[starts[1:] - 1] != eos):
        raise ValueError('Malformed boundaries')
    lengths = np.diff(np.r_[starts, len(stream)]).astype(np.int64) - 1
    if np.any(lengths < 1) or int(lengths.max()) > 512:
        raise ValueError('Expected nonempty short documents')
    count = int(len(lengths) * fraction)
    if count < 1:
        raise ValueError('Top group empty')
    selected = np.argsort(lengths, kind='stable')[-count:]
    top = set(selected.tolist())
    squared = lengths * lengths
    source_sha = hashlib.file_digest(Path(stream_path).open('rb'), 'sha256').hexdigest()
    results = []
    for name in checkpoints:
        payload = torch.load(name, map_location='cpu', weights_only=True)
        records = payload['coverage']['records']
        if any(r['source'] != source_sha or not 0 <= r['document'] < len(lengths) for r in records):
            raise ValueError('Coverage belongs to a different source')
        total = sum(r['total_positions'] for r in records)
        numerator = sum(r['total_positions'] for r in records if r['document'] in top)
        if total <= 0 or total != payload['counts']['story_source_targets']:
            raise ValueError('Coverage totals do not match')
        results.append({'checkpoint': str(name), 'sha256': hashlib.file_digest(Path(name).open('rb'), 'sha256').hexdigest(),
                        'top_group_targets': numerator, 'story_targets': total, 'observed_share': numerator / total})
    return {'source_sha256': source_sha, 'documents': len(lengths), 'total_targets': int(lengths.sum()),
            'min_length': int(lengths.min()), 'max_length': int(lengths.max()),
            'top_document_fraction': fraction, 'top_documents': count,
            'top_share_of_pool_targets': float(lengths[selected].sum() / lengths.sum()),
            'legacy_expected_share_of_exposures': float(squared[selected].sum() / squared.sum()),
            'tie_rule': 'stable ascending document index, take final positions', 'checkpoints': results,
            'limitations': 'Expectation assumes whole-document supervision and ignores update-level loss normalization; actual coverage excludes replaced focus rows and prior history.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stream', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, action='append', default=[])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.stream, args.tokenizer, args.checkpoint)
    with args.output.open('x', encoding='utf8', newline='\n') as out:
        json.dump(result, out, indent=2, sort_keys=True)
        out.write('\n')
    print(json.dumps(result))
