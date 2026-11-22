"""Extend a SPORA checkpoint whose learned position table ends in unused zero rows."""

import argparse
from pathlib import Path

import torch

from spora.integrated_checkpoint import _digest
from spora.language_training import atomic_save
from spora.value_position_checkpoint import VALUE_POSITION_PREFIX


def extend(source, destination, limit):
    if destination.exists() or destination.resolve() == source.resolve():
        raise FileExistsError(destination)
    original = torch.load(source, weights_only=True, map_location='cpu')
    metadata, tensors = dict(original['metadata']), dict(original['tensors'])
    if original['integrity_sha256'] != _digest(metadata, tensors):
        raise ValueError('The input checkpoint failed its integrity check')
    previous = metadata.get('position_table_max_position')
    value_layers = metadata.get('value_position_layers')
    if (type(previous) is not int or not previous < limit <= 65536 or not value_layers
            or 'position/coefficients' in tensors):
        raise ValueError('This migration expects a regenerated shared position table and learned value positions')
    for index in value_layers:
        key = f'{VALUE_POSITION_PREFIX}{index}/position'
        table = tensors[key]
        if table.ndim != 2 or table.shape[0] != previous or previous < 32 or torch.count_nonzero(table[-32:]):
            raise ValueError('Expected at least 32 unused zero rows at the end of each learned table')
        tensors[key] = torch.cat((table, table.new_zeros((limit - previous, table.shape[1]))), dim=0)
    metadata['position_table_max_position'] = limit
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_save({'metadata': metadata, 'tensors': tensors, 'integrity_sha256': _digest(metadata, tensors)},
                destination)
    return previous


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, required=True)
    args = parser.parse_args()
    previous = extend(args.checkpoint, args.output, args.limit)
    print(f'Extended positions from {previous} to {args.limit}: {args.output}')
