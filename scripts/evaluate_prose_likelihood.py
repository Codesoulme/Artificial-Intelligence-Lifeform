"""Read-only, fixed-window prose DEVELOPMENT likelihood, never chat acceptance."""
import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from spora.conversation import restore_conversation
from spora.growing import load
from spora.language_training import frozen_digest
from scripts.prepare_conversation_v2 import ROOT, sha


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--stream', choices=('story-development.bin', 'pretrain-development.bin'),
                        default='story-development.bin')
    parser.add_argument('--output', required=True)
    parser.add_argument('--windows', type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.windows <= 1024:
        raise ValueError('Bounded development evaluation required')
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads((ROOT / args.plan).read_text('utf-8'))
    data = ROOT / args.data
    manifest = json.loads((data / 'manifest.json').read_text('utf-8'))
    for name in ('tokenizer.json', args.stream):
        if sha(data / name) != manifest['files'][name]:
            raise ValueError('Development input changed: ' + name)
    parent = ROOT / plan['parent']
    checkpoint = ROOT / args.checkpoint
    checkpoint_hash = sha(checkpoint)
    if sha(parent) != plan['parent_sha256']:
        raise ValueError('Parent changed')
    torch.set_num_threads(2)
    organism, state, _ = load(parent, 'cuda')
    original = frozen_digest(organism, state)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = restore_conversation(organism, payload, plan['parent_sha256'],
                                 manifest['files']['tokenizer.json'], 'cuda')
    length = min(512, model.config.context)
    stream = np.memmap(data / args.stream, mode='r', dtype=np.uint16)
    if len(stream) <= length:
        raise ValueError('Insufficient development stream')
    rng = random.Random(281108)
    starts = [rng.randrange(len(stream) - length) for _ in range(args.windows)]
    total = 0.0
    targets = 0
    for start in range(0, len(starts), 4):
        values = np.stack([np.array(stream[i:i + length + 1], dtype=np.int64)
                           for i in starts[start:start + 4]])
        values = torch.tensor(values, device='cuda')
        logits = model(values[:, :-1], state)
        labels = values[:, 1:]
        total += float(F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='sum'))
        targets += labels.numel()
    if not np.isfinite(total):
        raise ValueError('Nonfinite development likelihood')
    intact = original == frozen_digest(organism, state)
    if not intact or checkpoint_hash != sha(checkpoint):
        raise AssertionError('Read-only evaluation changed protected state')
    report = dict(checkpoint_sha256=checkpoint_hash, data_manifest_sha256=sha(data / 'manifest.json'),
                  stream=args.stream, stream_sha256=manifest['files'][args.stream],
                  seed=281108, starts=starts, length=length, targets=targets, nll=total / targets,
                  old_individual_exact=intact, checkpoint_unchanged=True,
                  evaluator_sha256=sha(Path(__file__)),
                  scope='Seeded windows with replacement. Development only; likelihood is not a semantic or chat score.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'starts'}), flush=True)


if __name__ == '__main__':
    main()
