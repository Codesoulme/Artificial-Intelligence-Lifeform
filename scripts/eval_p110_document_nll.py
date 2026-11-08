"""Paired, read-only development NLL for the old pretraining stream."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from spora.conversation import restore_conversation
from spora.document_stream import DocumentStream
from spora.growing import load
from spora.language_training import frozen_digest


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def selected_windows(size, length, count, seed):
    if length < 1 or count < 1 or size <= length:
        raise ValueError('Invalid window length, count, or stream size')
    rng = random.Random(seed)
    return [rng.randrange(size - length) for _ in range(count)]


def segments(values, bos):
    """Yield input/target spans for a window, restarting context at each BOS.

    The first token of the window has no target. BOS is never a target;
    EOS is included. A trailing BOS has no within-window target.
    """
    cut = [0] + [i for i in range(1, len(values)) if values[i] == bos] + [len(values)]
    return [(a, b) for a, b in zip(cut, cut[1:]) if b - a >= 2]


def loss_sum(logits, labels):
    """Return summed NLL and actual unmasked target count."""
    count = int((labels != -100).sum().item())
    if not count:
        return 0.0, 0
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                           ignore_index=-100, reduction='sum')
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite NLL')
    return float(loss), count


@torch.no_grad()
def evaluate(model, state, stream, starts, length, bos, device):
    totals = {key: [0.0, 0] for key in ('document_within_window', 'stream_matched', 'stream_all')}
    boundaries = 0
    for start in starts:
        values = np.asarray(stream[start:start + length + 1], dtype=np.int64)
        if len(values) != length + 1:
            raise ValueError('Window exceeds stream')
        seq = torch.tensor(values, dtype=torch.long, device=device).unsqueeze(0)
        labels = seq[:, 1:]
        logits = model(seq[:, :-1], state)
        matched_labels = labels.masked_fill(labels == bos, -100)
        boundaries += int((labels == bos).sum().item())
        for name, target in (('stream_all', labels), ('stream_matched', matched_labels)):
            loss, count = loss_sum(logits, target)
            totals[name][0] += loss
            totals[name][1] += count
        del logits
        for a, b in segments(values, bos):
            part = seq[:, a:b]
            loss, count = loss_sum(model(part[:, :-1], state), part[:, 1:])
            totals['document_within_window'][0] += loss
            totals['document_within_window'][1] += count
    if totals['document_within_window'][1] != totals['stream_matched'][1]:
        raise AssertionError('Paired target counts differ')
    result = {}
    for name, (total, count) in totals.items():
        if not count:
            raise ValueError('No targets in evaluation')
        result[name] = {'nll': total / count, 'nll_sum': total, 'targets': count}
    result['cross_document_bos_targets'] = boundaries
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, nargs='+', help='One or more conversation checkpoints')
    p.add_argument('--plan', default='configs/p110-old-document-boundaries.json')
    p.add_argument('--windows', type=int, default=32)
    p.add_argument('--length', type=int, default=512)
    p.add_argument('--seed', type=int, default=391109)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if not 1 <= args.windows <= 1024:
        p.error('windows must be between 1 and 1024')
    plan_path = ROOT / args.plan
    plan = json.loads(plan_path.read_text(encoding='utf-8-sig'))
    data = ROOT / plan['base_data']
    manifest_path = data / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    names = ('tokenizer.json', 'pretrain-development.bin')
    for name in names:
        if sha(data / name) != manifest['files'][name]:
            raise ValueError('Development input changed: ' + name)
    if manifest.get('tokenizer_sha256') != manifest['files']['tokenizer.json']:
        raise ValueError('Tokenizer manifest mismatch')
    tokenizer = Tokenizer.from_file(str(data / 'tokenizer.json'))
    bos, eos, pad = (tokenizer.token_to_id(x) for x in ('<|bos|>', '<|eos|>', '<|pad|>'))
    if None in (bos, eos, pad) or len({bos, eos, pad}) != 3:
        raise ValueError('Invalid special tokens')
    stream = np.memmap(data / 'pretrain-development.bin', dtype=np.uint16, mode='r')
    docs = DocumentStream(stream, bos, eos, pad)  # validate the entire stream
    starts = selected_windows(len(stream), args.length, args.windows, args.seed)
    parent = ROOT / plan['parent']
    if sha(parent) != plan['parent_sha256']:
        raise ValueError('Parent checkpoint changed')
    torch.set_num_threads(2)
    core, state, _ = load(parent, args.device)
    before = frozen_digest(core, state)
    results = []
    for checkpoint_arg in args.checkpoint:
        checkpoint = ROOT / checkpoint_arg
        checkpoint_sha = sha(checkpoint)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model = restore_conversation(core, payload, plan['parent_sha256'], manifest['tokenizer_sha256'], args.device)
        model.eval()
        if args.length > model.config.context:
            raise ValueError('Window exceeds model context')
        scores = evaluate(model, state, stream, starts, args.length, bos, args.device)
        if sha(checkpoint) != checkpoint_sha or frozen_digest(core, state) != before:
            raise AssertionError('Protected model state changed')
        results.append({'checkpoint': str(checkpoint), 'sha256': checkpoint_sha,
                        'steps': payload.get('steps'), 'scores': scores})
        del model, payload
    report = {'scope': 'Development only; fixed windows sampled with replacement. NLL in nats per target.',
              'comparison': 'document_within_window resets context at BOS; stream_matched uses identical targets with cross-document context; stream_all also predicts BOS after EOS. Window starts reset context for all scores. EOS is scored; the first token in each window and BOS are excluded from paired scores.',
              'data': str(data / 'pretrain-development.bin'), 'stream_sha256': manifest['files'][names[1]],
              'tokenizer_sha256': manifest['tokenizer_sha256'], 'manifest_sha256': sha(manifest_path),
              'parent_sha256': plan['parent_sha256'], 'plan_sha256': sha(plan_path),
              'evaluator_sha256': sha(Path(__file__)), 'device': args.device,
              'seed': args.seed, 'window_length': args.length, 'windows': args.windows,
              'starts': starts, 'development_documents': len(docs.starts), 'results': results}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
