"""Read-only benchmark of trimming ignored trailing story positions."""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from scripts.prepare_conversation_v2 import sha
from spora.conversation import restore_conversation
from spora.document_stream import DocumentStream, trim_ignored_tail
from spora.growing import load
from spora.language_training import frozen_digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = ROOT / args.output
    if out.exists():
        raise FileExistsError(out)
    plan = json.loads((ROOT / args.plan).read_text(encoding='utf8'))
    data, parent, checkpoint = ROOT / plan['data'], ROOT / plan['parent'], ROOT / args.checkpoint
    if sha(parent) != plan['parent_sha256']:
        raise ValueError('Individual differs')
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf8'))
    for name in ('tokenizer.json', 'pretrain-train.bin'):
        if sha(data / name) != manifest['files'][name]:
            raise ValueError('Dataset differs: ' + name)
    before = sha(checkpoint)
    tokenizer = Tokenizer.from_file(str(data / 'tokenizer.json'))
    stream = np.memmap(data / 'pretrain-train.bin', mode='r', dtype=np.uint16)
    sampler = DocumentStream(stream, *(tokenizer.token_to_id(x) for x in ('<|bos|>', '<|eos|>', '<|pad|>')))
    rng = random.Random(671109)
    batches = [sampler.batch(rng, 4, 512, 'cuda') for _ in range(32)]
    torch.set_num_threads(2)
    organism, state, _ = load(parent, 'cuda')
    frozen = frozen_digest(organism, state)
    model = restore_conversation(organism, torch.load(checkpoint, map_location='cpu', weights_only=True),
                                 plan['parent_sha256'], sha(data / 'tokenizer.json'), 'cuda')
    model.train()
    x, y = batches[0]
    xt, yt = trim_ignored_tail(x, y)
    def gradients(a, b):
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(a, state).flatten(0, 1), b.flatten(), ignore_index=-100)
        loss.backward()
        return float(loss.detach()), [p.grad.detach().clone() for p in model.trainable_parameters() if p.grad is not None]
    loss_full, grad_full = gradients(x, y)
    loss_trim, grad_trim = gradients(xt, yt)
    if len(grad_full) != len(grad_trim):
        raise AssertionError('Different gradient parameter sets')
    for a, b in zip(grad_full, grad_trim):
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-6)
    def timed(crop):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for a, b in batches:
            if crop:
                a, b = trim_ignored_tail(a, b)
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(a, state).flatten(0, 1), b.flatten(), ignore_index=-100).backward()
        torch.cuda.synchronize()
        return time.perf_counter() - start
    # Both shapes were exercised above; alternate order to expose noise.
    full1, trimmed1, trimmed2, full2 = timed(False), timed(True), timed(True), timed(False)
    if sha(checkpoint) != before or frozen_digest(organism, state) != frozen:
        raise AssertionError('Protected state changed')
    widths = [trim_ignored_tail(a, b)[0].shape[1] for a, b in batches]
    report = dict(scope='32 seeded story-only microbatches; no focus replacement, optimizer or end-to-end training. Gradients checked on one batch; timing varies with hardware and workload.',
                  checkpoint_sha256=before, sampler_source_sha256=sha(ROOT / 'spora/document_stream.py'),
                  evaluator_sha256=sha(Path(__file__)), seed=671109, batch_count=len(batches),
                  full_width=512, trimmed_widths=widths, loss_full=loss_full, loss_trimmed=loss_trim,
                  gradients_close=True, full_seconds=[full1, full2], trimmed_seconds=[trimmed1, trimmed2],
                  individual_unchanged=True, checkpoint_unchanged=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
