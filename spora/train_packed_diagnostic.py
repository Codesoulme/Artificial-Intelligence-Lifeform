"""Run a bounded packed-stream diagnostic on an existing reviewed corpus."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .document_stream import DocumentStream
from .growing import load
from .hybrid_tokenizer import HybridLanguageBridge, HybridTokenizer, windows
from .language_bridge import BOS, EOS, PAD
from .language_training import atomic_save, frozen_digest, load_interface
from .packed_hybrid_training import PackedHybridTrainer
from .text_data import overlap_audit
from .train_hybrid import evaluate, read_non_test_documents, sha, validate_diagnostic_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('individual', 'manifest', 'review', 'audit', 'initial', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--resume')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--length', type=int, default=128)
    parser.add_argument('--context-layers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2713001)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if not (1 <= args.steps <= 200 and 1 <= args.batch <= 16 and 2 <= args.length <= 256
            and 0 <= args.context_layers <= 4):
        parser.error('Invalid diagnostic schedule')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    individual_sha, manifest_sha, review_sha, audit_sha, initial_sha = (
        sha(getattr(args, name)) for name in ('individual', 'manifest', 'review', 'audit', 'initial'))
    audit = json.loads(Path(args.audit).read_text(encoding='utf8'))
    review = json.loads(Path(args.review).read_text(encoding='utf8'))
    vocab_data = audit.get('hybrid_vocab', {})
    tokenizer = HybridTokenizer(vocab_data.get('symbols', []))
    if (vocab_data != tokenizer.payload() or tokenizer.size > 65536
            or audit.get('manifest_sha256') != manifest_sha):
        raise ValueError('Audit does not bind manifest and vocabulary')
    vocab_bytes = (json.dumps(vocab_data, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode('utf8')
    vocab_sha = hashlib.sha256(vocab_bytes).hexdigest()
    if (audit.get('source_checkpoint_sha256') != initial_sha or review.get('transfer_checkpoint_sha256') != initial_sha
            or audit.get('tokenizer_sha256') != vocab_sha):
        raise ValueError('Transfer checkpoint or vocabulary not covered by audit and review')
    documents = read_non_test_documents(args.manifest)
    overlap = overlap_audit(documents)
    train_docs = [d for d in documents if d['split'] == 'train']
    if not set(tokenizer.symbols) <= set(''.join(d['text'] for d in train_docs)):
        raise ValueError('Vocabulary not contained in training documents')
    encoded = np.concatenate([np.asarray(tokenizer.encode(d['text']), dtype=np.uint16) for d in train_docs])
    stream = DocumentStream(encoded, BOS, EOS, PAD)
    dev = windows(documents, tokenizer, 'development', args.length)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    organism, state, _ = load(args.individual, args.device)
    if args.length > organism.config.max_length:
        raise ValueError('Context exceeds individual limit')
    frozen_before = frozen_digest(organism, state)
    model = HybridLanguageBridge(organism, tokenizer, vocab_sha, args.context_layers).to(args.device)
    initial = torch.load(args.initial, map_location='cpu', weights_only=True)
    if initial.get('provenance', {}).get('manifest_sha256') != audit.get('previous_manifest_sha256'):
        raise ValueError('Transfer source lineage mismatch')
    load_interface(model, initial, individual_sha)
    provenance = {'tokens_sha256': hashlib.sha256(encoded.astype('<u2', copy=False).tobytes()).hexdigest(),
                  'index_sha256': manifest_sha, 'vocab_sha256': vocab_sha, 'verification_sha256': audit_sha,
                  'admission_sha256': review_sha, 'initial_sha256': initial_sha,
                  'seed': args.seed, 'batch': args.batch, 'length': args.length,
                  'context_layers': args.context_layers, 'device': args.device}
    trainer = PackedHybridTrainer(model, state, individual_sha, provenance, args.seed)
    if args.resume:
        trainer.restore(torch.load(args.resume, map_location='cpu', weights_only=True))
    approval = validate_diagnostic_review(review, manifest_sha=manifest_sha, parent_sha=individual_sha,
                                          audit_sha=audit_sha, steps=trainer.steps + args.steps)
    output.mkdir(parents=True)
    metadata = {'arguments': vars(args), 'provenance': provenance, 'approval': approval,
                'parent_sha256': individual_sha, 'overlap_train_dev': overlap,
                'inherited_steps': initial['steps'], 'inherited_tokens': initial['trained_tokens'],
                'resume_sha256': sha(args.resume) if args.resume else None,
                'test_split_read': False, 'note': 'Noisy stories; diagnostic only, not a chat evaluation.'}
    (output / 'config.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf8')
    start = time.perf_counter()
    with (output / 'metrics.jsonl').open('x', encoding='utf8') as log:
        baseline = {'step': trainer.steps, 'development_loss': evaluate(model, state, dev),
                    'tokens_seen': trainer.tokens_seen, 'seconds_this_run': 0.0}
        log.write(json.dumps(baseline) + '\n')
        log.flush()
        for i in range(args.steps):
            loss = trainer.step(stream)
            if trainer.steps % 10 == 0 or i == args.steps - 1:
                atomic_save(trainer.snapshot(), output / 'checkpoint.pt')
                entry = {'step': trainer.steps, 'train_loss': loss,
                         'development_loss': evaluate(model, state, dev), 'tokens_seen': trainer.tokens_seen,
                         'coverage': trainer.coverage.metrics(), 'seconds_this_run': time.perf_counter() - start}
                log.write(json.dumps(entry) + '\n')
                log.flush()
                print(json.dumps(entry), flush=True)
    if (sha(args.individual) != individual_sha or sha(args.initial) != initial_sha
            or frozen_digest(organism, state) != frozen_before):
        raise AssertionError('Protected inputs changed')
    (output / 'complete.json').write_text(json.dumps({'steps': trainer.steps, 'tokens_seen': trainer.tokens_seen,
        'coverage': trainer.coverage.metrics(), 'parent_unchanged': True, 'organism_digest': frozen_before}) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
