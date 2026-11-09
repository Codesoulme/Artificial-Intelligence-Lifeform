"""Train a fresh or resumed hybrid language interface on reviewed local data."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from .corpus_admission import validate_language_training_approval
from .growing import load
from .hybrid_tokenizer import FORMAT, HybridLanguageBridge, HybridTokenizer, windows
from .language_bridge import PAD, causal_loss
from .language_training import ByteTrainer, atomic_save, frozen_digest
from .text_data import overlap_audit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_non_test_documents(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding='utf8'))
    if manifest.get('schema') != 'spora-text-v1' or not isinstance(manifest.get('documents'), list):
        raise ValueError('Invalid manifest')
    documents = []
    seen = set()
    for row in manifest['documents']:
        if row.get('split') not in ('train', 'development'):
            continue
        if set(row) != {'path', 'sha256', 'split', 'source', 'license'}:
            raise ValueError('Invalid document metadata')
        source = (path.parent / row['path']).resolve()
        if not source.is_relative_to(path.parent) or sha(source) != row['sha256']:
            raise ValueError('Invalid document path or hash')
        text = source.read_bytes().decode('utf-8-sig')
        if not text.strip() or text in seen:
            raise ValueError('Empty or duplicate document')
        seen.add(text)
        documents.append({**row, 'text': text})
    if not all(any(row['split'] == split for row in documents) for split in ('train', 'development')):
        raise ValueError('Missing training or development documents')
    return documents


def evaluate(model, state, data):
    total = count = 0
    with torch.no_grad():
        for row in data:
            batch = row[None].to(next(model.parameters()).device)
            n = int((batch[:, 1:] != PAD).sum())
            total += float(causal_loss(model(batch, state), batch)) * n
            count += n
    if not count:
        raise ValueError('No targets')
    return total / count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('individual', 'manifest', 'review', 'audit', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--resume')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--length', type=int, default=128)
    p.add_argument('--context-layers', type=int, default=2)
    p.add_argument('--seed', type=int, default=2710000)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--start-language-training', action='store_true')
    a = p.parse_args()
    if not a.start_language_training or not 1 <= a.steps <= 1000 or not 1 <= a.batch <= 16 or not 2 <= a.length <= 256:
        p.error('Explicit start and bounded steps, batch and length required')
    if not 0 <= a.context_layers <= 4:
        p.error('Invalid context layers')
    output = Path(a.output)
    if output.exists():
        raise FileExistsError(output)
    parent_sha, manifest_sha, audit_sha, review_sha = (sha(getattr(a, name)) for name in ('individual', 'manifest', 'audit', 'review'))
    audit = json.loads(Path(a.audit).read_text(encoding='utf8'))
    data = audit.get('hybrid_vocab', {})
    if audit.get('manifest_sha256') != manifest_sha or data.get('format') != FORMAT:
        raise ValueError('Audit or vocabulary format mismatch')
    tokenizer = HybridTokenizer(data['symbols'])
    if data != tokenizer.payload():
        raise ValueError('Invalid tokenizer payload')
    vocab_bytes = (json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode('utf8')
    vocab_sha = hashlib.sha256(vocab_bytes).hexdigest()
    review = json.loads(Path(a.review).read_text(encoding='utf8'))
    if review.get('experiment') != 'hybrid-character-byte-diagnostic-v1' or review.get('audit_sha256') != audit_sha:
        raise ValueError('Review does not cover this hybrid diagnostic')
    documents = read_non_test_documents(a.manifest)
    if set(tokenizer.symbols) != set(''.join(d['text'] for d in documents if d['split'] == 'train')):
        raise ValueError('Vocabulary must derive from training documents')
    overlap = overlap_audit(documents)
    train, dev = (windows(documents, tokenizer, split, a.length) for split in ('train', 'development'))
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    organism, state, _ = load(a.individual, a.device)
    if a.length > organism.config.max_length:
        raise ValueError('Context exceeds individual limit')
    frozen_before = frozen_digest(organism, state)
    model = HybridLanguageBridge(organism, tokenizer, vocab_sha, a.context_layers).to(a.device)
    provenance = {'manifest_sha256': manifest_sha, 'vocab_sha256': vocab_sha, 'audit_sha256': audit_sha,
                  'review_sha256': review_sha, 'seed': a.seed, 'batch': a.batch, 'length': a.length,
                  'context_layers': a.context_layers, 'device': a.device}
    trainer = ByteTrainer(model, state, parent_sha, provenance, a.seed, format_name='spora-hybrid-training-v1')
    if a.resume:
        trainer.restore(torch.load(a.resume, map_location='cpu', weights_only=True))
    if trainer.steps + a.steps > 1000:
        raise ValueError('Total diagnostic budget exceeds 1000 updates')
    approval = validate_language_training_approval(review, manifest_sha256=manifest_sha, parent_sha256=parent_sha,
                                                   schedule_steps=trainer.steps + a.steps)
    output.mkdir(parents=True)
    (output / 'tokenizer.json').write_bytes(vocab_bytes)
    metadata = {'arguments': vars(a), 'parent_sha256': parent_sha, 'provenance': provenance,
                'approval': approval, 'overlap_train_dev': overlap, 'test_split_read': False,
                'limitations': 'Short synthetic corpus; no general Chinese or chat claim.'}
    (output / 'config.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf8')
    started = time.perf_counter()
    with (output / 'metrics.jsonl').open('x', encoding='utf8') as log:
        log.write(json.dumps({'step': trainer.steps, 'development_loss': evaluate(model, state, dev),
                              'tokens_seen': trainer.tokens_seen, 'seconds_this_run': 0.0}) + '\n')
        log.flush()
        for i in range(a.steps):
            loss = trainer.step(train, a.batch)
            final = i == a.steps - 1
            if trainer.steps % 10 == 0 or final:
                atomic_save(trainer.snapshot(), output / 'checkpoint.pt')
                entry = {'step': trainer.steps, 'train_loss': loss, 'development_loss': evaluate(model, state, dev),
                         'tokens_seen': trainer.tokens_seen, 'seconds_this_run': time.perf_counter() - started}
                log.write(json.dumps(entry) + '\n')
                log.flush()
                print(json.dumps(entry), flush=True)
    if sha(a.individual) != parent_sha or frozen_digest(organism, state) != frozen_before:
        raise AssertionError('Protected individual changed')
    (output / 'complete.json').write_text(json.dumps({'steps': trainer.steps, 'tokens_seen': trainer.tokens_seen,
        'parent_unchanged': True, 'all_organism_tensors_unchanged': True, 'organism_tensor_digest': frozen_before}) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
