"""Finite future language training; explicit launch, strict lineage, resumable."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from .growing import load
from .language_bridge import LanguageBridge, causal_loss, PAD
from .language_training import ByteTrainer, load_interface, atomic_save, frozen_digest
from .text_data import read_manifest, windows, overlap_audit
from .corpus_admission import validate_language_training_approval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--individual', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--interface')
    source.add_argument('--resume')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--review', required=True)
    parser.add_argument('--context-layers', type=int, default=0)
    parser.add_argument('--output', required=True)
    parser.add_argument('--start-language-training', action='store_true')
    parser.add_argument('--steps', type=int, default=100, help='Additional steps; total must be <=10000')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--length', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1910000)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--save-every', type=int, default=10)
    args = parser.parse_args()
    if not args.start_language_training:
        parser.error('Explicit --start-language-training required; readiness does not train text')
    if not 0 <= args.context_layers <= 4:
        parser.error('Invalid context layer count')
    if not 1 <= args.steps <= 10000 or not 1 <= args.batch <= 16 or not 2 <= args.length <= 256 or not 1 <= args.save_every <= 100:
        parser.error('Invalid finite budget')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Choose a new trial directory, including on resume')
    documents = read_manifest(args.manifest)
    train, dev = [windows(documents, split, args.length) for split in ('train', 'development')]
    # Test documents are checked for provenance/overlap, not used for selection.
    digest = hashlib.sha256(Path(args.individual).read_bytes()).hexdigest()
    review_path = Path(args.review)
    review = json.loads(review_path.read_text(encoding='utf-8'))
    manifest_sha256 = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    approval = validate_language_training_approval(
        review, manifest_sha256=manifest_sha256, parent_sha256=digest,
        schedule_steps=args.steps)
    source_file = Path(args.resume or args.interface)
    payload = torch.load(source_file, map_location='cpu', weights_only=True)
    provenance = {'manifest_sha256': manifest_sha256,
        'review_sha256': hashlib.sha256(review_path.read_bytes()).hexdigest(),
        'documents': [{k: v for k, v in d.items() if k != 'text'} for d in documents],
        'seed': args.seed, 'batch': args.batch, 'length': args.length, 'device': args.device}
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    organism, state, _ = load(args.individual, args.device)
    original_tensors = frozen_digest(organism, state)
    model = LanguageBridge(organism, context_layers=args.context_layers).to(args.device)
    if args.length > organism.config.max_length:
        raise ValueError('Context exceeds this organism configuration')
    trainer = ByteTrainer(model, state, digest, provenance, args.seed)
    if args.resume:
        trainer.restore(payload)
    else:
        expected_format = 'spora-byte-interface-v2' if args.context_layers else 'spora-byte-interface-v1'
        if payload.get('format') != expected_format or payload.get('trained_tokens') != 0:
            raise ValueError('Use --resume with a v2 checkpoint for trained interfaces')
        load_interface(model, payload, digest)
    if trainer.steps+args.steps > 10000:
        raise ValueError('Total pilot budget exceeds 10000 steps')
    output.mkdir(parents=True)
    metadata = {'arguments': vars(args), 'parent_sha256': digest, 'provenance': provenance,
        'approval': approval,
        'source_checkpoint_sha256': hashlib.sha256(source_file.read_bytes()).hexdigest(),
        'overlap': overlap_audit(documents),
        'limitations': 'Byte prediction; entire organism frozen; no text-to-event writer or skill alignment'}
    (output/'config.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    started = time.perf_counter()
    with (output/'metrics.jsonl').open('w', encoding='utf-8') as log:
        for additional in range(args.steps):
            loss = trainer.step(train, args.batch)
            final = additional == args.steps-1
            if trainer.steps % args.save_every == 0 or final:
                atomic_save(trainer.snapshot(), output/'checkpoint.pt')
            if trainer.steps % 10 == 0 or final:
                with torch.no_grad():
                    total, count = 0., 0
                    for row in dev:
                        batch = row[None].to(args.device)
                        weight = int((batch[:, 1:] != PAD).sum())
                        total += float(causal_loss(model(batch, state), batch))*weight
                        count += weight
                    validation = total/count
                record = {'step': trainer.steps, 'train_loss': loss, 'development_loss': validation,
                    'tokens_seen': trainer.tokens_seen, 'seconds_this_run': time.perf_counter()-started}
                log.write(json.dumps(record)+'\n')
                log.flush()
                print(json.dumps(record), flush=True)
    if hashlib.sha256(Path(args.individual).read_bytes()).hexdigest() != digest:
        raise AssertionError('Parent file changed')
    if frozen_digest(organism, state) != original_tensors:
        raise AssertionError('Frozen organism or experience tensors changed during training')
    (output/'complete.json').write_text(json.dumps({'steps': trainer.steps, 'tokens_seen': trainer.tokens_seen,
        'parent_unchanged': True, 'all_organism_tensors_unchanged': True, 'organism_tensor_digest': original_tensors}))


if __name__ == '__main__':
    main()
