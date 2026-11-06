"""Expand pretraining coverage while freezing vocabulary and held partitions.

This prepares material only; a later explicit training plan must select it.
No new final model outputs are inspected. Held text is read only for exclusion.
"""
import collections
import hashlib
import json
import shutil
import sys
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_conversation_v2 import ROOT, sha, clean, normalized, split_of
from scripts.curate_conversation import reject
from scripts.audit_conversation_data import signatures


def main():
    previous = ROOT / 'runs/conversation-data-v4'
    out = ROOT / 'runs/conversation-data-v5'
    old = json.loads((previous / 'manifest.json').read_text('utf-8'))
    source = ROOT / 'runs/language-material/pretrain_t2t_mini.jsonl'
    provenance = json.loads((source.parent / 'pretrain-provenance.json').read_text('utf-8'))
    if sha(source) != provenance['sha256']:
        raise ValueError('Unverified source')
    out.mkdir(exist_ok=False)
    for name, digest in old['files'].items():
        if name in ('pretrain-train.bin', 'pretrain-train.jsonl'):
            continue
        if sha(previous / name) != digest:
            raise ValueError('Previous data changed: ' + name)
        shutil.copyfile(previous / name, out / name)
    held_sizes = []
    index = collections.defaultdict(list)
    groups, digests = set(), set()
    for directory in ('conversation-data-v3', 'conversation-data-v4'):
        for split in ('development', 'test'):
            for kind in ('pretrain', 'dialogue'):
                with (ROOT / 'runs' / directory / f'{kind}-{split}.jsonl').open(encoding='utf-8') as stream:
                    for line in stream:
                        row = json.loads(line)
                        groups.add(row['group']); digests.add(row['sha256'])
                        sigs = signatures(row['text']); number = len(held_sizes)
                        held_sizes.append(len(sigs))
                        for sig in sigs:
                            index[sig].append(number)
    seen = set()
    with (out / 'dialogue-train.jsonl').open(encoding='utf-8') as stream:
        seen.update(json.loads(line)['sha256'] for line in stream)
    tokenizer = Tokenizer.from_file(str(out / 'tokenizer.json'))
    bos, eos = (tokenizer.token_to_id(s) for s in ('<|bos|>', '<|eos|>'))
    stats = collections.Counter()
    with source.open('rb') as stream, (out / 'pretrain-train.jsonl').open('x', encoding='utf-8') as raw, (out / 'pretrain-train.bin').open('xb') as binary:
        position = 0
        for number, line in enumerate(stream):
            bucket = min(9, position * 10 // source.stat().st_size)
            position += len(line); stats['scanned'] += 1
            text = clean(json.loads(line).get('text'))
            if text is None:
                stats['basic_filtered'] += 1; continue
            if split_of(text) != 'train':
                stats['non_training_partition'] += 1; continue
            key = normalized(text)
            digest = hashlib.sha256(key.encode()).hexdigest()
            group = hashlib.sha256(key[:96].encode()).hexdigest()
            if digest in seen:
                stats['duplicate'] += 1; continue
            row = dict(text=text, source_line=number, source_byte_bucket=bucket,
                       sha256=digest, group=group)
            why = reject(row, 'pretrain')
            if why:
                stats['removed_' + why] += 1; continue
            if group in groups or digest in digests:
                stats['held_exact_or_group'] += 1; continue
            sigs = signatures(text)
            shared = collections.Counter(i for s in sigs for i in index.get(s, ()))
            if any(min(len(sigs), held_sizes[i]) >= 8 and n / min(len(sigs), held_sizes[i]) >= .8
                   for i, n in shared.items()):
                stats['held_lexical_overlap'] += 1; continue
            seen.add(digest)
            seq = [bos] + tokenizer.encode(text).ids + [eos]
            np.array(seq, dtype=np.uint16).tofile(binary)
            raw.write(json.dumps(row, ensure_ascii=False) + '\n')
            stats['pretrain_train_documents'] += 1
            stats['pretrain_train_tokens'] += len(seq)
            stats[f'train_bucket{bucket}'] += 1
            if stats['pretrain_train_documents'] % 100000 == 0:
                print(json.dumps(dict(stage='expanded_pretrain', stats=stats)), flush=True)
    manifest = dict(old)
    manifest.update(previous_manifest_sha256=sha(previous / 'manifest.json'),
                    preparation_stats=stats,
                    source_code_sha256={name: sha(ROOT / name) for name in (
                        'scripts/prepare_conversation_v5.py', 'scripts/prepare_conversation_v2.py',
                        'scripts/curate_conversation.py', 'scripts/audit_conversation_data.py')},
                    sampling='Full accepted pretraining TRAIN partition; original v4 dialogue and held partitions unchanged.',
                    limitations='Expands coverage, not a factual-quality guarantee. Excludes v3/v4 exact/group and bounded lexical held overlaps. No semantic-isolation proof. No model training in this script.')
    manifest['stats'] = dict(old['stats'])
    manifest['stats'].update({k: stats[k] for k in ('pretrain_train_documents', 'pretrain_train_tokens')})
    manifest['files'] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(stage='completed', stats=stats)), flush=True)


if __name__ == '__main__':
    main()
