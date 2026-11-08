"""Sample a reproducible train-only corpus review, without authorizing use."""
import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Directory with a manifest and pretrain-train.jsonl')
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=691109)
    p.add_argument('--count', type=int, default=24)
    p.add_argument('--prefix', type=int, default=350)
    args = p.parse_args()
    if not 1 <= args.count <= 200 or not 1 <= args.prefix <= 2000:
        raise ValueError('Expected a bounded audit sample')
    data = ROOT / args.data
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    path = data / 'pretrain-train.jsonl'
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf8'))
    digest = sha256(path)
    if manifest['files'][path.name] != digest:
        raise ValueError('Training input differs from manifest')
    rng = random.Random(args.seed)
    sample = []
    with path.open(encoding='utf8') as f:
        for index, line in enumerate(f):
            row = json.loads(line)
            text = row['text']
            item = dict(index=index, text_sha256=hashlib.sha256(text.encode('utf8')).hexdigest(),
                        characters=len(text), source=row.get('source'), prefix=text[:args.prefix])
            if index < args.count:
                sample.append(item)
            else:
                position = rng.randrange(index + 1)
                if position < args.count:
                    sample[position] = item
    if len(sample) < args.count:
        raise ValueError('Fewer rows than requested')
    result = dict(status='sample_for_review_not_approved_for_training', seed=args.seed,
                  source_sha256=digest, manifest_sha256=sha256(data / 'manifest.json'),
                  sample_size=len(sample), prefix_characters=args.prefix,
                  rows=sorted(sample, key=lambda row: row['index']))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
