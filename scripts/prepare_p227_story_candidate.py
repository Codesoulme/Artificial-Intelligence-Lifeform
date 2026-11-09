"""Select second-review clean stories; output is still a limited candidate."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.prepare_p227_story_review import SAMPLE, SAMPLE_SHA, validated_sample

ROOT = Path(__file__).resolve().parents[1]
QUEUE_SHA = 'a1fb693e78d688f807d95253ff3c0cf6e0a0089122a93e061201e7d429a7c9d8'
PARTS = ((0, 20), (21, 40), (41, 60), (61, 80))
VOCAB_SHA = 'b3629001784896437c4aa9f279f6b4cdc5a534be191b537009e7293f971fda20'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build(queue, reviews, output):
    queue, output = Path(queue), Path(output)
    reviews = list(map(Path, reviews))
    if output.exists():
        raise FileExistsError(output)
    runs = (ROOT/'runs').resolve()
    if output.resolve() == runs or not output.resolve().is_relative_to(runs):
        raise ValueError('Output must be below runs')
    queue_raw = queue.read_bytes()
    source_raw = SAMPLE.read_bytes()
    if sha(queue_raw) != QUEUE_SHA or sha(source_raw) != SAMPLE_SHA or len(reviews) != len(PARTS):
        raise ValueError('Input digest or review count mismatch')
    original = validated_sample(source_raw)
    items = [json.loads(line) for line in queue_raw.splitlines()]
    if len(items) != 81 or len({item['sample_id'] for item in items}) != len(items):
        raise ValueError('Queue identity mismatch')
    counts, records, hashes = Counter(), [], []
    for path, (first, last) in zip(reviews, PARTS):
        raw = path.read_bytes()
        verdicts = json.loads(raw)
        if not isinstance(verdicts, list) or sorted(v.get('index') for v in verdicts) != list(range(first, last+1)):
            raise ValueError('Review indices mismatch')
        hashes.append(sha(raw))
        for verdict in verdicts:
            index = verdict['index']
            item = items[index]
            sid = item['sample_id']
            if (verdict.get('sample_id') != sid or sid not in original or item['text'] != original[sid]['text']
                    or verdict.get('label') not in ('clean', 'minor', 'major', 'uncertain')
                    or not isinstance(verdict.get('reason'), str) or not verdict['reason'].strip()):
                raise ValueError('Review or story identity mismatch')
            counts[verdict['label']] += 1
            if verdict['label'] == 'clean':
                source = original[sid]
                records.append({'source_row_1based': len(records)+1, 'text': source['text'],
                                'sha256': source['normalized_sha256'], 'group': source['group_sha256'],
                                'source_original_sha256': source['source_original_sha256'],
                                'origin_source': 'p120_multilingual_tinystories',
                                'origin_source_row_1based': source['source_row_1based'],
                                'sample_id': sid, 'content_tokens': source['content_tokens']})
    if not records or sum(counts.values()) != len(items):
        raise ValueError('Empty or incomplete review')
    candidate = ''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':'))+'\n' for r in records).encode('utf8')
    manifest = {'status': 'two_model_reviewed_story_candidate_not_training_approval',
                'queue_sha256': QUEUE_SHA, 'source_sha256': SAMPLE_SHA, 'review_sha256': hashes,
                'tokenizer_sha256': VOCAB_SHA, 'candidate_sha256': sha(candidate),
                'counts': {'reviews': dict(counts), 'records': len(records),
                           'content_tokens': sum(r['content_tokens'] for r in records)},
                'limitations': ['Independent model review is not human or rights certification.',
                                'Synthetic translated stories are narrow and may contain remaining errors.',
                                'Historical, cross-source, semantic and final-test overlap remain unverified.']}
    output.mkdir(parents=False)
    (output/'candidate.jsonl').write_bytes(candidate)
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    queue = ROOT/'runs/p227-story-review-20261109/samples.jsonl'
    reviews = [ROOT/f'runs/p227-story-review-{a}-{b}.json' for a, b in PARTS]
    print(json.dumps(build(queue, reviews, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
