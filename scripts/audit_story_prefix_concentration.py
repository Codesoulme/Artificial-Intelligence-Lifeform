"""Aggregate exact story-boundary repetition without exporting story text."""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path


def audit(path):
    lengths = (20, 40, 80, 120)
    prefix = {n: collections.Counter() for n in lengths}
    suffix = {n: collections.Counter() for n in lengths}
    digests = collections.Counter()
    document_count = 0
    file_hash = hashlib.sha256()
    with path.open('rb') as stream:
        for line in stream:
            file_hash.update(line)
            row = json.loads(line)
            text = row['text']
            actual = hashlib.sha256(text.encode('utf-8')).hexdigest()
            normalized = re.sub(r'\s+', '', text)
            if not normalized:
                raise ValueError('Empty story')
            document_count += 1
            digests[actual] += 1
            for n in lengths:
                if len(normalized) >= n:
                    prefix[n][normalized[:n]] += 1
                    suffix[n][normalized[-n:]] += 1

    def summarize(counter):
        return {'eligible_documents': sum(counter.values()),
                'unique_strings': len(counter),
                'documents_in_shared_groups': sum(v for v in counter.values() if v > 1),
                'duplicate_occurrences_beyond_first': sum(v - 1 for v in counter.values()),
                'largest_group': max(counter.values(), default=0)}

    return {'source_sha256': file_hash.hexdigest(), 'documents': document_count,
            'exact_text': summarize(digests),
            'prefix': {str(n): summarize(prefix[n]) for n in lengths},
            'suffix': {str(n): summarize(suffix[n]) for n in lengths},
            'method': 'Unicode codepoints after removing whitespace; exact boundaries only; groups may overlap across lengths',
            'limitations': 'No semantic, interior substring, paraphrase or contamination detection; shared beginnings/endings do not imply full duplicate stories.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.source)
    with args.output.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(result, out, ensure_ascii=False, indent=2, sort_keys=True)
        out.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
