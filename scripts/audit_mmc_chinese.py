"""Inspect Chinese MMC development data without exporting subtitle text."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/mmc-source-a7007d1/dev.chinese.512.jsonlines'
SOURCE_SHA = '91dc0d91dbf5433640ab9926c85532359b1da9187b9d27dbdf41359a3b00ce03'
OUTPUT = ROOT / 'runs/mmc-chinese-dev-audit-20261110.json'
PRONOUNS = {'\u6211': 'first_singular', '\u4f60': 'second_singular',
            '\u4ed6': 'third_male', '\u5979': 'third_female'}


def mentions(document):
    tokens = document['tokens']
    mapping = document['subtoken_map']
    speakers = [speaker for segment in document['speakers'] for speaker in segment]
    sentences = [token for segment in document['sentences'] for token in segment]
    if len(speakers) != len(mapping) or len(sentences) != len(mapping):
        raise ValueError('Subtoken and speaker lengths differ')
    if any(type(index) is not int or index < 0 or index >= len(tokens) for index in mapping):
        raise ValueError('Subtoken mapping out of bounds')
    if mapping != sorted(mapping):
        raise ValueError('Subtoken mapping is not monotonic')
    for cluster_id, cluster in enumerate(document['clusters']):
        for span in cluster:
            if (len(span) != 2 or any(type(value) is not int for value in span)
                    or not 0 <= span[0] <= span[1] < len(mapping)):
                raise ValueError('Invalid cluster span')
            left, right = mapping[span[0]], mapping[span[1]]
            span_speakers = set(speakers[span[0]:span[1] + 1])
            yield {'cluster': cluster_id, 'text': ''.join(tokens[left:right + 1]),
                   'speaker': next(iter(span_speakers)) if len(span_speakers) == 1 else None}


def summarize(documents):
    counts = Counter()
    for document in documents:
        counts['documents'] += 1
        clusters = defaultdict(list)
        for mention in mentions(document):
            counts['mentions'] += 1
            counts['multi_speaker_span'] += mention['speaker'] is None
            kind = PRONOUNS.get(mention['text'])
            if kind:
                counts[kind] += 1
                clusters[mention['cluster']].append((kind, mention['speaker']))
        for values in clusters.values():
            first = [speaker for kind, speaker in values if kind == 'first_singular' and speaker not in (None, '[SPL]', '-')]
            if len(first) >= 2:
                counts['clusters_with_multiple_first_person'] += 1
                if len(set(first)) > 1:
                    counts['multiple_first_person_different_speakers'] += 1
    return dict(counts)


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Source digest mismatch')
    with SOURCE.open(encoding='utf-8') as stream:
        stats = summarize(json.loads(line) for line in stream if line.strip())
    report = {'source_sha256': SOURCE_SHA, 'repository_revision': 'a7007d1d4556a3f4347a3d7b686f71d66bd1e2d9',
              'split': 'corrected Chinese development', 'summary': stats,
              'limitations': ['Counts depend on projected/corrected annotation and subtoken map integrity.',
                              'An annotated mention or cluster is not a factual assertion or permission to write.',
                              'Speaker labels are dataset identifiers, not authenticated real-world identities.',
                              'Checking first-person co-clusters does not establish coreference annotation accuracy.',
                              'Corpus license, overlap with base pretraining and transfer to live chat remain unknown.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(stats))


if __name__ == '__main__':
    run()
