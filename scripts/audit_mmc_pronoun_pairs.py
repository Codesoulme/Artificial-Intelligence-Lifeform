"""Assess candidate pronoun-pair task and a role heuristic on MMC development data."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.audit_mmc_chinese import SOURCE as DEV, SOURCE_SHA as DEV_SHA

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'runs/mmc-source-a7007d1/train.chinese.512.jsonlines'
TRAIN_SHA = 'd8b923c291dab6deeadb89545253d202cf47bed0634637a248502cd8ebb469f9'
OUTPUT = ROOT / 'runs/mmc-pronoun-pair-audit-20261110.json'
FIRST, SECOND = '\u6211', '\u4f60'


def pair_candidates(document, window=128, prior_limit=5):
    if type(window) is not int or window < 1 or type(prior_limit) is not int or prior_limit < 1:
        raise ValueError('Positive window and prior limit required')
    mapping = document['subtoken_map']
    tokens = document['tokens']
    speakers = [s for segment in document['speakers'] for s in segment]
    if (len(speakers) != len(mapping) or mapping != sorted(mapping)
            or any(type(i) is not int or not 0 <= i < len(tokens) for i in mapping)):
        raise ValueError('Invalid alignment')
    entities, occurrences = [], Counter()
    for cluster, spans in enumerate(document['clusters']):
        for span in spans:
            if (len(span) != 2 or any(type(i) is not int for i in span)
                    or not 0 <= span[0] <= span[1] < len(mapping)):
                raise ValueError('Invalid span')
            start, end = span
            surface = ''.join(tokens[mapping[start]:mapping[end] + 1])
            labels = set(speakers[start:end + 1])
            if surface not in (FIRST, SECOND) or len(labels) != 1:
                continue
            speaker = next(iter(labels))
            if not isinstance(speaker, str) or speaker in ('', '[SPL]', '-'):
                continue
            entities.append({'start': start, 'end': end, 'surface': surface, 'speaker': speaker, 'cluster': cluster})
            occurrences[(start, end)] += 1
    entities = sorted((m for m in entities if occurrences[(m['start'], m['end'])] == 1), key=lambda m: m['start'])
    for i, current in enumerate(entities):
        preceding = [m for m in entities[:i] if m['end'] < current['start']
                     and current['start'] - m['end'] <= window]
        for previous in preceding[-prior_limit:]:
            present_speakers = {s for s in speakers[previous['start']:current['end'] + 1]
                                if isinstance(s, str) and s not in ('', '[SPL]', '-')}
            # In two-party conversation, "I" denotes the speaker and "you" the other party.
            same_speaker = previous['speaker'] == current['speaker']
            same_form = previous['surface'] == current['surface']
            yield {'previous': previous, 'current': current, 'same_entity': previous['cluster'] == current['cluster'],
                   'heuristic_same': same_speaker == same_form, 'speaker_count': len(present_speakers)}


def summarize(documents):
    counts = Counter()
    for document in documents:
        counts['documents'] += 1
        for pair in pair_candidates(document):
            counts['pairs'] += 1
            group = 'two_speakers' if pair['speaker_count'] == 2 else 'other_speaker_count'
            counts[group + '_pairs'] += 1
            counts[group + '_same_entity'] += pair['same_entity']
            counts[group + '_heuristic_correct'] += pair['same_entity'] == pair['heuristic_same']
            counts[group + '_false_same'] += pair['heuristic_same'] and not pair['same_entity']
            counts[group + '_false_different'] += not pair['heuristic_same'] and pair['same_entity']
    return dict(counts)


def load(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Source digest mismatch')
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    train, dev = load(TRAIN, TRAIN_SHA), load(DEV, DEV_SHA)
    keys = [{d['doc_key'] for d in split} for split in (train, dev)]
    scenes = [{d['doc_key'].rsplit('_', 1)[0] for d in split} for split in (train, dev)]
    texts = [{tuple(d['tokens']) for d in split} for split in (train, dev)]
    result = {'source_sha256': {'train': TRAIN_SHA, 'dev': DEV_SHA},
              'exact_overlap': {'doc_keys': len(keys[0] & keys[1]), 'scene_keys': len(scenes[0] & scenes[1]),
                                'whole_token_sequences': len(texts[0] & texts[1])},
              'train': summarize(train), 'dev': summarize(dev),
              'limitations': ['Only annotated I/you mentions, with at most five earlier candidates within 128 subtokens.',
                              'Different annotated clusters are treated as different entities; annotation may be incomplete or wrong.',
                              'Two speakers in a local window need not mean a two-person conversation or identify addressee.',
                              'Counts are correlated pairs, not independent samples or an estimate of live-chat accuracy.',
                              'Development set is inspected, not a held-out final benchmark; paraphrase and base pretraining overlap unknown.',
                              'No fact truth, user identity, permission or write safety is established.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('exact_overlap', 'train', 'dev')}))


if __name__ == '__main__':
    run()
