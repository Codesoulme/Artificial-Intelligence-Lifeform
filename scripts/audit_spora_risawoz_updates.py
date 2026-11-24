"""Count RiSAWOZ development-state updates and link matching coreference transcripts."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile


DATA_MEMBER = 'source-data/all_dev600.json'
COREF_MEMBER = 'task4-data-Coreference resolution/dev.chinese.v4_gold_conll'
SOURCE_HASHES = ('8f2396b96a5254df06ff7d856f78ff1f5438e1ce7d037e7a7f40152bb6e0c1cd',
                 '726e9a9a94a21bd308617d601028eda100595e5e59083c60d6734a5d60f34eae',
                 '42bd584281ce71f43cc511a8fc74d886dd1dde818afcf279dd8a1021c6a14437')


def compact(text):
    return ''.join(text.split())


def coref_index(archive):
    transcripts = defaultdict(list)
    with ZipFile(archive) as bundle, bundle.open(COREF_MEMBER) as stream:
        words, mentions, offset, opened = [], [], 0, defaultdict(list)
        for line in stream:
            line = line.decode('utf-8').strip()
            if line.startswith('#begin document'):
                document = line
            elif line == '#end document':
                if opened:
                    raise ValueError('Unclosed coreference annotation')
                transcripts[''.join(words)].append((document, mentions))
                words, mentions, offset = [], [], 0
            elif line:
                cols = line.split('\t')
                word = compact(cols[3])
                stop = offset + len(word)
                speaker = cols[9]
                for tag in cols[-1].split('|'):
                    if tag == '-':
                        continue
                    cluster = tag.strip('()')
                    if tag.startswith('('):
                        opened[cluster].append((offset, speaker))
                    if tag.endswith(')'):
                        start, owner = opened[cluster].pop()
                        if not opened[cluster]:
                            del opened[cluster]
                        mentions.append((cluster, start, stop, owner))
                words.append(word)
                offset = stop
                sentence = []
    return transcripts


def inspect(dialogues, transcripts):
    counters = Counter()
    fields = Counter()
    examples = []
    for dialogue in dialogues:
        turns = dialogue['dialogue']
        text = ''.join(compact(turn[role]) for turn in turns for role in ('user_utterance', 'system_utterance'))
        matches = transcripts.get(text, ())
        counters['dialogues_aligned'] += len(matches) == 1
        mentions = matches[0][1] if len(matches) == 1 else ()
        first_mentions = {}
        for cluster, start, _, _ in mentions:
            first_mentions[cluster] = min(start, first_mentions.get(cluster, start))
        previous, origins, offset = {}, {}, 0
        for turn in turns:
            counters['turns'] += 1
            utterance = turn['user_utterance']
            surface = compact(utterance)
            start, end = offset, offset + len(surface)
            offset = end + len(compact(turn['system_utterance']))
            state = turn['belief_state']
            active = state['inform slot-values']
            delta = state['turn_inform'] or {}
            actions = {('-'.join((domain, slot)), compact(value))
                       for act, domain, slot, value in turn['user_actions'] if act == 'Inform'}
            prior_ref = any(owner == 'user' and start <= left < right <= end and first_mentions[cluster] < start
                            for cluster, left, right, owner in mentions)
            for key, value in active.items():
                new, old = compact(value), compact(previous.get(key, ''))
                if new == old:
                    continue
                prior_turn = origins.get(key)
                origins[key] = turn['turn_id']
                if not old or not new:
                    continue
                counters['changed_slots'] += 1
                if compact(delta.get(key, '')) != new or (key, new) not in actions:
                    continue
                counters['agreed_updates'] += 1
                if len(new) < 2 or surface.count(new) != 1:
                    continue
                counters['unique_literal_updates'] += 1
                counters['literal_updates_aligned'] += len(matches) == 1
                counters['literal_updates_with_prior_coref'] += prior_ref
                fields[key] += 1
                examples.append({'dialogue': dialogue['dialogue_id'], 'turn': turn['turn_id'], 'field': key,
                                 'old_turn': prior_turn, 'prior_coref': prior_ref})
            previous = active
    counters['update_turns'] = len({(row['dialogue'], row['turn']) for row in examples})
    counters['update_dialogues'] = len({row['dialogue'] for row in examples})
    return {'dialogues': len(dialogues), **dict(counters), 'field_counts': dict(fields.most_common()), 'rows': examples}


def main(args):
    hashes = tuple(sha256(path.read_bytes()).hexdigest() for path in (args.source, args.coref, args.splits))
    if hashes != SOURCE_HASHES:
        raise ValueError('Expected the pinned official RiSAWOZ release')
    with ZipFile(args.source) as bundle, bundle.open(DATA_MEMBER) as stream:
        dialogues = json.load(stream)
    splits = json.loads(args.splits.read_text(encoding='utf-8'))
    dev = {row['dialogue_id'] for row in dialogues}
    if (len(dev) != len(dialogues) or dev != set(splits['all_dev600.json'])
            or dev & (set(splits['all_train10000.json']) | set(splits['all_test600_new.json']))):
        raise ValueError('Official development IDs do not match or overlap another split')
    result = inspect(dialogues, coref_index(args.coref))
    result['source_sha256'] = list(hashes)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print({key: value for key, value in result.items() if isinstance(value, int)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'coref', 'splits', 'output'):
        parser.add_argument(name, type=Path)
    main(parser.parse_args())
