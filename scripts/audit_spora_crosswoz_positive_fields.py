"""Audit real Chinese requested-field supervision before adapting the address head."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

from scripts.audit_spora_natural_lookup_cli import digest
from scripts.train_spora_recurrent_counterfactual import store_json


PINNED = {
    'train': 'b81f6a0845011bb4593d21dd58e2d76af4dd5d52e300ecc68cb8dee17ccc3cf8',
    'val': 'b2b717990f6f0a1c98daaf5dc465e61d42f7de69c0d31e88e674275b1f34c94c',
    'test': 'e57703894995211bba8638f130dc7df81e8992b71f3c05c10022ad23d6ca5310',
}
REVISION = 'df82c9fdff91b9b130f2d6b89110d3870ba6260e'


def requested(turn):
    """Use annotated Request acts, with domain to disambiguate equal slot names."""
    return sorted({(domain, field) for action, domain, field, _ in turn['dialog_act']
                   if action == 'Request' and domain not in ('none', '') and field not in ('none', '')})


def annotate(dialogues):
    counts, fields = Counter(), Counter()
    rows = []
    for identifier, dialogue in sorted(dialogues.items()):
        counts['dialogues'] += 1
        previous_turn, previous_domains = None, set()
        messages = dialogue['messages']
        for turn, message in enumerate(messages):
            if message['role'] != 'usr':
                continue
            counts['user_turns'] += 1
            addresses = requested(message)
            if not addresses:
                continue
            counts['request_turns'] += 1
            domains = {domain for domain, _ in addresses}
            multi_field = len(addresses) > 1
            multi_domain = len(domains) > 1
            switched = bool(previous_domains and previous_domains.isdisjoint(domains))
            fields.update(f'{domain}/{field}' for domain, field in addresses)
            counts.update(multiple_fields=multi_field, multiple_domains=multi_domain, domain_shift=switched)
            if multi_field or multi_domain or switched:
                flags = [name for name, active in (('multiple_fields', multi_field), ('multiple_domains', multi_domain),
                                                   ('domain_shift', switched)) if active]
                rows.append({'dialogue': identifier, 'turn': turn, 'previous_request': previous_turn,
                             'requested': addresses, 'flags': flags,
                             'utterance_sha256': sha256(message['content'].encode('utf-8')).hexdigest()})
            previous_turn, previous_domains = turn, domains
    counts['positive_manifest_rows'] = len(rows)
    counts['unique_domain_fields'] = len(fields)
    manifested = Counter(flag for row in rows for flag in row['flags'])
    if any(counts[key] != manifested[key] for key in ('multiple_fields', 'multiple_domains', 'domain_shift')):
        raise AssertionError('Manifest counts must reproduce annotation counts')
    return counts, fields, rows


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    observed = set()
    summary = {}
    subjects = {}
    for split, expected in PINNED.items():
        path = args.corpus / f'{split}.json.zip'
        if digest(path) != expected:
            raise ValueError(f'{split} archive does not match the pinned CrossWOZ release')
        with ZipFile(path) as archive:
            dialogues = json.loads(archive.read(f'{split}.json'))
        if observed & set(dialogues):
            raise ValueError('Official dialogue identifiers overlap across splits')
        observed.update(dialogues)
        counts, fields, rows = annotate(dialogues)
        manifest = args.output / f'{split}.json'
        store_json(manifest, {'rows': rows, 'source_archive_sha256': expected})
        summary[split] = {**counts, 'manifest_sha256': digest(manifest)}
        subjects[split] = sorted(fields)
    store_json(args.output / 'result.json', {'revision': REVISION, 'splits': summary,
                                            'fields': subjects, 'sources': PINNED})
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('output', type=Path)
    main(parser.parse_args())
