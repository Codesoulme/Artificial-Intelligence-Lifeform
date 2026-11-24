"""Sample annotated address requests from the pinned CrossWOZ training split."""

import argparse
import codecs
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import random
import struct
from urllib.request import urlopen
from zipfile import ZipFile
import zlib


REVISION = 'df82c9fdff91b9b130f2d6b89110d3870ba6260e'
URL = f'https://raw.githubusercontent.com/thu-coai/CrossWOZ/{REVISION}/data/crosswoz/train.json.zip'
DOMAINS = ('景点', '酒店', '餐馆')


def training_prefix(chunk):
    """Recover only complete JSON records within the first deflated ZIP member."""
    signature, _, flags, method, _, _, _, _, _, name_size, extra_size = struct.unpack_from('<4s5H3I2H', chunk)
    start = 30 + name_size + extra_size
    if signature != b'PK\x03\x04' or method != 8 or flags & 1 or chunk[30:30 + name_size] != b'train.json':
        raise ValueError('Expected a deflated first member called train.json')
    plain = zlib.decompressobj(-15).decompress(chunk[start:])
    text = codecs.getincrementaldecoder('utf-8')().decode(plain, final=False)
    decoder = json.JSONDecoder()
    cursor = text.index('{') + 1
    during = {}
    while cursor < len(text):
        while cursor < len(text) and (text[cursor].isspace() or text[cursor] == ','):
            cursor += 1
        if cursor == len(text) or text[cursor] == '}':
            break
        try:
            name, ending = decoder.raw_decode(text, cursor)
            cursor = text.index(':', ending) + 1
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            row, cursor = decoder.raw_decode(text, cursor)
        except (json.JSONDecodeError, ValueError):
            break
        during[name] = row
    if not during:
        raise ValueError('No complete training records in the source prefix')
    return during


def sample(dialogues):
    groups = defaultdict(dict)
    for record in dialogues.values():
        for turn in record['messages']:
            if turn['role'] != 'usr':
                continue
            requests = [(domain, field) for act, domain, field, _ in turn['dialog_act'] if act == 'Request']
            text = turn['content'].strip()
            if not 4 <= len(text) <= 100 or len({domain for domain, _ in requests}) != 1:
                continue
            domain = requests[0][0]
            if domain not in DOMAINS:
                continue
            asks_address = (domain, '地址') in requests
            groups[domain, asks_address][text] = None
    rng = random.Random(26140)
    all_rows = []
    selected = set()
    for domain in DOMAINS:
        for label in (True, False):
            other = groups[domain, not label]
            pool = sorted(set(groups[domain, label]) - other.keys() - selected)
            chosen = rng.sample(pool, 16)
            selected.update(chosen)
            all_rows.extend({'text': text, 'required': label, 'domain': domain} for text in chosen)
    if len({row['text'] for row in all_rows}) != len(all_rows):
        raise ValueError('Selected CrossWOZ utterances overlap between domains')
    return all_rows


def main(path, local_archive, prefix):
    if local_archive is not None or prefix is not None:
        payload = (local_archive if prefix is None else prefix).read_bytes()
    else:
        with urlopen(URL, timeout=60) as response:
            payload = response.read()
    if prefix is None:
        with ZipFile(BytesIO(payload)) as source, source.open('train.json') as stream:
            dialogues = json.load(stream)
    else:
        dialogues = training_prefix(payload)
    rows = sample(dialogues)
    with path.open('x', encoding='utf-8') as stream:
        json.dump({'source': URL, 'payload_sha256': sha256(payload).hexdigest(), 'rows': rows,
                   'source_bytes': len(payload), 'sampled_dialogues': len(dialogues), 'prefix': prefix is not None},
                  stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(f'Prepared {len(rows)} address-supervision turns', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    input_kind = parser.add_mutually_exclusive_group()
    input_kind.add_argument('--archive', type=Path, help='Local copy of the pinned training archive')
    input_kind.add_argument('--prefix', type=Path, help='Range response beginning at byte zero')
    args = parser.parse_args()
    main(args.output, args.archive, args.prefix)
