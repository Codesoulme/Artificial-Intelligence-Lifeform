"""Fetch the pinned CrossWOZ training archive and summarize its structure."""
import argparse
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

REVISION = 'df82c9fdff91b9b130f2d6b89110d3870ba6260e'
URL = f'https://raw.githubusercontent.com/thu-coai/CrossWOZ/{REVISION}/data/crosswoz/train.json.zip'
GIT_BLOB = '57eb9cbd004a2e59f301f9e1bdc5500ddf2a98ef'


def git_blob_sha(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode('ascii') + b'\0' + data).hexdigest()


def summarize(data):
    if not isinstance(data, dict) or not data:
        raise ValueError('Expected nonempty dialogue mapping')
    roles, lengths, types, risk = Counter(), Counter(), Counter(), Counter()
    turns = 0
    for identifier, dialog in data.items():
        if not isinstance(identifier, str) or not isinstance(dialog, dict):
            raise ValueError('Invalid dialogue')
        messages = dialog.get('messages')
        if not isinstance(messages, list) or not messages:
            raise ValueError('Missing messages')
        types[str(dialog.get('type', 'unknown'))] += 1
        lengths[str(min(len(messages) // 10 * 10, 100))] += 1
        for message in messages:
            role, text = message.get('role'), message.get('content')
            if role not in ('usr', 'sys') or not isinstance(text, str):
                raise ValueError('Invalid message role or content')
            roles[role] += 1
            turns += 1
            if re.search(r'(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)', text):
                risk['mobile_pattern_messages'] += 1
            if re.search(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', text):
                risk['email_pattern_messages'] += 1
            if re.search(r'https?://|www\.', text, re.I):
                risk['url_pattern_messages'] += 1
    return {'dialogues': len(data), 'messages': turns, 'roles': dict(roles),
            'dialogue_length_buckets_10': dict(lengths), 'dialogue_types': dict(types),
            'risk_pattern_counts': dict(risk)}


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    with urlopen(URL, timeout=60) as response:
        raw = response.read(20_000_001)
    if len(raw) > 20_000_000 or git_blob_sha(raw) != GIT_BLOB:
        raise ValueError('Archive size or pinned git blob mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].filename != 'train.json' or members[0].file_size > 450_000_000:
            raise ValueError('Unexpected training archive structure')
        text = archive.read(members[0])
    data = json.loads(text.decode('utf8'))
    stats = summarize(data)
    output.mkdir(parents=False)
    (output / 'train.json.zip').write_bytes(raw)
    result = {'status': 'candidate_not_approved_for_training', 'training_started': False,
              'source_url': URL, 'source_revision': REVISION, 'git_blob_sha1': GIT_BLOB,
              'archive_sha256': hashlib.sha256(raw).hexdigest(), 'train_json_sha256': hashlib.sha256(text).hexdigest(),
              'archive_bytes': len(raw), 'train_json_bytes': len(text), 'statistics': stats,
              'test_split_read': False,
              'limitations': ['Apache-2.0 is a dataset-card declaration; the card says annotators agreed to research use.',
                              'Underlying knowledge and third-party content rights require further review.',
                              'Regex matches are risk indicators, not PII determinations.',
                              'Task dialogue is not an independent general-chat assessment.']}
    (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
