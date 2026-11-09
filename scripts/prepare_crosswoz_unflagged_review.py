"""Select a fixed review sample from rule-unflagged CrossWOZ dialogues."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def choose(data, flags, count=20):
    if type(count) is not int or count < 1:
        raise ValueError('Invalid sample count')
    by_id = {}
    for row in flags:
        identifier = row['dialogue_id']
        if identifier in by_id or identifier not in data or not isinstance(row['flags'], list):
            raise ValueError('Invalid or duplicate flag identity')
        by_id[identifier] = row
    if set(by_id) != set(data):
        raise ValueError('Flags do not cover all source dialogues')
    unflagged = [identifier for identifier, row in by_id.items() if not row['flags']]
    if len(unflagged) < count:
        raise ValueError('Insufficient unflagged dialogues')
    selected = sorted(unflagged, key=lambda item: (sha(('p213-unflagged-v1:' + item).encode()), item))[:count]
    rows = []
    for identifier in selected:
        dialog = data[identifier]
        messages = [{'role': message['role'], 'content': message['content']} for message in dialog['messages']]
        content = json.dumps(messages, ensure_ascii=False, separators=(',', ':')).encode('utf8')
        rows.append({'dialogue_id': identifier, 'type': dialog.get('type', 'unknown'),
                     'messages_sha256': sha(content), 'messages': messages, 'status': 'review_pending_not_approved'})
    return rows, len(unflagged)


def run(source, flags_dir, output):
    source, flags_dir, output = Path(source), Path(flags_dir), Path(output)
    if output.exists():
        raise FileExistsError(output)
    source_manifest = json.loads((source / 'manifest.json').read_text(encoding='utf8'))
    flags_manifest = json.loads((flags_dir / 'manifest.json').read_text(encoding='utf8'))
    archive = source / 'train.json.zip'
    flags_raw = (flags_dir / 'flags.jsonl').read_bytes()
    if (sha(archive.read_bytes()) != source_manifest['archive_sha256']
            or flags_manifest['archive_sha256'] != source_manifest['archive_sha256']
            or sha(flags_raw) != flags_manifest['flags_sha256']):
        raise ValueError('Input digest mismatch')
    with zipfile.ZipFile(archive) as bundle:
        if bundle.namelist() != ['train.json']:
            raise ValueError('Unexpected archive members')
        raw = bundle.read('train.json')
    if sha(raw) != source_manifest['train_json_sha256']:
        raise ValueError('Source JSON digest mismatch')
    rows, population = choose(json.loads(raw.decode('utf8')),
                              [json.loads(line) for line in flags_raw.splitlines()])
    output.mkdir(parents=False)
    path = output / 'review.jsonl'
    with path.open('x', encoding='utf8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    summary = {'status': 'review_queue_not_training_approval', 'unflagged_population': population,
               'sample_size': len(rows), 'messages': sum(len(row['messages']) for row in rows),
               'archive_sha256': source_manifest['archive_sha256'], 'flags_sha256': sha(flags_raw),
               'queue_sha256': sha(path.read_bytes()), 'test_split_read': False,
               'selection': 'lowest SHA-256 of fixed salt and dialogue ID',
               'limitations': ['Unflagged does not mean safe or correct.', 'Sample does not prove source rights.',
                               'No training approval.']}
    (output / 'manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'flags-dir', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.flags_dir, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
