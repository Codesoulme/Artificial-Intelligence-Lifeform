"""Create a deterministic training-dialogue review queue; never trains."""
import argparse
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def make_queue(data, per_type=4):
    groups = defaultdict(list)
    for identifier, dialog in data.items():
        key = sha(('p211-crosswoz-review-v1:' + identifier).encode('utf8'))
        groups[dialog.get('type', 'unknown')].append((key, identifier, dialog))
    selected = []
    for category in sorted(groups):
        for _, identifier, dialog in sorted(groups[category])[:per_type]:
            messages = [{'role': message['role'], 'content': message['content']} for message in dialog['messages']]
            canonical = json.dumps(messages, ensure_ascii=False, separators=(',', ':')).encode('utf8')
            selected.append({'dialogue_id': identifier, 'type': category, 'messages_sha256': sha(canonical),
                             'messages': messages, 'status': 'review_pending_not_approved'})
    return selected


def run(source_dir, output):
    source_dir, output = Path(source_dir), Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest_bytes = (source_dir / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    raw = (source_dir / 'train.json.zip').read_bytes()
    if sha(raw) != manifest.get('archive_sha256'):
        raise ValueError('Archive digest mismatch')
    with zipfile.ZipFile(source_dir / 'train.json.zip') as archive:
        if archive.namelist() != ['train.json']:
            raise ValueError('Unexpected archive members')
        text = archive.read('train.json')
    if sha(text) != manifest.get('train_json_sha256'):
        raise ValueError('Training JSON digest mismatch')
    queue = make_queue(json.loads(text.decode('utf8')))
    output.mkdir(parents=False)
    queue_path = output / 'review.jsonl'
    with queue_path.open('x', encoding='utf8', newline='\n') as stream:
        for row in queue:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    summary = {'status': 'review_queue_not_training_approval', 'source_manifest_sha256': sha(manifest_bytes),
               'archive_sha256': sha(raw), 'queue_sha256': sha(queue_path.read_bytes()),
               'dialogues': len(queue), 'messages': sum(len(row['messages']) for row in queue),
               'selection': 'four dialogues per type by SHA-256 of fixed salt and dialogue ID',
               'test_split_read': False,
               'limitations': ['Fixed stratified sample is not representative of the full population.',
                               'Role labels do not imply factual correctness or authorization.',
                               'No independent human review or training approval.']}
    (output / 'manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
