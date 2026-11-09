"""Flag CrossWOZ training dialogues for review without transforming content."""
import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path


PATTERNS = {
    'phone': re.compile(r'(?<!\d)(?:(?:\+?86[- ]?)?1[3-9]\d{9}|0\d{2,3}[- ]?\d{7,8}|400[- ]?\d{3}[- ]?\d{4})(?!\d)'),
    'email': re.compile(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', re.I),
    'url': re.compile(r'https?://|www\.', re.I),
    'address_terms': re.compile(r'地址|位于|具体位置'),
    'phone_terms': re.compile(r'电话|联系方式|手机号|号码'),
    'price_terms': re.compile(r'价格|价钱|消费|门票|房费|\d+\s*元'),
    'hours_terms': re.compile(r'营业时间|营业到|几点营业|几点关门'),
    'query_claim': re.compile(r'我(?:帮您|帮你|这边)?(?:查询|查了|查到|查一下|看了下|看了一下|搜索)'),
    'booking_claim': re.compile(r'(?:已|已经|帮您|帮你|给您|给你)(?:预订|订好|预定|预约|订了)'),
}


def flag_dialogue(dialog):
    flags = set()
    for message in dialog['messages']:
        role, text = message['role'], message['content']
        for name, pattern in PATTERNS.items():
            if name in ('query_claim', 'booking_claim') and role != 'sys':
                continue
            if pattern.search(text):
                flags.add(name)
    return sorted(flags)


def screen(data):
    rows, counts = [], Counter()
    for identifier in sorted(data):
        dialog = data[identifier]
        flags = flag_dialogue(dialog)
        counts.update(flags)
        rows.append({'dialogue_id': identifier, 'type': dialog.get('type', 'unknown'),
                     'messages': len(dialog['messages']), 'flags': flags,
                     'status': 'unreviewed_not_approved'})
    return rows, dict(counts)


def run(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest_bytes = (source / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    archive_path = source / 'train.json.zip'
    archive_digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if archive_digest != manifest.get('archive_sha256'):
        raise ValueError('Archive digest mismatch')
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ['train.json']:
            raise ValueError('Unexpected archive')
        raw = archive.read('train.json')
    if hashlib.sha256(raw).hexdigest() != manifest.get('train_json_sha256'):
        raise ValueError('Training JSON digest mismatch')
    rows, counts = screen(json.loads(raw.decode('utf8')))
    output.mkdir(parents=False)
    path = output / 'flags.jsonl'
    with path.open('x', encoding='utf8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    result = {'status': 'rule_flags_not_training_approval', 'archive_sha256': archive_digest,
              'source_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
              'flags_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'dialogues': len(rows), 'flagged_dialogues': sum(bool(row['flags']) for row in rows),
              'counts_by_flag': counts, 'test_split_read': False,
              'limitations': ['Flags are heuristic and are neither privacy determinations nor factual verification.',
                              'Unflagged dialogue is not approved or guaranteed safe.',
                              'No text is rewritten and no training is started.']}
    (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
