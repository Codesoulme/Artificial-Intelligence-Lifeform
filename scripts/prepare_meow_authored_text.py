"""Prepare one author-licensed work and its essays as one provenance group."""
import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

PROSE_SHA = 'c4147980bdc45c4068a3f94094b2a26f7020e100ee734e06c2d12428e6c55f4b'
DEEP_SHA = '453282ed483dc89464260a98f2fa460ad03b2dfac3417b68849e7701c6e749cc'
VOCAB_SHA = 'b3629001784896437c4aa9f279f6b4cdc5a534be191b537009e7293f971fda20'
REVISION = 'f6f8fcbd9f9bc9d7686013705a2bc69c66e7bea5'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(prose_path, essays_path, tokenizer_path, output):
    prose_path, essays_path, tokenizer_path, output = map(Path, (prose_path, essays_path, tokenizer_path, output))
    if output.exists():
        raise FileExistsError(output)
    prose, essays, vocab = (p.read_bytes() for p in (prose_path, essays_path, tokenizer_path))
    if (sha(prose), sha(essays), sha(vocab)) != (PROSE_SHA, DEEP_SHA, VOCAB_SHA):
        raise ValueError('Source or tokenizer digest mismatch')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    records = [json.loads(line) for line in essays.decode('utf8').splitlines() if line.strip()]
    if len(records) != 5:
        raise ValueError('Expected five authored essays')
    texts = [('original', prose.decode('utf8'))]
    for index, row in enumerate(records, 1):
        if (set(row) != {'id', 'title', 'body'} or row['id'] != f'meow-v2-deepdive-{index}'
                or not isinstance(row['title'], str) or not isinstance(row['body'], list)
                or not row['body'] or any(not isinstance(p, str) or not p.strip() for p in row['body'])):
            raise ValueError('Invalid authored essay structure')
        texts.append((row['id'], '\n'.join(row['body'])))
    rows = []
    for name, text in texts:
        if not text.strip() or '<|' in text or '\ufffd' in text:
            raise ValueError('Invalid authored text')
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if not ids or tokenizer.decode(ids, skip_special_tokens=False) != text:
            raise ValueError('Tokenizer roundtrip mismatch')
        rows.append({'id': name, 'work_group': 'meow-v2-one-work', 'text': text,
                     'text_sha256': sha(text.encode('utf8')), 'characters': len(text),
                     'content_tokens': len(ids), 'targets_with_eos': len(ids) + 1})
    raw = ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in rows).encode('utf8')
    result = {'status': 'prepared_for_review_not_training', 'source_revision': REVISION,
              'source_repository': 'https://github.com/tingyulingfeng/MeowMeow-Kitchen',
              'author': '听雨聆风的小厨房', 'declared_license': 'CC BY 4.0',
              'license_url': 'https://creativecommons.org/licenses/by/4.0/',
              'sources': {'prose_sha256': PROSE_SHA, 'deep_dive_sha256': DEEP_SHA,
                          'tokenizer_sha256': VOCAB_SHA},
              'transformation': 'Prose unchanged; each authored essay joins its ordered body strings with newline; titles omitted.',
              'excluded': 'Analysis and counterfactual fields excluded, including declared generated examples.',
              'work_groups': 1, 'documents': len(rows), 'characters': sum(r['characters'] for r in rows),
              'content_tokens': sum(r['content_tokens'] for r in rows),
              'targets_with_eos': sum(r['targets_with_eos'] for r in rows),
              'output_sha256': sha(raw), 'test_split_read': False,
              'limitations': ['One author and work; essays reference the same work and are not independent.',
                              'Narrative first-person voice is not the model\'s lived memory.',
                              'No factual, comprehensive source, historical or final-test overlap certification.',
                              'Small material cannot sustain general Chinese training without heavy repetition.']}
    output.mkdir(parents=False)
    (output / 'candidate.jsonl').write_bytes(raw)
    (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prose', required=True)
    parser.add_argument('--essays', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.prose, args.essays, args.tokenizer, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
