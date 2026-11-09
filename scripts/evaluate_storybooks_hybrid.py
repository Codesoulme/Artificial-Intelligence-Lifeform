"""Evaluate existing hybrid checkpoints on previously reviewed Storybooks text."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import torch

from scripts.compare_byte_codepoint_dev import text_nll
from spora.growing import load
from spora.hybrid_tokenizer import HybridLanguageBridge, HybridTokenizer
from spora.language_training import frozen_digest, load_interface
from spora.train_hybrid import sha


def reviewed_documents(root):
    root = Path(root)
    names = ('page-text-verification-v2.json', 'quality-review-a.json', 'quality-review-b.json')
    records = [json.loads((root / name).read_text(encoding='utf8')) for name in names]
    reviews = records[1]['stories'] + records[2]['reviews']
    by_story = {row['story']: row for row in reviews}
    if len(reviews) != 29 or len(by_story) != 29:
        raise ValueError('Review coverage mismatch')
    docs = []
    for item in records[0]['stories']:
        if item['status'] != 'matched_pending_quality_and_rights_review':
            continue
        review = by_story[item['story']]
        if review.get('decision', review.get('verdict')) != '通过':
            continue
        path = root / 'raw-v1' / (item['story'] + '.md')
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if (digest != item['source_sha256'] or digest != review['source_sha256']
                or not item['text_matches_page'] or not item['allowed_page_license']):
            raise ValueError('Story verification mismatch')
        sections = re.split(r'(?m)^##\s*$', raw.decode('utf-8-sig'))
        count = item['paragraphs']
        if len(sections) < count + 2:
            raise ValueError('Missing body or attribution section')
        text = '\n'.join(section.strip() for section in sections[1:count + 1])
        if not text:
            raise ValueError('Empty story')
        docs.append({'story': item['story'], 'source_sha256': digest, 'text': text})
    if len(docs) != 20:
        raise ValueError('Expected 20 previously reviewed stories')
    return docs, {name: sha(root / name) for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--individual', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--checkpoint', action='append', required=True)
    parser.add_argument('--storybooks-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--length', type=int, default=128)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    parent_sha, vocab_sha = sha(args.individual), sha(args.tokenizer)
    checkpoint_hashes = [sha(path) for path in args.checkpoint]
    docs, review_hashes = reviewed_documents(args.storybooks_root)
    tokenizer = HybridTokenizer.from_file(args.tokenizer)
    torch.set_num_threads(2)
    organism, state, _ = load(args.individual, args.device)
    if not 2 <= args.length <= organism.config.max_length:
        raise ValueError('Invalid context length')
    before = frozen_digest(organism, state)
    results = []
    for path, checkpoint_sha in zip(args.checkpoint, checkpoint_hashes):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('format') != 'spora-hybrid-training-v1' or payload['provenance']['vocab_sha256'] != vocab_sha:
            raise ValueError('Checkpoint or tokenizer mismatch')
        model = HybridLanguageBridge(organism, tokenizer, vocab_sha, payload['context_layers']).to(args.device)
        if payload['provenance'].get('train_backbone_copy') is True:
            model.enable_language_block_copies()
        model.eval()
        load_interface(model, payload, parent_sha)
        rows = []
        for doc in docs:
            text = doc['text']
            rows.append({'story': doc['story'], 'source_sha256': doc['source_sha256'],
                         'characters': len(text), 'utf8_bytes': len(text.encode('utf8')),
                         'content_targets': len(tokenizer.encode(text)) - 2,
                         'total_nll': text_nll(model, state, tokenizer.encode(text)[1:-1], args.length)})
        total_bytes = sum(row['utf8_bytes'] for row in rows)
        results.append({'checkpoint_sha256': checkpoint_sha, 'documents': rows,
                        'bits_per_utf8_byte': sum(row['total_nll'] for row in rows) / total_bytes / math.log(2),
                        'total_utf8_bytes': total_bytes})
    if (sha(args.individual) != parent_sha or sha(args.tokenizer) != vocab_sha
            or any(sha(path) != digest for path, digest in zip(args.checkpoint, checkpoint_hashes))
            or frozen_digest(organism, state) != before):
        raise AssertionError('Protected inputs changed')
    result = {'individual_sha256': parent_sha, 'tokenizer_sha256': vocab_sha, 'review_sha256': review_hashes,
              'length': args.length, 'story_count': len(docs), 'results': results,
              'test_split_read': False, 'parent_and_inputs_unchanged': True,
              'limitations': ['Previously reviewed exploratory stories, not independent final evaluation.',
                              'Canonical token path NLL excludes EOS; alternative token paths are not summed.',
                              'Context resets at each fixed window; text is not a chat assessment.',
                              'Historical and semantic overlap are not exhaustively excluded.']}
    with output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps([{'checkpoint_sha256': row['checkpoint_sha256'], 'bits_per_utf8_byte': row['bits_per_utf8_byte'],
                       'total_utf8_bytes': row['total_utf8_bytes']} for row in results]))


if __name__ == '__main__':
    main()
