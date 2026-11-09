"""Compare existing interfaces on identical representable development spans."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from spora.codepoint import CodepointLanguageBridge, CodepointVocabulary, UNK
from spora.growing import load
from spora.language_bridge import BOS, LanguageBridge
from spora.language_training import frozen_digest, load_interface


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def development_documents(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest.get('schema') != 'spora-text-v1':
        raise ValueError('Invalid manifest')
    docs = []
    for row in manifest['documents']:
        if row['split'] != 'development':
            continue
        path = (manifest_path.parent / row['path']).resolve()
        if not path.is_relative_to(manifest_path.parent) or digest(path) != row['sha256']:
            raise ValueError('Invalid development document path or digest')
        docs.append((row['path'], path.read_bytes().decode('utf-8-sig')))
    if not docs:
        raise ValueError('No development documents')
    return docs


def representable_spans(text, vocab):
    spans, current = [], []
    for char in text:
        if char in vocab.ids:
            current.append(char)
        elif current:
            spans.append(''.join(current))
            current = []
    if current:
        spans.append(''.join(current))
    return spans


@torch.no_grad()
def text_nll(model, state, token_ids, length):
    """Sum each content target once; BOS is context and EOS is excluded."""
    if length < 2 or not token_ids:
        raise ValueError('Nonempty text and context of at least two required')
    tokens = [BOS] + token_ids
    total = 0.0
    for start in range(0, len(tokens) - 1, length - 1):
        piece = torch.tensor([tokens[start:start + length]], dtype=torch.long,
                             device=next(model.parameters()).device)
        logits = model(piece, state)[:, :-1, :]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               piece[:, 1:].reshape(-1), reduction='sum')
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite loss')
        total += float(loss)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('individual', 'byte', 'codepoint', 'vocab', 'manifest', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--length', type=int, default=128)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    paths = [args.individual, args.byte, args.codepoint, args.vocab, args.manifest]
    hashes = {name: digest(getattr(args, name)) for name in ('individual', 'byte', 'codepoint', 'vocab', 'manifest')}
    byte_payload = torch.load(args.byte, map_location='cpu', weights_only=True)
    char_payload = torch.load(args.codepoint, map_location='cpu', weights_only=True)
    if byte_payload.get('format') != 'spora-byte-training-v2' or char_payload.get('format') != 'spora-codepoint-training-v1':
        raise ValueError('Expected trained byte and codepoint interfaces')
    if any(p['provenance']['manifest_sha256'] != hashes['manifest'] for p in (byte_payload, char_payload)):
        raise ValueError('Manifest mismatch')
    if char_payload['provenance']['vocab_sha256'] != hashes['vocab']:
        raise ValueError('Vocabulary mismatch')
    vocab = CodepointVocabulary.from_file(args.vocab)
    torch.set_num_threads(2)
    organism, state, _ = load(args.individual, args.device)
    before = frozen_digest(organism, state)
    if not 2 <= args.length <= organism.config.max_length:
        raise ValueError('Invalid context length')
    byte_model = LanguageBridge(organism, context_layers=byte_payload['context_layers']).to(args.device).eval()
    char_model = CodepointLanguageBridge(organism, vocab, context_layers=char_payload['context_layers']).to(args.device).eval()
    load_interface(byte_model, byte_payload, hashes['individual'])
    load_interface(char_model, char_payload, hashes['individual'])
    rows = []
    for path, text in development_documents(args.manifest):
        char_ids = vocab.encode(text)[1:-1]
        unknown = sum(token == UNK for token in char_ids)
        spans = representable_spans(text, vocab)
        row = {'path': path, 'utf8_bytes': len(text.encode('utf8')), 'characters': len(text),
               'unknown_characters': unknown, 'spans': len(spans),
               'scored_utf8_bytes': sum(len(span.encode('utf8')) for span in spans),
               'byte_nll': 0.0, 'codepoint_nll': 0.0}
        for span in spans:
            row['byte_nll'] += text_nll(byte_model, state, list(span.encode('utf8')), args.length)
            row['codepoint_nll'] += text_nll(char_model, state, vocab.encode(span)[1:-1], args.length)
        rows.append(row)
    selected = [row for row in rows if row['scored_utf8_bytes']]
    byte_count = sum(row['scored_utf8_bytes'] for row in selected)
    if not byte_count:
        raise ValueError('No representable development text')
    summary = {name: sum(row[name + '_nll'] for row in selected) / byte_count / math.log(2)
               for name in ('byte', 'codepoint')}
    if frozen_digest(organism, state) != before or any(digest(path) != hashes[name] for path, name in zip(paths, hashes)):
        raise AssertionError('Protected inputs changed')
    result = {'input_sha256': hashes, 'split': 'development', 'context_tokens': args.length,
              'documents': rows, 'included_documents': len(selected), 'included_utf8_bytes': byte_count,
              'bits_per_utf8_byte': summary, 'parent_and_inputs_unchanged': True,
              'limitations': ['Excludes EOS and unknown codepoints; each representable span resets to BOS.',
                              'Nonoverlapping windows reset context; equal token limits cover different amounts of text.',
                              'Existing runs differ in effective target exposure and compute.',
                              'Development likelihood does not establish chat quality.']}
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('included_documents', 'included_utf8_bytes', 'bits_per_utf8_byte')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
