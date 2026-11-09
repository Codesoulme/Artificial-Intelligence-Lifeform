"""Verified, document-bounded access to hybrid-token candidate streams.

Loading a stream does not authorize or start training. The caller must enforce
source review and curriculum admission separately.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .document_stream import DocumentStream
from .language_bridge import BOS, EOS, PAD


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def open_packed_hybrid(directory, tokenizer_path, verification_path):
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf8'))
    verification = json.loads(Path(verification_path).read_text(encoding='utf8'))
    if (manifest.get('format') != 'spora-hybrid-packed-u16le-v1'
            or manifest.get('status') != 'candidate_not_approved_for_training'
            or verification.get('status') != 'packed_candidate_verified_not_training_approval'):
        raise ValueError('Unexpected candidate or verification status')
    keys = {'tokens_sha256': root / 'train.u16', 'index_sha256': root / 'index.jsonl',
            'tokenizer_sha256': Path(tokenizer_path)}
    for key, path in keys.items():
        actual = file_sha256(path)
        if actual != manifest.get(key) or actual != verification.get('input_sha256', {}).get(key):
            raise ValueError('Packed input hash mismatch: ' + key)
    if verification.get('input_sha256', {}).get('source_sha256') != manifest.get('source_sha256'):
        raise ValueError('Verification source mismatch')
    if Path(root / 'train.u16').stat().st_size != manifest['tokens_including_boundaries'] * 2:
        raise ValueError('Packed stream byte length mismatch')
    vocabulary = json.loads(Path(tokenizer_path).read_text(encoding='utf8'))
    if vocabulary.get('format') != 'spora-hybrid-utf8-v1' or vocabulary.get('vocab_size') != manifest.get('vocab_size'):
        raise ValueError('Tokenizer format or vocabulary mismatch')
    tokens = np.memmap(root / 'train.u16', dtype='<u2', mode='r')
    if np.any(tokens >= manifest['vocab_size']) or np.any(tokens == PAD):
        raise ValueError('Packed stream contains invalid or padding tokens')
    stream = DocumentStream(tokens, BOS, EOS, PAD)
    if (len(stream.starts) != manifest['documents'] or len(stream.starts) != verification.get('documents')
            or int(stream.cumulative[-1]) != manifest['within_document_targets']
            or len(tokens) != verification.get('tokens')):
        raise ValueError('Packed document count or target count mismatch')
    return stream
