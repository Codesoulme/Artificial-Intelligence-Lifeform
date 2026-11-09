"""Lossless character vocabulary with UTF-8 byte fallback.

The format has distinct semantics from the older codepoint tokenizer even
though their character token numbers can coincide. Never reuse an interface
checkpoint without explicitly binding it to this format and vocabulary.
"""
import json
from pathlib import Path

from .language_bridge import BOS, EOS, PAD

CHAR_BASE = 259
FORMAT = 'spora-hybrid-utf8-v1'


class HybridTokenizer:
    def __init__(self, symbols):
        self.symbols = tuple(symbols)
        if any(not isinstance(c, str) or len(c) != 1 for c in self.symbols):
            raise ValueError('Vocabulary entries must be single Unicode characters')
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError('Duplicate vocabulary symbol')
        for char in self.symbols:
            char.encode('utf8')
        self.ids = {char: CHAR_BASE + i for i, char in enumerate(self.symbols)}

    @property
    def size(self):
        return CHAR_BASE + len(self.symbols)

    def encode(self, text):
        if not isinstance(text, str):
            raise TypeError('Text required')
        tokens = [BOS]
        for char in text:
            if char in self.ids:
                tokens.append(self.ids[char])
            else:
                tokens.extend(char.encode('utf8'))
        return tokens + [EOS]

    def decode(self, tokens):
        pieces = []
        pending = bytearray()

        def flush():
            if pending:
                pieces.append(pending.decode('utf8', errors='strict'))
                pending.clear()

        for token in tokens:
            if type(token) is not int:
                raise TypeError('Integer tokens required')
            if 0 <= token < 256:
                pending.append(token)
            elif CHAR_BASE <= token < self.size:
                flush()
                pieces.append(self.symbols[token - CHAR_BASE])
            elif token in (BOS, EOS, PAD):
                flush()
            else:
                raise ValueError('Token outside vocabulary')
        flush()
        return ''.join(pieces)

    def payload(self):
        return {'format': FORMAT, 'symbols': list(self.symbols), 'vocab_size': self.size}

    @classmethod
    def from_file(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf8'))
        if data.get('format') != FORMAT or not isinstance(data.get('symbols'), list):
            raise ValueError('Invalid hybrid vocabulary')
        vocab = cls(data['symbols'])
        if data.get('vocab_size') != vocab.size:
            raise ValueError('Vocabulary size mismatch')
        return vocab
