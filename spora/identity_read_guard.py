"""Experimental exact-name check before accessing a projected memory key.

This verifies the extracted text against manually enrolled fingerprints. It
does not establish that an extracted span is complete or disambiguate homonyms.
It must not be represented as a general identity or access-control solution.
"""
import torch

from .identity_fingerprints import IdentityFingerprints


class IdentityReadGuard:
    def __init__(self, fingerprints, tokenizer):
        if not isinstance(fingerprints, IdentityFingerprints):
            raise TypeError('Explicit identity fingerprints required')
        self.fingerprints = fingerprints
        self.tokenizer = tokenizer

    def authorize(self, question, selected, key):
        if (not isinstance(question, str) or not question or type(key) is not int
                or not isinstance(selected, torch.Tensor) or selected.ndim != 1
                or selected.dtype != torch.bool or not bool(selected.any())):
            return False
        text = self.tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
                                                  tokenize=False, add_generation_prompt=True)
        start = text.find(question)
        if start < 0 or text.find(question, start + 1) >= 0:
            return False
        encoded = self.tokenizer([text], return_tensors='pt', padding=True, add_special_tokens=False,
                                 return_offsets_mapping=True)
        offsets = encoded['offset_mapping']
        if offsets.ndim != 3 or offsets.shape[0] != 1 or offsets.shape[1] != len(selected) or offsets.shape[2] != 2:
            return False
        indices = selected.nonzero(as_tuple=True)[0].tolist()
        if indices[-1] - indices[0] + 1 != len(indices):
            return False
        chars = set()
        for token in indices:
            left, right = offsets[0, token].tolist()
            if left < start or right > start + len(question) or left >= right:
                return False
            chars.update(range(left - start, right - start))
        if not chars or max(chars) - min(chars) + 1 != len(chars):
            return False
        name = ''.join(question[i] for i in sorted(chars))
        try:
            return self.fingerprints.lookup(name) == key
        except ValueError:
            return False
