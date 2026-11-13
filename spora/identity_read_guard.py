"""Experimental exact-name check before accessing a projected memory key.

This verifies the extracted text against manually enrolled fingerprints. It
does not establish that an extracted span is complete or disambiguate homonyms.
It must not be represented as a general identity or access-control solution.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .identity_fingerprints import IdentityFingerprints


class IdentityReadGuard:
    def __init__(self, fingerprints, tokenizer):
        if not isinstance(fingerprints, IdentityFingerprints):
            raise TypeError('Explicit identity fingerprints required')
        self.fingerprints = fingerprints
        self.tokenizer = tokenizer

    def selected_name(self, question, selected):
        if (not isinstance(question, str) or not question
                or not isinstance(selected, torch.Tensor) or selected.ndim != 1
                or selected.dtype != torch.bool or not bool(selected.any())):
            return None
        text = self.tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
                                                  tokenize=False, add_generation_prompt=True)
        start = text.find(question)
        if start < 0 or text.find(question, start + 1) >= 0:
            return None
        encoded = self.tokenizer([text], return_tensors='pt', padding=True, add_special_tokens=False,
                                 return_offsets_mapping=True)
        offsets = encoded['offset_mapping']
        if offsets.ndim != 3 or offsets.shape[0] != 1 or offsets.shape[1] != len(selected) or offsets.shape[2] != 2:
            return None
        indices = selected.nonzero(as_tuple=True)[0].tolist()
        if indices[-1] - indices[0] + 1 != len(indices):
            return None
        chars = set()
        for token in indices:
            left, right = offsets[0, token].tolist()
            if left < start or right > start + len(question) or left >= right:
                return None
            chars.update(range(left - start, right - start))
        if not chars or max(chars) - min(chars) + 1 != len(chars):
            return None
        return ''.join(question[i] for i in sorted(chars))

    def authorize(self, question, selected, key):
        if type(key) is not int:
            return False
        name = self.selected_name(question, selected)
        if name is None:
            return False
        try:
            return self.fingerprints.lookup(name) == key
        except ValueError:
            return False


class ConsensusIdentityReadGuard(IdentityReadGuard):
    """Experimental veto requiring two independently trained span heads to agree.

    Heads share the same underlying model features, so agreement is not proof
    of an intact name, identity, authorization, or independence of errors.
    """

    def __init__(self, fingerprints, tokenizer, locator_state, threshold):
        super().__init__(fingerprints, tokenizer)
        if not isinstance(locator_state, dict) or 'weight' not in locator_state:
            raise ValueError('Secondary locator weights are required')
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError('Secondary locator threshold must be in [0, 1]')
        weight = locator_state['weight']
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.shape[0] != 1:
            raise ValueError('Invalid secondary locator shape')
        self.locator = nn.Linear(weight.shape[1], 1).eval()
        self.locator.load_state_dict(locator_state, strict=True)
        self.threshold = float(threshold)

    @torch.no_grad()
    def authorize_features(self, question, selected, key, features, masks):
        from scripts.deepseek_learned_name_span import single_span_mask
        if (not isinstance(features, torch.Tensor) or features.ndim != 4
                or features.shape[0] != 1 or features.shape[1] != 2
                or not isinstance(masks, torch.Tensor) or masks.ndim != 2 or masks.shape[0] != 1):
            return False
        vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],))
                             for i in range(2)], dim=-1).cpu()
        if vectors.shape[-1] != self.locator.in_features:
            return False
        scores = self.locator(vectors).squeeze(-1).sigmoid()[0]
        other, single = single_span_mask(scores, masks[0].cpu(), self.threshold)
        if not single or other.shape != selected.shape or not torch.equal(other, selected.cpu()):
            return False
        return super().authorize(question, selected, key)
