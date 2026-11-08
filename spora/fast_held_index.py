"""Independent, early-exit equivalent of the P28 lexical index.

This is not connected to any screening pipeline. Indexes must remain separate.
"""
from collections import Counter, defaultdict
from itertools import chain
import hashlib
import re
import unicodedata


def _normalized(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).lower()


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _signatures(key):
    return {hashlib.blake2b(key[i:i + 16].encode(), digest_size=8).digest()
            for i in range(0, max(0, len(key) - 15), 16)}


class FastHeldIndex:
    """Same insertion and reason results as prepare_p28_prose.HeldIndex."""

    def __init__(self):
        self.exact = set()
        self.groups = set()
        self.sizes = []
        self.index = defaultdict(list)

    def add(self, text, extra_group=None, extra_digest=None):
        key = _normalized(text)
        self.exact.add(_digest(key))
        self.groups.add(_digest(key[:96]))
        if extra_group:
            self.groups.add(extra_group)
        if extra_digest:
            self.exact.add(extra_digest)
        sigs = _signatures(key)
        number = len(self.sizes)
        self.sizes.append(len(sigs))
        for sig in sigs:
            self.index[sig].append(number)

    def reason(self, text):
        key = _normalized(text)
        if _digest(key) in self.exact:
            return "held_exact"
        if _digest(key[:96]) in self.groups:
            return "held_leading96"
        sigs = _signatures(key)
        n = len(sigs)
        if n < 8:
            return None
        # Every qualifying document needs at least seven distinct query
        # signatures. Check a batch first, then finish the full count if needed.
        postings = sorted((self.index[s] for s in sigs if s in self.index),
                          key=len, reverse=True)
        shared = Counter(chain.from_iterable(postings[:8]))

        def qualifies():
            return any(min(n, self.sizes[i]) >= 8 and
                       count / min(n, self.sizes[i]) >= .8
                       for i, count in shared.items())

        if qualifies():
            return "held_sampled_lexical"
        shared.update(chain.from_iterable(postings[8:]))
        if qualifies():
            return "held_sampled_lexical"
        return None
