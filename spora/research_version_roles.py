"""Learn clause roles before choosing a version from a verified timeline."""

import re
import unicodedata

import torch
from torch import nn
from torch.nn import functional as F

from .research_version_scope import _DATE_HINT, _ORDINAL_HINT, choose_scoped_version
from .research_version_timeline import VersionQuery, select_version


NEUTRAL, SEEK, EXCLUDE, DENY = range(4)
CURRENT, FIRST, PREVIOUS, EXPIRED, UNSPECIFIED = range(5)
ABSTAIN = 3
_CLAUSES = re.compile(r'[^,，;；。?!？！]+')


def clause_spans(question):
    return [match.span() for match in _CLAUSES.finditer(question)]


class VersionRoleHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(width, 64), nn.GELU())
        self.roles = nn.Linear(64, 4)
        self.versions = nn.Linear(64, 5)

    def forward(self, clauses):
        vectors = self.project(F.layer_norm(clauses.float(), (clauses.shape[-1],)))
        return self.roles(vectors), self.versions(vectors)


def version_plan(roles, versions):
    requested = {version for role, version in zip(roles, versions, strict=True)
                 if role == SEEK and version != UNSPECIFIED}
    excluded = {version for role, version in zip(roles, versions, strict=True) if role == EXCLUDE}
    if (DENY in roles or len(requested) > 1 or requested & excluded
            or (EXPIRED in excluded and requested & {FIRST, PREVIOUS})):
        return ABSTAIN
    if requested:
        choice = next(iter(requested))
        return choice if choice < ABSTAIN else ABSTAIN
    if excluded:
        return CURRENT if excluded == {EXPIRED} and SEEK in roles else ABSTAIN
    return None


def apply_version_plan(question, revisions, today, legacy, plan):
    """Preserve date coordinates and outer read restrictions before applying a plan."""
    slot, reason = choose_scoped_version(question, revisions, today, legacy)
    text = unicodedata.normalize('NFKC', question)
    if (plan is None or reason == 'outside_lookup_scope'
            or _DATE_HINT.search(text) or _ORDINAL_HINT.search(text)):
        return slot, reason
    if plan == ABSTAIN:
        return None, 'roles_abstain'
    requests = (VersionQuery('current'), VersionQuery('ordinal', 1), VersionQuery('previous'))
    selected = select_version(revisions, requests[plan], today)
    return selected, 'role_plan' if selected is not None else 'no_matching_version'
