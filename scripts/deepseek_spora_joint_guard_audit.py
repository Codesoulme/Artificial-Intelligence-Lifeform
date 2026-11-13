"""Experimental joint-span veto in the real read path; never a default guard."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch.nn import functional as F

from scripts import deepseek_spora_pre_read_guard_audit as audit
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_spora_beverage_intent_distraction_audit import CASES as INTENT_CASES
from scripts.deepseek_spora_consensus_identity_varied_audit import CASES as VARIED_CASES
from scripts.deepseek_spora_joint_span_pilot import JointSpan
from spora.identity_read_guard import ConsensusIdentityReadGuard, IdentityReadGuard

CANDIDATE = ROOT / 'runs/deepseek-spora-joint-span-diverse-pilot-20261113'
OUTPUT = ROOT / 'runs/deepseek-spora-joint-guard-audit-20261113.json'


class JointSpanReadGuard(ConsensusIdentityReadGuard):
    def __init__(self, fingerprints, tokenizer, head):
        # The runtime dispatches feature-aware checks by this parent type.
        IdentityReadGuard.__init__(self, fingerprints, tokenizer)
        self.head = head.eval()

    @torch.no_grad()
    def authorize_features(self, question, selected, key, features, masks):
        if (not isinstance(selected, torch.Tensor) or selected.ndim != 1 or selected.dtype != torch.bool
                or not isinstance(features, torch.Tensor) or features.ndim != 4 or features.shape[:2] != (1, 2)
                or not isinstance(masks, torch.Tensor) or masks.shape != (1, len(selected))
                or masks.dtype != torch.bool or features.shape[2] != len(selected)
                or not bool(torch.isfinite(features).all())):
            return False
        vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],))
                             for i in range(2)], dim=-1).cpu()[0]
        if vectors.shape[-1] != self.head.start.in_features:
            return False
        logits = self.head(vectors, masks[0].cpu())
        if not bool(torch.isfinite(logits).all()):
            return False
        index = int(logits.argmax())
        if index == 0:
            return False
        left, right = divmod(index - 1, len(selected))
        if left > right or right >= len(selected) or not bool(masks[0, left:right + 1].all()):
            return False
        other = torch.zeros_like(selected, device='cpu')
        other[left:right + 1] = True
        if not torch.equal(other, selected.cpu()):
            return False
        return super().authorize(question, selected, key)


def main():
    path = CANDIDATE / 'span.pt'
    info = json.loads((CANDIDATE / 'result.json').read_text(encoding='utf-8'))
    candidate_sha = runtime.sha(path)
    if candidate_sha != info['weights_sha256']:
        raise ValueError('Joint head checksum mismatch')
    candidate = torch.load(path, map_location='cpu', weights_only=True)
    original_loader, original_factory = runtime.load_chat, audit.RecordingGuard
    original_cases, original_output = audit.CASES, audit.OUTPUT

    def checked_loader(*args, **kwargs):
        chat = original_loader(*args, **kwargs)
        if candidate['feature_variant_sha256'] != chat.router.get('feature_variant_sha256'):
            raise ValueError('Joint head feature mismatch')
        return chat

    def factory(fingerprints, tokenizer):
        head = JointSpan(candidate['state']['start.weight'].shape[1], width=candidate['width'])
        head.load_state_dict(candidate['state'], strict=True)

        class RecordingJoint(JointSpanReadGuard):
            def authorize_features(self, question, selected, key, features, masks):
                decision = super().authorize_features(question, selected, key, features, masks)
                probe = original_factory(fingerprints, tokenizer)
                probe.authorize(question, selected, key)
                self.last = {'primary': probe.last, 'authorized': decision}
                return decision
        return RecordingJoint(fingerprints, tokenizer, head)

    try:
        runtime.load_chat = checked_loader
        audit.RecordingGuard = factory
        audit.CASES = list(VARIED_CASES) + list(INTENT_CASES)
        audit.OUTPUT = OUTPUT
        audit.main()
    finally:
        runtime.load_chat = original_loader
        audit.RecordingGuard = original_factory
        audit.CASES, audit.OUTPUT = original_cases, original_output
    if runtime.sha(path) != candidate_sha:
        raise ValueError('Joint head changed')


if __name__ == '__main__':
    main()
