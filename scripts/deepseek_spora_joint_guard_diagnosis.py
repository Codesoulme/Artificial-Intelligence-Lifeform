"""Record why the research-only joint-span guard accepted or vetoed a read."""
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
from scripts.deepseek_spora_joint_guard_audit import CANDIDATE, JointSpanReadGuard
from scripts.deepseek_spora_joint_span_pilot import JointSpan

OUTPUT = ROOT / 'runs/deepseek-spora-joint-guard-diagnosis-20261113.json'


def main():
    path = CANDIDATE / 'span.pt'
    info = json.loads((CANDIDATE / 'result.json').read_text(encoding='utf-8'))
    digest = runtime.sha(path)
    if digest != info['weights_sha256']:
        raise ValueError('Joint head checksum mismatch')
    candidate = torch.load(path, map_location='cpu', weights_only=True)
    old_loader, old_guard, old_cases, old_output = runtime.load_chat, audit.RecordingGuard, audit.CASES, audit.OUTPUT

    def checked_loader(*args, **kwargs):
        chat = old_loader(*args, **kwargs)
        if candidate['feature_variant_sha256'] != chat.router.get('feature_variant_sha256'):
            raise ValueError('Feature variant mismatch')
        return chat

    def factory(fingerprints, tokenizer):
        head = JointSpan(candidate['state']['start.weight'].shape[1], width=candidate['width'])
        head.load_state_dict(candidate['state'], strict=True)

        class RecordingJoint(JointSpanReadGuard):
            @torch.no_grad()
            def authorize_features(self, question, selected, key, features, masks):
                authorized = super().authorize_features(question, selected, key, features, masks)
                vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],))
                                     for i in range(2)], dim=-1).cpu()[0]
                index = int(self.head(vectors, masks[0].cpu()).argmax())
                primary = old_guard(fingerprints, tokenizer)
                primary.authorize(question, selected, key)
                secondary = None
                if index:
                    left, right = divmod(index - 1, len(selected))
                    span = torch.zeros_like(selected, device='cpu')
                    span[left:right + 1] = True
                    secondary = old_guard(fingerprints, tokenizer)
                    secondary.authorize(question, span, key)
                    reason = ('both_match' if torch.equal(span, selected.cpu()) else 'span_disagreement')
                else:
                    reason = 'joint_abstained'
                if reason == 'both_match' and not authorized:
                    reason = 'identity_rejected'
                self.last = {'primary': primary.last, 'joint': getattr(secondary, 'last', None),
                             'reason': reason, 'authorized': authorized}
                return authorized
        return RecordingJoint(fingerprints, tokenizer, head)

    try:
        runtime.load_chat, audit.RecordingGuard = checked_loader, factory
        audit.CASES, audit.OUTPUT = list(VARIED_CASES) + list(INTENT_CASES), OUTPUT
        audit.main()
    finally:
        runtime.load_chat, audit.RecordingGuard = old_loader, old_guard
        audit.CASES, audit.OUTPUT = old_cases, old_output
    if runtime.sha(path) != digest:
        raise ValueError('Candidate changed')


if __name__ == '__main__':
    main()
