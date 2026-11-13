"""Compare default routing with a dual-locator pre-read veto on temporary state."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from scripts import deepseek_spora_pre_read_guard_audit as audit
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_learned_name_span import single_span_mask
from spora.identity_read_guard import ConsensusIdentityReadGuard

CANDIDATE = ROOT / 'runs/deepseek-spora-cluener-locator-retrained-20261111'
OUTPUT = ROOT / 'runs/deepseek-spora-consensus-identity-guard-audit-20261113.json'


def main(*, cases=None, output=OUTPUT):
    path = CANDIDATE / 'locator.pt'
    info = json.loads((CANDIDATE / 'result.json').read_text(encoding='utf-8'))
    digest = runtime.sha(path)
    if digest != info['locator_sha256']:
        raise ValueError('Secondary locator checksum mismatch')
    candidate = torch.load(path, map_location='cpu', weights_only=True)
    loader = runtime.load_chat
    original_guard = audit.RecordingGuard
    original_cases = audit.CASES
    original_output = audit.OUTPUT

    def checked_loader(*args, **kwargs):
        chat = loader(*args, **kwargs)
        if candidate.get('feature_variant_sha256') != chat.router.get('feature_variant_sha256'):
            raise ValueError('Secondary locator feature mismatch')
        return chat

    def guard_factory(fingerprints, tokenizer):
        class RecordingConsensus(ConsensusIdentityReadGuard):
            def authorize_features(self, question, selected, key, features, masks):
                result = super().authorize_features(question, selected, key, features, masks)
                vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],))
                                     for i in range(2)], dim=-1).cpu()
                with torch.no_grad():
                    other, single = single_span_mask(
                        self.locator(vectors).squeeze(-1).sigmoid()[0], masks[0].cpu(), self.threshold)
                first_probe = original_guard(fingerprints, tokenizer)
                first_probe.authorize(question, selected, key)
                second_probe = original_guard(fingerprints, tokenizer)
                if single:
                    second_probe.authorize(question, other, key)
                self.last = {'primary': first_probe.last,
                             'secondary': getattr(second_probe, 'last', None),
                             'secondary_single': single, 'authorized': result}
                return result
        return RecordingConsensus(fingerprints, tokenizer, candidate['state'], candidate['threshold'])

    try:
        runtime.load_chat = checked_loader
        audit.RecordingGuard = guard_factory
        audit.OUTPUT = Path(output)
        audit.CASES = list(cases) if cases is not None else list(original_cases) + [
            ('小林子现在想喝哪种饮料？', None),
            ('小林娜现在想喝哪种饮料？', None),
            ('林小林现在想喝哪种饮料？', None),
            ('你知道阿宁现在想喝哪种饮料吗？', 1),
        ]
        audit.main()
    finally:
        runtime.load_chat = loader
        audit.RecordingGuard = original_guard
        audit.CASES = original_cases
        audit.OUTPUT = original_output
    if runtime.sha(path) != digest:
        raise ValueError('Secondary locator changed')


if __name__ == '__main__':
    main()
