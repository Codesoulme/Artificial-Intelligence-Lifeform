"""Run existing pre-read guard audit with an isolated candidate locator."""
import json
from pathlib import Path

import torch

from scripts import deepseek_spora_pre_read_guard_audit as audit
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_spora_boundary_train_pilot import OUTPUT as CANDIDATE


def main():
    path = CANDIDATE / 'locator.pt'
    candidate_sha = runtime.sha(path)
    report = json.loads((CANDIDATE / 'result.json').read_text(encoding='utf-8'))
    if candidate_sha != report['locator_sha256'] or report['source_checkpoint_sha256'] != runtime.sha(runtime.CHECKPOINT):
        raise ValueError('Locator provenance mismatch')
    candidate = torch.load(path, map_location='cpu', weights_only=True)
    prior_loader = runtime.load_chat

    def load_with_candidate(*args, **kwargs):
        chat = prior_loader(*args, **kwargs)
        if candidate['feature_variant_sha256'] != chat.router['feature_variant_sha256']:
            raise ValueError('Locator feature variant mismatch')
        chat.router = dict(chat.router)
        chat.router['locator_state'] = candidate['state']
        chat.router['locator_threshold'] = candidate['threshold']
        return chat

    audit.OUTPUT = Path(audit.p.ROOT / 'runs/deepseek-spora-candidate-pre-read-guard-audit-20261112.json')
    try:
        runtime.load_chat = load_with_candidate
        audit.main()
    finally:
        runtime.load_chat = prior_loader
    if runtime.sha(path) != candidate_sha:
        raise ValueError('Candidate locator changed')


if __name__ == '__main__':
    main()
