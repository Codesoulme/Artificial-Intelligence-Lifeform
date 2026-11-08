"""Strict P91 source gate, separate from the synthetic training primitives."""
import json
from pathlib import Path

from .source_review import file_sha256, verify_training_source_reviews

P54 = 'reports/P54-story-focus-source-review.json'
P54_SHA = 'd67ab061b83e45f096af46fd1f574bd8d1a4dc38d21b4fe2261578a754167a92'
P28 = 'reports/P28-stories-content-review.json'
P28_SHA = 'b3da3e063d25616922aa482a67d107258e39a2b1af63ff205b44a4cf46e2054f'
DATA = 'runs/p28-stories-data/candidate-v3'
MANIFEST_SHA = 'fd57c0ffa3878c1aef3e762d4c4153672c7c899088f6369ceec33663243af5bd'
AUDIT_SHA = '4cfb611228c59f634b3760e211175e250892a0188e0c7e41c7ae51b5efbf8e6b'


def verify_p91_inputs(root, plan_path, approval_path):
    """Verify a separately reviewed plan and the original immutable source gate.

    Approval must bind the exact plan file, not merely declare generic approval.
    This function neither creates an approval nor starts training.
    """
    root, plan_path, approval_path = Path(root), Path(plan_path), Path(approval_path)
    plan = json.loads(plan_path.read_text(encoding='utf8'))
    approval = json.loads(approval_path.read_text(encoding='utf8'))
    if approval.get('approved_for_pilot') is not True or approval.get('plan_sha256') != file_sha256(plan_path):
        raise ValueError('Explicit review of the exact P91 plan required')
    if approval.get('architectures') != ['memory_bridge', 'standard_causal'] or approval.get('initialization') != 'paired_random' or approval.get('seeds') != plan.get('seeds'):
        raise ValueError('Review does not cover both architectures and all seeds')
    if not isinstance(plan.get('seeds'), list) or len(plan['seeds']) < 2 or len(set(plan['seeds'])) != len(plan['seeds']) or any(type(s) is not int or not 0 <= s < 2**64 for s in plan['seeds']):
        raise ValueError('At least two distinct initialization seeds required')
    steps = plan.get('schedule_steps')
    if type(steps) is not int or not 1 <= steps <= 100:
        raise ValueError('P54 permits at most 100 updates')
    if plan.get('data') != DATA or plan.get('target_shares') != {'story': 1.0} or plan.get('focus_every') != 0 or plan.get('initial_language') is not None:
        raise ValueError('Only randomly initialized, story-only training is allowed')
    if file_sha256(root/P54) != P54_SHA or file_sha256(root/P28) != P28_SHA or file_sha256(root/DATA/'manifest.json') != MANIFEST_SHA:
        raise ValueError('Original review or manifest changed')
    p54 = json.loads((root/P54).read_text(encoding='utf8'))
    p28 = json.loads((root/P28).read_text(encoding='utf8'))
    if p54.get('story_source_manifest_sha256') != MANIFEST_SHA or p54.get('story_original_review_sha256') != P28_SHA or p28.get('data_manifest_sha256') != MANIFEST_SHA or p28.get('approved_for_pilot') is not True:
        raise ValueError('Original content review binding failed')
    gate = dict(data=DATA, curriculum=DATA, target_shares={'story': 1.0},
                schedule_steps=steps, focus_every=steps+1,
                training_source_reviews={'story': {'path': P54, 'sha256': P54_SHA}})
    verified = verify_training_source_reviews(root, gate)
    manifest = json.loads((root/DATA/'manifest.json').read_text(encoding='utf8'))
    for name in ('pretrain-train.bin', 'story-development.bin', 'tokenizer.json'):
        if file_sha256(root/DATA/name) != manifest['files'][name]:
            raise ValueError('Data artifact changed: '+name)
    audit_path = root/DATA/'lineage-audit.json'
    if file_sha256(audit_path) != AUDIT_SHA:
        raise ValueError('Isolation audit changed')
    audit = json.loads(audit_path.read_text(encoding='utf8'))
    if not audit.get('passed') or not audit.get('new_train_does_not_overlap_new_held_under_declared_rules') or audit.get('manifest_sha256', {}).get(DATA) != MANIFEST_SHA:
        raise ValueError('Missing or stale isolation audit')
    for name in ('pretrain-train.bin', 'story-development.bin', 'tokenizer.json'):
        if audit.get('data_artifact_sha256', {}).get(name) != manifest['files'][name]:
            raise ValueError('Isolation audit covers different data')
    return {'plan_sha256': file_sha256(plan_path), 'approval_sha256': file_sha256(approval_path),
            'audit_sha256': file_sha256(audit_path), 'manifest_sha256': MANIFEST_SHA,
            'source_reviews': verified}
