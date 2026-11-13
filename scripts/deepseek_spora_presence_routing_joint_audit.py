"""Audit language routing, identity check and native presence together.

The test uses manually enrolled identities and numeric fact writes; the guard
is opt-in and is not a proof of identity or a general safety boundary.
"""
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_named_binding_probe import KEYS
from scripts.deepseek_spora_pre_read_guard_audit import RecordingGuard
from spora.identity_fingerprints import IdentityFingerprints

OUTPUT = ROOT / 'runs/deepseek-spora-presence-routing-joint-20261114.json'
NAMES = ('小林', '阿宁', '小雨', '阿哲')
VALUES = (6, 1, 2, 3)
CASES = (
    ('known', '请告诉我小林目前想喝哪种饮料？', 0),
    ('known', '阿宁现在喜欢什么饮品？', 1),
    ('known', '小雨当前想喝什么？请说饮料名。', 2),
    ('known', '对于阿哲，记录里现在的饮料是什么？', 3),
    ('known', '小林现在喝什么？只回答名称。', 0),
    ('known', '请查阿宁的饮料偏好。', 1),
    ('near_name', '请告诉我小林华目前想喝哪种饮料？', None),
    ('near_name', '阿宁悦现在喜欢什么饮品？', None),
    ('near_name', '小雨晴当前想喝什么？请说饮料名。', None),
    ('near_name', '对于阿哲轩，记录里现在的饮料是什么？', None),
    ('unknown', '小王现在喜欢什么饮品？', None),
    ('unknown', '请查小雪的饮料偏好。', None),
    ('wrong_relation', '小林喜欢什么颜色？', None),
    ('wrong_relation', '请说出阿宁最喜欢什么运动。', None),
    ('multi', '小林和阿宁现在分别喜欢什么饮料？', None),
    ('multi', '小雨与阿哲各自想喝什么饮品？', None),
    ('ordinary', '请写一句鼓励学习的话。', None),
    ('ordinary', '喝水有什么好处？', None),
)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Base digest mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix='spora-route-audit-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        source_state = chat.state
        source_digest = p.frozen_digest(chat.organism, source_state)
        device = next(chat.organism.parameters()).device
        fingerprints = IdentityFingerprints(chat.organism.config.key_count).to(device)
        state = p.fresh_controlled_state(chat.organism)
        for index, name in enumerate(NAMES):
            fingerprints.bind(KEYS[index], name)
            state, decision = p.observe_sourced(chat.organism, state, KEYS[index], VALUES[index], 'direct', 1.0)
            if not decision['current_write']:
                raise ValueError('Temporary controlled write was not accepted')
        chat.state = state
        state_digest = p.frozen_digest(chat.organism, state)
        guard = RecordingGuard(fingerprints, chat.tokenizer)
        for group, question, expected_index in CASES:
            results = {}
            for label, policy in (('baseline', None), ('guarded', guard)):
                guard.last = None
                with patch.object(p.runtime, 'current_memory_status', wraps=p.runtime.current_memory_status) as status, \
                     patch.object(chat.decoder, 'generate', wraps=chat.decoder.generate) as generation:
                    result = chat.answer(question, identity_guard=policy)
                if result['reason'] == 'identity_guard_rejected' and (status.call_count or generation.call_count):
                    raise ValueError('Identity guard ran after memory or generation')
                results[label] = {'result': result, 'guard_span': guard.last if policy is not None else None,
                                  'status_calls': status.call_count, 'generation_calls': generation.call_count}
            expected_key = None if expected_index is None else KEYS[expected_index]
            expected_value = None if expected_index is None else p.WORDS[VALUES[expected_index]]
            row = {'group': group, 'question': question, 'expected_key': expected_key,
                   'expected_value': expected_value, 'results': results}
            rows.append(row)
            print(json.dumps({'group': group, 'done': len(rows),
                              'baseline': results['baseline']['result']['reason'],
                              'guarded': results['guarded']['result']['reason']}, ensure_ascii=False), flush=True)
        if p.frozen_digest(chat.organism, state) != state_digest or p.frozen_digest(chat.organism, source_state) != source_digest:
            raise ValueError('Evaluation changed protected states')
    if p.sha(p.runtime.CHECKPOINT) != checkpoint_sha or p.sha(p.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Protected files changed')
    summary = {}
    for label in ('baseline', 'guarded'):
        summary[label] = {}
        for group in sorted(set(row['group'] for row in rows)):
            subset = [row for row in rows if row['group'] == group]
            summary[label][group] = {'cases': len(subset),
                'accepted': sum(row['results'][label]['result']['accepted'] for row in subset),
                'correct_key': sum(row['expected_key'] is not None and row['results'][label]['result']['accepted']
                                   and row['results'][label]['result'].get('key') == row['expected_key'] for row in subset),
                'exact_answer': sum(row['expected_value'] is not None and row['results'][label]['result']['accepted']
                                    and row['results'][label]['result'].get('answer', '').strip().rstrip('。.!！ ') == row['expected_value']
                                    for row in subset)}
    result = {'status': 'joint_presence_route_development_audit', 'summary': summary, 'rows': rows,
              'base_sha256': base_sha, 'checkpoint_sha256': checkpoint_sha,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Manually enrolled names and numeric facts in an isolated temporary state.',
                              'Authored development questions with reused entities and model; no independent population estimate.',
                              'Fingerprint guard depends on extracted span completeness and does not resolve homonyms or aliases.',
                              'Rejected ordinary or out-of-domain requests are not equivalent to general chat answers.',
                              'Does not integrate the online low-rank candidates, natural writes or persistent state.',
                              'Exact answer scoring may miss valid paraphrases.']}
    with OUTPUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
