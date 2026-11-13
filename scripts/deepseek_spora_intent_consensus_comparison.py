"""Compare existing intent heads with the same opt-in identity guard and state."""
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
from scripts.deepseek_spora_presence_routing_joint_audit import CASES, NAMES, VALUES
from spora.identity_fingerprints import IdentityFingerprints
from spora.identity_read_guard import ConsensusIdentityReadGuard

OUTPUT = ROOT / 'runs/deepseek-spora-intent-consensus-comparison-20261114.json'
LOCATOR = ROOT / 'runs/deepseek-spora-cluener-locator-retrained-20261111'
EXTRA = (
    ('negation', '不要告诉我小林喜欢喝什么饮料。', None),
    ('negation', '别查询阿宁的饮品偏好。', None),
    ('write', '把小林喜欢的饮料改成茶。', None),
    ('write', '请记住阿宁喜欢喝咖啡。', None),
    ('hypothetical', '假设小林喜欢喝茶，请写个故事。', None),
    ('hypothetical', '猜猜阿宁可能喜欢什么饮品。', None),
    ('wrong_relation', '小林喜欢喝的饮料是什么颜色？', None),
    ('wrong_relation', '阿宁喜欢的饮品多少钱？', None),
    ('known', '你还记得小林喜欢喝哪种饮品吗？', 0),
    ('known', '查一下小雨保存的饮料喜好。', 2),
)


def checked_candidate(folder, name, field):
    path = folder / name
    report = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    digest = p.sha(path)
    if digest != report[field]:
        raise ValueError(f'Checksum mismatch: {path}')
    return torch.load(path, map_location='cpu', weights_only=True), path, digest


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Base digest mismatch')
    candidates = {}
    for version in ('v4', 'v6', 'v7'):
        day = '20261111' if version == 'v4' else '20261112'
        candidates[version] = checked_candidate(
            ROOT / f'runs/deepseek-spora-read-intent-gate-{version}-{day}', 'intent.pt', 'intent_sha256')
    locator, locator_path, locator_sha = checked_candidate(LOCATOR, 'locator.pt', 'locator_sha256')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix='spora-intent-comparison-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        source_state = chat.state
        source_digest = p.frozen_digest(chat.organism, source_state)
        for version, (candidate, _, _) in candidates.items():
            for key in ('base_sha256', 'feature_variant_sha256', 'router_sha256', 'entity_gate_sha256'):
                if candidate.get(key) != chat.intent_data.get(key):
                    raise ValueError(f'{version} provenance mismatch: {key}')
        if locator.get('feature_variant_sha256') != chat.router.get('feature_variant_sha256'):
            raise ValueError('Locator feature mismatch')
        device = next(chat.organism.parameters()).device
        fingerprints = IdentityFingerprints(chat.organism.config.key_count).to(device)
        state = p.fresh_controlled_state(chat.organism)
        for index, name in enumerate(NAMES):
            fingerprints.bind(KEYS[index], name)
            state, decision = p.observe_sourced(chat.organism, state, KEYS[index], VALUES[index], 'direct', 1.0)
            if not decision['current_write']:
                raise ValueError('Controlled write rejected')
        chat.state = state
        state_digest = p.frozen_digest(chat.organism, state)
        guard = ConsensusIdentityReadGuard(fingerprints, chat.tokenizer, locator['state'], locator['threshold'])
        for group, question, expected_index in CASES + EXTRA:
            expected_key = None if expected_index is None else KEYS[expected_index]
            expected_value = None if expected_index is None else p.WORDS[VALUES[expected_index]]
            results = {}
            for version, (candidate, _, _) in candidates.items():
                chat.intent_data = candidate
                with patch.object(p.runtime, 'current_memory_status', wraps=p.runtime.current_memory_status) as status, \
                     patch.object(chat.decoder, 'generate', wraps=chat.decoder.generate) as generate:
                    result = chat.answer(question, identity_guard=guard)
                if result['reason'] == 'identity_guard_rejected' and (status.call_count or generate.call_count):
                    raise ValueError('Guard rejected after memory access')
                results[version] = {'result': result, 'status_calls': status.call_count,
                                    'generation_calls': generate.call_count}
            rows.append({'group': group, 'question': question, 'expected_key': expected_key,
                         'expected_value': expected_value, 'results': results})
            print(json.dumps({'done': len(rows), 'group': group,
                              'reasons': {v: r['result']['reason'] for v, r in results.items()}},
                             ensure_ascii=False), flush=True)
        if p.frozen_digest(chat.organism, state) != state_digest or p.frozen_digest(chat.organism, source_state) != source_digest:
            raise ValueError('Protected state changed')
    for _, path, digest in candidates.values():
        if p.sha(path) != digest:
            raise ValueError('Candidate changed')
    if p.sha(locator_path) != locator_sha or p.sha(p.runtime.CHECKPOINT) != checkpoint_sha or p.sha(p.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Protected file changed')
    summary = {}
    for version in candidates:
        summary[version] = {}
        for group in sorted({row['group'] for row in rows}):
            subset = [row for row in rows if row['group'] == group]
            summary[version][group] = {
                'cases': len(subset),
                'accepted': sum(r['results'][version]['result']['accepted'] for r in subset),
                'correct_key': sum(r['expected_key'] is not None and r['results'][version]['result']['accepted']
                                   and r['results'][version]['result'].get('key') == r['expected_key'] for r in subset),
                'exact_answer': sum(r['expected_value'] is not None and r['results'][version]['result']['accepted']
                                    and r['results'][version]['result'].get('answer', '').strip().rstrip('。.!！ ') == r['expected_value']
                                    for r in subset)}
    result = {'summary': summary, 'rows': rows, 'base_sha256': base_sha, 'checkpoint_sha256': checkpoint_sha,
              'candidates_sha256': {v: d for v, (_, _, d) in candidates.items()}, 'locator_sha256': locator_sha,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Small authored development set; not an independent population estimate.',
                              'Manually enrolled identities and numeric writes, not natural language learning.',
                              'Dual locator can share errors and does not resolve homonyms or aliases.',
                              'Rejection of ordinary requests is not a general chat answer.']}
    with OUTPUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
