"""Compare existing output adapters through guarded routing after corrections."""
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
from scripts.deepseek_spora_presence_routing_joint_audit import NAMES
from spora.identity_fingerprints import IdentityFingerprints
from spora.identity_read_guard import ConsensusIdentityReadGuard
from spora.memory_output import MemoryOutputAdapter

SOURCE = ROOT / 'runs/deepseek-spora-memory-output-adapter-v2-20261111'
LOCATOR = ROOT / 'runs/deepseek-spora-cluener-locator-retrained-20261111'
OUTPUT = ROOT / 'runs/deepseek-spora-output-v2-guarded-history-20261114.json'
FACTS = ((1, 7), (3, 4), (2, 6), (4, 5))
CASES = (
    ('current', '小林现在喜欢喝什么？', 0, 0, 7),
    ('past', '小林改口以前喜欢什么饮料？', 0, 1, 1),
    ('current', '阿宁现在喜欢喝什么？', 1, 0, 4),
    ('past', '阿宁改口以前喜欢什么饮料？', 1, 1, 3),
    ('current', '小雨目前爱喝哪种饮品？', 2, 0, 6),
    ('past', '小雨更新之前喜欢喝什么饮料？', 2, 1, 2),
    ('current', '阿哲现在的饮品偏好是什么？', 3, 0, 5),
    ('past', '阿哲更改前喜欢哪种饮料？', 3, 1, 4),
    ('unknown', '小雪现在喜欢喝什么？', None, None, None),
    ('unknown', '小王以前喜欢什么饮料？', None, None, None),
    ('near_name', '小林华现在喜欢喝什么？', None, None, None),
    ('near_name', '阿宁悦现在喜欢喝什么？', None, None, None),
    ('wrong_relation', '小林喜欢喝的饮料是什么颜色？', None, None, None),
    ('wrong_relation', '阿宁喜欢的饮品多少钱？', None, None, None),
    ('multi', '小林和阿宁现在分别喜欢喝什么？', None, None, None),
    ('negation', '不要告诉我小林喜欢喝什么饮料。', None, None, None),
    ('write', '把阿宁喜欢的饮料改成茶。', None, None, None),
    ('ordinary', '请写一句鼓励学习的话。', None, None, None),
)


def load_checked(folder, filename, digest_field):
    path = folder / filename
    digest = p.sha(path)
    info = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    if digest != info[digest_field]:
        raise ValueError(f'Checksum mismatch: {path}')
    return torch.load(path, map_location='cpu', weights_only=True), path, digest


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    candidate, candidate_path, candidate_sha = load_checked(SOURCE, 'adapter.pt', 'adapter_sha256')
    locator, locator_path, locator_sha = load_checked(LOCATOR, 'locator.pt', 'locator_sha256')
    if base_sha != p.WEIGHT_SHA or candidate.get('base_sha256') != base_sha:
        raise ValueError('Candidate base mismatch')
    for layer in (6, 12):
        layer_path = ROOT / 'runs/deepseek-spora-diverse-memory-replay-20261111' / f'layer-{layer}.pt'
        if candidate['candidate_sha256'][str(layer)] != p.sha(layer_path):
            raise ValueError('Candidate layer mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix='spora-output-history-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        if locator.get('feature_variant_sha256') != chat.router.get('feature_variant_sha256'):
            raise ValueError('Locator feature mismatch')
        original_state = chat.state
        original_digest = p.frozen_digest(chat.organism, original_state)
        fingerprints = IdentityFingerprints(chat.organism.config.key_count).to(next(chat.organism.parameters()).device)
        state = p.fresh_controlled_state(chat.organism)
        for i, name in enumerate(NAMES):
            fingerprints.bind(KEYS[i], name)
        for phase in (0, 1):
            for i, pair in enumerate(FACTS):
                state, decision = p.observe_sourced(chat.organism, state, KEYS[i], pair[phase], 'direct', 1.0)
                if not decision['current_write']:
                    raise ValueError('Controlled observation rejected')
        chat.state = state
        state_digest = p.frozen_digest(chat.organism, state)
        guard = ConsensusIdentityReadGuard(fingerprints, chat.tokenizer, locator['state'], locator['threshold'])
        old = chat.decoder.output_adapter
        new = MemoryOutputAdapter(**candidate['config']).to('cuda').eval()
        new.load_state_dict(candidate['state'], strict=True)
        for group, question, entity, mode, value in CASES:
            results = {}
            for label, adapter in (('current', old), ('v2', new)):
                chat.decoder.output_adapter = adapter
                with patch.object(p.runtime, 'current_memory_status', wraps=p.runtime.current_memory_status) as status, \
                     patch.object(chat.decoder, 'generate', wraps=chat.decoder.generate) as generation:
                    result = chat.answer(question, identity_guard=guard)
                if result['reason'] == 'identity_guard_rejected' and (status.call_count or generation.call_count):
                    raise ValueError('Identity guard ran after reading')
                answer = result.get('answer') or ''
                route = entity is not None and result['accepted'] and result.get('key') == KEYS[entity] and result['temporal_intent'] == mode
                selection = result.get('memory_event_selection') or {}
                historical_event_correct = bool(route and mode == 1 and selection.get('event_time') == entity)
                results[label] = {'result': result, 'route_correct': route,
                                  'historical_event_correct': historical_event_correct,
                                  'exact': bool(route and answer.strip().rstrip('。.!！ ') == p.WORDS[value]),
                                  'contains_expected': bool(route and p.WORDS[value] in answer),
                                  'other_known_values': [word for word in p.WORDS if value is not None and word != p.WORDS[value] and word in answer],
                                  'status_calls': status.call_count, 'generation_calls': generation.call_count}
            rows.append({'group': group, 'question': question, 'expected_key': None if entity is None else KEYS[entity],
                         'expected_mode': mode, 'expected_value': None if value is None else p.WORDS[value], 'results': results})
            print(json.dumps({'done': len(rows), 'group': group,
                              'reasons': {k: v['result']['reason'] for k, v in results.items()}}, ensure_ascii=True), flush=True)
        chat.decoder.output_adapter = old
        if p.frozen_digest(chat.organism, state) != state_digest or p.frozen_digest(chat.organism, original_state) != original_digest:
            raise ValueError('Protected state changed')
    for path, digest in ((p.MODEL / 'model.safetensors', base_sha), (p.runtime.CHECKPOINT, checkpoint_sha),
                         (candidate_path, candidate_sha), (locator_path, locator_sha)):
        if p.sha(path) != digest:
            raise ValueError('Protected file changed')
    summary = {}
    for label in ('current', 'v2'):
        summary[label] = {}
        for group in sorted({row['group'] for row in rows}):
            subset = [row for row in rows if row['group'] == group]
            summary[label][group] = {'cases': len(subset),
                                     'accepted': sum(r['results'][label]['result']['accepted'] for r in subset),
                                     'route_correct': sum(r['results'][label]['route_correct'] for r in subset),
                                     'historical_event_correct': sum(r['results'][label]['historical_event_correct'] for r in subset),
                                     'exact': sum(r['results'][label]['exact'] for r in subset),
                                     'other_value_mentions': sum(bool(r['results'][label]['other_known_values']) for r in subset)}
    result = {'summary': summary, 'rows': rows, 'base_sha256': base_sha, 'checkpoint_sha256': checkpoint_sha,
              'candidate_sha256': candidate_sha, 'locator_sha256': locator_sha,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Small authored development questions and manually enrolled identities, not an independent population estimate.',
                              'Controlled numerical writes and limited finite beverage values, not natural language learning.',
                              'Keyword-based other-value detection is incomplete and can overlap; inspect answers.',
                              'Identity consensus shares model features and is not a general identity or access control system.',
                              'Ordinary request rejection is not a general chat response.']}
    with OUTPUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
