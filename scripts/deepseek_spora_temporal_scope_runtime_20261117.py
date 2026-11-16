"""Compare temporal policies through an isolated, guarded integrated runtime."""
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch import nn
from scripts import deepseek_spora_output_v2_guarded_history_audit as prior
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.identity_fingerprints import IdentityFingerprints
from spora.identity_read_guard import ConsensusIdentityReadGuard
from spora.memory_output import MemoryOutputAdapter

p = prior.p
SCOPE = ROOT / 'runs/deepseek-spora-temporal-scope-probe-20261117'
OUT = ROOT / 'runs/deepseek-spora-temporal-scope-runtime-20261117.json'
# group, question, expected entity (None => reject), expected mode, expected value
CASES = (
    ('current', '小林现在喜欢喝什么饮料？', 0, 0, 7),
    ('current', '阿宁目前喜欢喝哪种饮品？', 1, 0, 4),
    ('past', '小林还没改口时喜欢什么饮料？', 0, 1, 1),
    ('past', '阿宁在变更之前偏爱什么饮料？', 1, 1, 3),
    ('later', '小林更新后喜欢喝什么饮料？', 0, 2, 7),
    ('later', '阿宁更改后爱喝什么饮料？', 1, 2, 4),
    ('negative', '不要告诉我小林以前喜欢什么饮料。', None, None, None),
    ('negative', '假设阿宁原来喜欢咖啡，请写个故事。', None, None, None),
    ('negative', '小林过去和现在分别喜欢什么饮料？', None, None, None),
    ('negative', '把阿宁以前的饮料改成茶。', None, None, None),
    ('negative', '小林华以前喜欢喝什么饮料？', None, None, None),
    ('negative', '请写一句鼓励学习的话。', None, None, None),
)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    candidate_path = SCOPE / 'temporal.pt'
    scope_report = json.loads((SCOPE / 'result.json').read_text(encoding='utf-8'))
    candidate_sha = p.sha(candidate_path)
    if candidate_sha != scope_report['candidate_sha256']:
        raise ValueError('Candidate temporal hash mismatch')
    candidate = torch.load(candidate_path, map_location='cpu', weights_only=True)
    adapter, adapter_path, adapter_sha = prior.load_checked(prior.SOURCE, 'adapter.pt', 'adapter_sha256')
    locator, locator_path, locator_sha = prior.load_checked(prior.LOCATOR, 'locator.pt', 'locator_sha256')
    source = {str(path): p.sha(path) for path in (p.MODEL / 'model.safetensors', p.runtime.CHECKPOINT,
                                                 candidate_path, adapter_path, locator_path)}
    if source[str(p.MODEL / 'model.safetensors')] != p.WEIGHT_SHA or adapter.get('base_sha256') != p.WEIGHT_SHA:
        raise ValueError('Base hash mismatch')
    for index in (6, 12):
        path = ROOT / 'runs/deepseek-spora-diverse-memory-replay-20261111' / f'layer-{index}.pt'
        if adapter['candidate_sha256'][str(index)] != p.sha(path):
            raise ValueError('Adapter replacement layer provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    rows = []
    original_constraint = p.runtime.constrain_temporal_intent

    def preserve(question, predicted, **kwargs):
        return original_constraint(question, predicted, preserve_abstain=True)

    def raw(question, predicted, **kwargs):
        return predicted

    with tempfile.TemporaryDirectory(prefix='spora-scope-runtime-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        for key in ('base_sha256', 'feature_variant_sha256', 'classes'):
            if candidate[key] != chat.temporal_data[key]:
                raise ValueError('Temporal provenance mismatch')
        if locator['feature_variant_sha256'] != chat.router['feature_variant_sha256']:
            raise ValueError('Identity locator provenance mismatch')
        original_state = chat.state
        initial = p.frozen_digest(chat.organism, original_state)
        state = p.fresh_controlled_state(chat.organism)
        fingerprints = IdentityFingerprints(chat.organism.config.key_count).to(next(chat.organism.parameters()).device)
        for i, name in enumerate(prior.NAMES):
            fingerprints.bind(prior.KEYS[i], name)
        for phase in (0, 1):
            for i, pair in enumerate(prior.FACTS):
                state, decision = p.observe_sourced(chat.organism, state, prior.KEYS[i], pair[phase], 'direct', 1.0)
                if not decision['current_write']:
                    raise ValueError('Controlled write rejected')
        chat.state = state
        state_digest = p.frozen_digest(chat.organism, state)
        guard = ConsensusIdentityReadGuard(fingerprints, chat.tokenizer, locator['state'], locator['threshold'])
        original_head, original_data, original_adapter = chat.temporal, chat.temporal_data, chat.decoder.output_adapter
        new_head = nn.Linear(chat.temporal.in_features, 4).eval()
        new_head.load_state_dict(candidate['state'], strict=True)
        v2 = MemoryOutputAdapter(**adapter['config']).to('cuda').eval()
        v2.load_state_dict(adapter['state'], strict=True)
        chat.decoder.output_adapter = v2
        for group, question, entity, expected_mode, value in CASES:
            variants = {}
            for name, head, head_data, constraint in (
                ('baseline', original_head, original_data, original_constraint),
                ('baseline_preserve_abstain', original_head, original_data, preserve),
                ('candidate_raw', new_head, candidate, raw),
            ):
                with patch.object(chat, 'temporal', head), patch.object(chat, 'temporal_data', head_data), \
                     patch.object(p.runtime, 'constrain_temporal_intent', side_effect=constraint), \
                     patch.object(p.runtime, 'current_memory_status', wraps=p.runtime.current_memory_status) as status, \
                     patch.object(chat.decoder, 'generate', wraps=chat.decoder.generate) as generation:
                    result = chat.answer(question, identity_guard=guard)
                route_correct = bool(entity is not None and result['accepted'] and result.get('key') == prior.KEYS[entity]
                                     and result.get('temporal_intent') == expected_mode)
                answer = result.get('answer') or ''
                variants[name] = {'result': result, 'route_correct': route_correct,
                                  'answer_exact': bool(route_correct and answer.strip().rstrip('。.!！ ') == p.WORDS[value]),
                                  'status_calls': status.call_count, 'generation_calls': generation.call_count}
            rows.append({'group': group, 'question': question, 'expected_key': None if entity is None else prior.KEYS[entity],
                         'expected_mode': expected_mode, 'expected_value': None if value is None else p.WORDS[value],
                         'variants': variants})
            print(json.dumps({'case': len(rows), 'group': group,
                              'reasons': {k: v['result']['reason'] for k, v in variants.items()}}), flush=True)
        chat.decoder.output_adapter = original_adapter
        if p.frozen_digest(chat.organism, state) != state_digest:
            raise ValueError('Temporary individual changed')
        # The original state is distinct from the constructed state.
        if initial != p.frozen_digest(chat.organism, original_state) or chat.state is not state:
            raise ValueError('Source individual or runtime state changed')
    if any(p.sha(Path(path)) != digest for path, digest in source.items()):
        raise ValueError('Protected input changed')
    summary = {}
    for variant in ('baseline', 'baseline_preserve_abstain', 'candidate_raw'):
        positive = [r for r in rows if r['expected_key'] is not None]
        negative = [r for r in rows if r['expected_key'] is None]
        summary[variant] = {'positive_cases': len(positive),
                            'route_correct': sum(r['variants'][variant]['route_correct'] for r in positive),
                            'answer_exact': sum(r['variants'][variant]['answer_exact'] for r in positive),
                            'positive_rejected': sum(not r['variants'][variant]['result']['accepted'] for r in positive),
                            'wrong_route_accepted': sum(r['variants'][variant]['result']['accepted'] and not r['variants'][variant]['route_correct'] for r in positive),
                            'negative_cases': len(negative),
                            'negative_accepted': sum(r['variants'][variant]['result']['accepted'] for r in negative)}
    report = {'status': 'isolated_development_not_acceptance', 'summary': summary, 'rows': rows,
              'source_sha256': source, 'seconds': time.perf_counter()-start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Known names, authored questions and numeric writes; not independent natural dialogue.',
                              'Candidate was designed after related errors; cases are not blind final evaluation.',
                              'Manually supplied identity enrollment and opt-in experimental guard; not authentication.',
                              'One candidate and one seed, with uncalibrated zero confidence threshold.',
                              'No real individual changes, natural writes or crash recovery.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
