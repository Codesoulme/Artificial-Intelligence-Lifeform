"""Test the event-scope research head through isolated three-event memory reads."""
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
from scripts import deepseek_spora_three_event_first_audit_20261117 as previous
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.identity_fingerprints import IdentityFingerprints
from spora.identity_read_guard import ConsensusIdentityReadGuard
from spora.memory_output import MemoryOutputAdapter

p = prior.p
SOURCE = ROOT / 'runs/deepseek-spora-event-scope-probe-20261117'
OUT = ROOT / 'runs/deepseek-spora-event-scope-runtime-20261117.json'
CASES = (
    ('first', '小林最初喜欢喝什么饮料？', 0, 0),
    ('previous', '小林更改前喜欢喝什么饮料？', 0, 1),
    ('current', '小林现在喜欢喝什么饮料？', 0, 2),
    ('first', '阿宁最初喜欢喝什么饮料？', 1, 0),
    ('previous', '阿宁更改前喜欢喝什么饮料？', 1, 1),
    ('current', '阿宁现在喜欢喝什么饮料？', 1, 2),
    ('negative', '不要查小林原来喜欢什么饮料。', None, None),
    ('negative', '如果阿宁以前喜欢咖啡，请写一个故事。', None, None),
    ('negative', '请分别说小林最初和现在喜欢什么饮料。', None, None),
    ('negative', '把阿宁最初的饮料记录改成茶。', None, None),
    ('negative', '小林华最初喜欢喝什么饮料？', None, None),
    ('negative', '请帮我计算七减四。', None, None),
)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    head_path = SOURCE / 'event_scope.pt'
    info = json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))
    head_sha = p.sha(head_path)
    payload = torch.load(head_path, map_location='cpu', weights_only=True)
    if head_sha != info['head_sha256'] or payload.get('format') != 'spora-research-event-scope-v1':
        raise ValueError('Event head provenance mismatch')
    if tuple(payload['classes']) != ('current', 'first', 'previous', 'abstain'):
        raise ValueError('Unexpected event head semantics')
    adapter, adapter_path, _ = prior.load_checked(prior.SOURCE, 'adapter.pt', 'adapter_sha256')
    locator, locator_path, _ = prior.load_checked(prior.LOCATOR, 'locator.pt', 'locator_sha256')
    paths = (p.MODEL / 'model.safetensors', p.runtime.CHECKPOINT, head_path, adapter_path, locator_path)
    hashes = {str(path): p.sha(path) for path in paths}
    if hashes[str(paths[0])] != p.WEIGHT_SHA or adapter.get('base_sha256') != p.WEIGHT_SHA:
        raise ValueError('Base mismatch')
    for index in (6, 12):
        layer = ROOT / 'runs/deepseek-spora-diverse-memory-replay-20261111' / f'layer-{index}.pt'
        if adapter['candidate_sha256'][str(index)] != p.sha(layer):
            raise ValueError('Adapter replacement provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix='spora-event-scope-runtime-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        if (payload['base_sha256'] != chat.temporal_data['base_sha256']
                or payload['feature_variant_sha256'] != chat.temporal_data['feature_variant_sha256']
                or locator['feature_variant_sha256'] != chat.router['feature_variant_sha256']):
            raise ValueError('Feature variant mismatch')
        head = nn.Linear(chat.temporal.in_features, 4).eval()
        head.load_state_dict(payload['state'], strict=True)
        original_state = chat.state
        original_digest = p.frozen_digest(chat.organism, original_state)
        state = p.fresh_controlled_state(chat.organism)
        fingerprints = IdentityFingerprints(chat.organism.config.key_count).to(next(chat.organism.parameters()).device)
        for i in range(len(previous.VALUES)):
            fingerprints.bind(prior.KEYS[i], prior.NAMES[i])
        for phase in range(3):
            for i, values in enumerate(previous.VALUES):
                state, decision = p.observe_sourced(chat.organism, state, prior.KEYS[i], values[phase], 'direct', 1.0)
                if not decision['current_write'] or decision.get('history_time') != phase*2+i:
                    raise ValueError('Controlled event write mismatch')
        chat.state = state
        digest = p.frozen_digest(chat.organism, state)
        guard = ConsensusIdentityReadGuard(fingerprints, chat.tokenizer, locator['state'], locator['threshold'])
        old_adapter = chat.decoder.output_adapter
        v2 = MemoryOutputAdapter(**adapter['config']).to('cuda').eval()
        v2.load_state_dict(adapter['state'], strict=True)
        chat.decoder.output_adapter = v2
        original_selector = p.runtime.select_event
        for group, question, person, phase in CASES:
            x, mask = p.runtime.extract(chat.base, chat.tokenizer, [question], batch_size=1)
            with torch.inference_mode():
                probs = head(pool(x, mask).float()).softmax(-1)[0]
            winner = int(probs.argmax())
            choice = winner if winner != 3 and float(probs[:3].max()) >= payload['threshold'] else 3
            baseline = chat.answer(question, identity_guard=guard)
            if choice == 3:
                candidate = {'accepted': False, 'reason': 'event_scope_abstain'}
            else:
                def mode_policy(text, predicted, **kwargs):
                    return 0 if choice == 0 else 1
                def selector(model, current, key, order, **kwargs):
                    desired = ('effective_first' if choice == 1 else 'effective_previous')
                    return original_selector(model, current, key, desired, **kwargs)
                with patch.object(p.runtime, 'constrain_temporal_intent', side_effect=mode_policy), \
                     patch.object(p.runtime, 'select_event', side_effect=selector):
                    candidate = chat.answer(question, identity_guard=guard)
            expected_event = None if phase in (None, 2) else phase*2+person
            expected_value = None if person is None else p.WORDS[previous.VALUES[person][phase]]
            def correct(result):
                if person is None or not result['accepted'] or result.get('key') != prior.KEYS[person]:
                    return False
                if phase == 2:
                    return result.get('temporal_intent') == 0
                return (result.get('temporal_intent') == 1 and
                        (result.get('memory_event_selection') or {}).get('event_time') == expected_event)
            row = {'group': group, 'question': question, 'expected_event': expected_event,
                   'expected_value': expected_value, 'scope_class': choice,
                   'scope_probabilities': [float(v) for v in probs], 'baseline': baseline, 'candidate': candidate}
            for label in ('baseline', 'candidate'):
                result = row[label]
                row[label + '_event_correct'] = correct(result)
                row[label + '_answer_correct'] = bool(row[label + '_event_correct'] and
                    (result.get('answer') or '').strip().rstrip('。.!！ ') == expected_value)
            rows.append(row)
            print(json.dumps({'done': len(rows), 'group': group, 'scope_class': choice,
                              'baseline': baseline['reason'], 'candidate': candidate['reason']}), flush=True)
        chat.decoder.output_adapter = old_adapter
        if p.frozen_digest(chat.organism, state) != digest or p.frozen_digest(chat.organism, original_state) != original_digest:
            raise ValueError('Individual state changed')
    if any(p.sha(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError('Protected input changed')
    summary = {}
    for label in ('baseline', 'candidate'):
        positive = [r for r in rows if r['group'] != 'negative']
        negative = [r for r in rows if r['group'] == 'negative']
        summary[label] = {'positive_cases': len(positive),
                          'event_correct': sum(r[label + '_event_correct'] for r in positive),
                          'answer_correct': sum(r[label + '_answer_correct'] for r in positive),
                          'positive_rejected': sum(not r[label]['accepted'] for r in positive),
                          'negative_cases': len(negative), 'negative_accepted': sum(r[label]['accepted'] for r in negative)}
    report = {'status': 'isolated_research_prototype_not_deployed', 'summary': summary, 'rows': rows,
              'source_sha256': hashes, 'seconds': time.perf_counter()-start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Six positive and six negative authored cases; seen names and previously examined wording.',
                              'The head chooses an event class, then overrides temporal mode in this research-only wrapper.',
                              'Manually bound identities and numeric writes; no natural learning, authentication or crash recovery.',
                              'Non-target inputs can be rejected by other gates, so rejection is not proof of head safety.',
                              'No real individual or default runtime changed.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
