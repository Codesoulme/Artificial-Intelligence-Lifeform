"""Contrast earliest and previous event reads in an isolated individual."""
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from scripts import deepseek_spora_output_v2_guarded_history_audit as prior
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.identity_fingerprints import IdentityFingerprints
from spora.identity_read_guard import ConsensusIdentityReadGuard
from spora.memory_output import MemoryOutputAdapter

p = prior.p
OUT = ROOT / 'runs/deepseek-spora-three-event-first-audit-20261117.json'
VALUES = ((1, 3, 6), (2, 4, 7))
TEMPLATES = (
    ('first', '{name}最初喜欢喝什么饮料？', 0),
    ('previous', '{name}更改前喜欢喝什么饮料？', 1),
    ('current', '{name}现在喜欢喝什么饮料？', 2),
)


def selected_order(question, order):
    """Experimental prototype: only an explicit first marker changes past selection."""
    return 'effective_first' if order == 'effective_previous' and '最初' in question else order


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    adapter, adapter_path, _ = prior.load_checked(prior.SOURCE, 'adapter.pt', 'adapter_sha256')
    locator, locator_path, _ = prior.load_checked(prior.LOCATOR, 'locator.pt', 'locator_sha256')
    paths = (p.MODEL / 'model.safetensors', p.runtime.CHECKPOINT, adapter_path, locator_path)
    hashes = {str(path): p.sha(path) for path in paths}
    if hashes[str(paths[0])] != p.WEIGHT_SHA or adapter.get('base_sha256') != p.WEIGHT_SHA:
        raise ValueError('Base mismatch')
    for index in (6, 12):
        layer = ROOT / 'runs/deepseek-spora-diverse-memory-replay-20261111' / f'layer-{index}.pt'
        if adapter['candidate_sha256'][str(index)] != p.sha(layer):
            raise ValueError('Adapter feature mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix='spora-three-event-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        if locator['feature_variant_sha256'] != chat.router['feature_variant_sha256']:
            raise ValueError('Locator feature mismatch')
        original_state = chat.state
        original_digest = p.frozen_digest(chat.organism, original_state)
        state = p.fresh_controlled_state(chat.organism)
        fingerprints = IdentityFingerprints(chat.organism.config.key_count).to(next(chat.organism.parameters()).device)
        for person in range(len(VALUES)):
            fingerprints.bind(prior.KEYS[person], prior.NAMES[person])
        for phase in range(3):
            for person, facts in enumerate(VALUES):
                state, decision = p.observe_sourced(chat.organism, state, prior.KEYS[person], facts[phase], 'direct', 1.0)
                if not decision['current_write'] or decision.get('history_time') != phase * len(VALUES) + person:
                    raise ValueError('Controlled event write mismatch')
        chat.state = state
        digest = p.frozen_digest(chat.organism, state)
        guard = ConsensusIdentityReadGuard(fingerprints, chat.tokenizer, locator['state'], locator['threshold'])
        original_adapter = chat.decoder.output_adapter
        v2 = MemoryOutputAdapter(**adapter['config']).to('cuda').eval()
        v2.load_state_dict(adapter['state'], strict=True)
        chat.decoder.output_adapter = v2
        select_event = p.runtime.select_event
        for person, values in enumerate(VALUES):
            for group, template, phase in TEMPLATES:
                question = template.format(name=prior.NAMES[person])
                expected_event = phase * len(VALUES) + person if group != 'current' else None
                variants = {}
                for label in ('existing', 'first_marker'):
                    def select(model, current_state, key, order, **kwargs):
                        return select_event(model, current_state, key,
                                            selected_order(question, order) if label == 'first_marker' else order,
                                            **kwargs)
                    with patch.object(p.runtime, 'select_event', side_effect=select):
                        result = chat.answer(question, identity_guard=guard)
                    selection = result.get('memory_event_selection') or {}
                    route = bool(result['accepted'] and result.get('key') == prior.KEYS[person]
                                 and (group == 'current' and result['temporal_intent'] == 0
                                      or group != 'current' and selection.get('event_time') == expected_event))
                    answer = (result.get('answer') or '').strip().rstrip('。.!！ ')
                    variants[label] = {'result': result, 'correct_event': route,
                                       'correct_answer': bool(route and answer == p.WORDS[values[phase]])}
                rows.append({'person': person, 'group': group, 'question': question,
                             'expected_event': expected_event, 'expected_value': p.WORDS[values[phase]],
                             'variants': variants})
                print(json.dumps({'person': person, 'group': group,
                                  'existing': variants['existing']['correct_event'],
                                  'first_marker': variants['first_marker']['correct_event']}), flush=True)
        chat.decoder.output_adapter = original_adapter
        if p.frozen_digest(chat.organism, state) != digest or p.frozen_digest(chat.organism, original_state) != original_digest:
            raise ValueError('Individual modified')
    if any(p.sha(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError('Source modified')
    summary = {label: {'correct_event': sum(r['variants'][label]['correct_event'] for r in rows),
                       'correct_answer': sum(r['variants'][label]['correct_answer'] for r in rows),
                       'cases': len(rows)} for label in ('existing', 'first_marker')}
    report = {'status': 'isolated_three_event_prototype_not_deployed', 'summary': summary,
              'rows': rows, 'source_sha256': hashes, 'seconds': time.perf_counter()-start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Six authored questions, two known manually enrolled names, three finite values each.',
                              'Prototype recognizes only one explicit marker and is not identity or temporal understanding.',
                              'Known numerical writes and program-controlled event ordering; no natural write or restart.',
                              'Neural provenance can be incomplete or incorrect even when an event is returned.',
                              'No production runtime or real individual modified.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
