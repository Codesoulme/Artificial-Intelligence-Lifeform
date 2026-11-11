"""Exercise temporal agreement against full temporary memory answers."""
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch import nn
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_spora_boundary_runtime_pilot import ENROLLMENTS
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_temporal_regression_runtime import CANDIDATE_DIR, QUESTIONS
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.internal_memory import enable_event_key_index, enable_independent_provenance, observe_sourced
from spora.language_training import frozen_digest
from spora.temporal_agreement import TemporalAgreement

OUTPUT = ROOT / 'runs/deepseek-spora-temporal-agreement-runtime-20261112.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_sha = runtime.sha(runtime.CHECKPOINT)
    info = json.loads((CANDIDATE_DIR / 'result.json').read_text(encoding='utf-8'))
    path = CANDIDATE_DIR / 'temporal.pt'
    if runtime.sha(path) != info['temporal_sha256']:
        raise ValueError('Candidate digest mismatch')
    candidate = torch.load(path, map_location='cpu', weights_only=True)
    torch.set_num_threads(2)
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        if any(p.is_meta for p in chat.base.parameters()) or any(b.is_meta for b in chat.base.buffers()):
            raise ValueError('Unmaterialized layer')
        for k in ('base_sha256', 'feature_variant_sha256', 'classes'):
            if candidate[k] != chat.temporal_data[k]:
                raise ValueError('Candidate provenance mismatch')
        head = nn.Linear(chat.temporal.in_features, 4).eval()
        head.load_state_dict(candidate['state'], strict=True)
        agreement = TemporalAgreement(chat.temporal, head, chat.temporal_data['threshold'], candidate['threshold']).eval()
        for key, name, _ in ENROLLMENTS[:3]:
            state = enable_independent_provenance(chat.organism,
                enable_event_key_index(chat.organism, fresh_controlled_state(chat.organism)))
            for value in (1, 6):
                state, decision = observe_sourced(chat.organism, state, key, value, 'direct', 1.0)
                if not decision['current_write']:
                    raise ValueError('Initialization refused')
            chat.state = state
            digest = frozen_digest(chat.organism, state)
            for group, template, expected_time, expected in QUESTIONS:
                question = template.format(name=name)
                baseline = chat.answer(question)
                with patch.object(chat, 'temporal', head), patch.object(chat, 'temporal_data', candidate):
                    candidate_answer = chat.answer(question)
                with patch.object(chat, 'temporal', agreement):
                    guarded = chat.answer(question)
                row = {'group': group, 'key': key, 'expected_time': expected_time, 'expected': expected,
                       'baseline': baseline, 'candidate': candidate_answer, 'agreement': guarded}
                for label in ('baseline', 'candidate', 'agreement'):
                    result = row[label]
                    row[label + '_correct'] = bool(result['accepted'] and result.get('key') == key
                                                  and result.get('temporal_intent') == expected_time
                                                  and result.get('answer') == expected)
                    row[label + '_wrong_time'] = bool(result['accepted'] and result.get('key') == key
                                                     and result.get('temporal_intent') != expected_time)
                rows.append(row)
                print(json.dumps({'key': key, 'group': group,
                                  **{label + '_reason': row[label]['reason'] for label in ('baseline', 'candidate', 'agreement')},
                                  **{label + '_correct': row[label + '_correct'] for label in ('baseline', 'candidate', 'agreement')}}), flush=True)
            if frozen_digest(chat.organism, state) != digest:
                raise ValueError('Read modified temporary state')
    if runtime.sha(runtime.CHECKPOINT) != source_sha:
        raise ValueError('Protected source changed')
    summary = {}
    for group, _, _, _ in QUESTIONS:
        subset = [r for r in rows if r['group'] == group]
        summary[group] = {'cases': len(subset), **{
            label: {'correct': sum(r[label + '_correct'] for r in subset),
                    'wrong_time': sum(r[label + '_wrong_time'] for r in subset),
                    'accepted': sum(r[label]['accepted'] for r in subset)}
            for label in ('baseline', 'candidate', 'agreement')}}
    report = {'status': 'temporary_temporal_agreement_runtime', 'summary': summary, 'rows': rows,
              'source_checkpoint_sha256': source_sha, 'candidate_sha256': info['temporal_sha256'],
              'seconds': time.perf_counter() - started,
              'limitations': ['Three known names, two synthetic facts and four previously inspected questions.',
                              'Agreement is a heuristic, not a calibrated guarantee; both heads can agree incorrectly.',
                              'The old intent gate remains in place and can abstain before temporal routing.',
                              'No real individual state, default route, or natural-language writing changed.',
                              'Research-only full model loader.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
