"""Test temporal-head regressions through actual temporary memory answers."""
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
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.internal_memory import enable_event_key_index, enable_independent_provenance, observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-temporal-regression-runtime-20261112.json'
CANDIDATE_DIR = ROOT / 'runs/deepseek-spora-temporal-gate-v3-20261112'
QUESTIONS = (('current', '截至目前，{name}的饮品偏好是什么？', 0, '水'),
             ('past', '{name}以前喜欢哪种饮料？', 1, '咖啡'),
             ('later', '后续记录中，{name}的饮品偏好是什么？', 2, '水'),
             ('past_new', '告诉我{name}曾经喝的饮品。', 1, '咖啡'))


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_sha = runtime.sha(runtime.CHECKPOINT)
    info = json.loads((CANDIDATE_DIR / 'result.json').read_text(encoding='utf-8'))
    model_path = CANDIDATE_DIR / 'temporal.pt'
    if runtime.sha(model_path) != info['temporal_sha256']:
        raise ValueError('Candidate digest mismatch')
    candidate = torch.load(model_path, map_location='cpu', weights_only=True)
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
        for key, name, _ in ENROLLMENTS[:3]:
            state = enable_independent_provenance(chat.organism,
                enable_event_key_index(chat.organism, fresh_controlled_state(chat.organism)))
            for value in (1, 6):
                state, decision = observe_sourced(chat.organism, state, key, value, 'direct', 1.0)
                if not decision['current_write']:
                    raise ValueError('Initialization was not accepted')
            chat.state = state
            digest = frozen_digest(chat.organism, state)
            for group, template, expected_time, expected in QUESTIONS:
                question = template.format(name=name)
                baseline = chat.answer(question)
                with patch.object(chat, 'temporal', head), patch.object(chat, 'temporal_data', candidate):
                    experimental = chat.answer(question)
                row = {'group': group, 'key': key, 'expected_time': expected_time, 'expected': expected,
                       'baseline': baseline, 'candidate': experimental}
                for label in ('baseline', 'candidate'):
                    result = row[label]
                    row[label + '_route_correct'] = bool(result['accepted'] and result.get('key') == key
                                                         and result.get('temporal_intent') == expected_time)
                    row[label + '_answer_correct'] = bool(row[label + '_route_correct'] and result.get('answer') == expected)
                rows.append(row)
                print(json.dumps({'key': key, 'group': group,
                                  'baseline_reason': baseline['reason'], 'baseline_time': baseline['temporal_intent'],
                                  'candidate_reason': experimental['reason'], 'candidate_time': experimental['temporal_intent'],
                                  'baseline_correct': row['baseline_answer_correct'],
                                  'candidate_correct': row['candidate_answer_correct']}), flush=True)
            if frozen_digest(chat.organism, state) != digest:
                raise ValueError('Read modified temporary state')
    if runtime.sha(runtime.CHECKPOINT) != source_sha:
        raise ValueError('Protected source changed')
    summary = {}
    for group, _, _, _ in QUESTIONS:
        subset = [r for r in rows if r['group'] == group]
        summary[group] = {'cases': len(subset),
                          **{label + '_route_correct': sum(r[label + '_route_correct'] for r in subset)
                             for label in ('baseline', 'candidate')},
                          **{label + '_answer_correct': sum(r[label + '_answer_correct'] for r in subset)
                             for label in ('baseline', 'candidate')}}
    report = {'status': 'temporary_temporal_head_runtime_contrast', 'summary': summary, 'rows': rows,
              'source_checkpoint_sha256': source_sha, 'candidate_sha256': info['temporal_sha256'],
              'seconds': time.perf_counter() - started,
              'limitations': [f'Three known names, two manually written facts and {len(QUESTIONS)} authored questions.',
                              'Regression and fresh questions were previously inspected during development.',
                              'Fixed numerical memory values, not a natural language write or general timeline.',
                              'End-to-end other gates may abstain before or after temporal classification.',
                              'Research loader and temporary candidate only; real individual unchanged.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
