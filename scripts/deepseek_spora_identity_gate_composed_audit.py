"""Compare the actual default read path with a research identity gate.

The gate is evaluated as an additional veto on exactly the same predictions;
it does not update the default runtime or any real user individual.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch import nn
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_named_binding_probe import KEYS
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_spora_identity_gate_probe import OUTPUT as GATE
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.internal_memory import observe_sourced

OUTPUT = ROOT / 'runs/deepseek-spora-identity-gate-composed-audit-20261111.json'

# Expected labels are authored for scoring only and never sent to the model.
CASES = [
    ('known', KEYS[0], '我想了解小林目前喜欢喝什么饮品。'),
    ('known', KEYS[1], '告诉我阿宁的饮料喜好。'),
    ('known', KEYS[2], '小雨现在常喝哪种饮料？'),
    ('known', KEYS[3], '阿哲爱喝什么饮品？'),
    ('unknown', None, '我想了解小林乐目前喜欢喝什么饮品。'),
    ('unknown', None, '我想了解林小林目前喜欢喝什么饮品。'),
    ('unknown', None, '告诉我阿宁悦的饮料喜好。'),
    ('unknown', None, '告诉我宁阿宁的饮料喜好。'),
    ('unknown', None, '小雨晴现在常喝哪种饮料？'),
    ('unknown', None, '小小雨现在常喝哪种饮料？'),
    ('unknown', None, '阿哲轩爱喝什么饮品？'),
    ('unknown', None, '哲阿哲爱喝什么饮品？'),
]


def summarize(rows, key):
    known = [r for r in rows if r['group'] == 'known']
    unknown = [r for r in rows if r['group'] == 'unknown']
    return {'known_total': len(known),
            'known_correct_key': sum(r[key] and r['runtime'].get('key') == r['expected_key'] for r in known),
            'known_wrong_key': sum(r[key] and r['runtime'].get('key') != r['expected_key'] for r in known),
            'known_rejected': sum(not r[key] for r in known),
            'unknown_total': len(unknown), 'unknown_false_accept': sum(r[key] for r in unknown)}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    gate_path = GATE / 'gate.pt'
    gate_report = json.loads((GATE / 'result.json').read_text(encoding='utf-8'))
    gate_sha = runtime.sha(gate_path)
    if gate_sha != gate_report['checkpoint_sha256']:
        raise ValueError('Gate digest mismatch')
    data = torch.load(gate_path, map_location='cpu', weights_only=True)
    source_sha = runtime.sha(runtime.CHECKPOINT)
    if source_sha != data['source_checkpoint_sha256']:
        raise ValueError('Gate source checkpoint mismatch')
    torch.set_num_threads(2)
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = runtime.load_chat()
        chat.state = fresh_controlled_state(chat.organism)
        for key, value in zip(KEYS[:4], (6, 1, 2, 3)):
            chat.state, decision = observe_sourced(chat.organism, chat.state, key, value, 'direct', 1.0)
            if not decision['current_write']:
                raise ValueError('Controlled write rejected')
        features, masks = runtime.extract(chat.base, chat.tokenizer, [q for _, _, q in CASES], batch_size=2)
        gate = nn.Linear(data['state']['weight'].shape[1], 1).eval()
        gate.load_state_dict(data['state'], strict=True)
        with torch.no_grad():
            scores = gate(pool(features, masks).float()).squeeze(-1).sigmoid()
        rows = []
        for (group, expected_key, question), score in zip(CASES, scores):
            answer = chat.answer(question)
            gate_accept = bool(score >= data['threshold'])
            row = {'group': group, 'expected_key': expected_key, 'question': question,
                   'runtime': answer, 'gate_score': float(score), 'gate_accepted': gate_accept,
                   'baseline_accepted': bool(answer['accepted']),
                   'combined_accepted': bool(answer['accepted'] and gate_accept)}
            rows.append(row)
            print(json.dumps({'question': question, 'baseline': row['baseline_accepted'],
                              'gate': gate_accept, 'combined': row['combined_accepted'],
                              'key': answer.get('key')}, ensure_ascii=False), flush=True)
    if runtime.sha(runtime.CHECKPOINT) != source_sha or runtime.sha(gate_path) != gate_sha:
        raise ValueError('Protected input changed')
    result = {'status': 'development_only_no_user_state', 'baseline': summarize(rows, 'baseline_accepted'),
              'combined': summarize(rows, 'combined_accepted'), 'rows': rows,
              'source_checkpoint_sha256': source_sha, 'gate_sha256': gate_sha,
              'threshold': data['threshold'], 'seconds': time.perf_counter() - start,
              'limitations': ['Twelve manually authored questions; prior experiments informed their construction.',
                              'Four known people with supervised pre-existing address bindings.',
                              'Only key correctness is scored; answer content is included but not evaluated.',
                              'Gate is an offline additional veto, not integrated into the runtime.',
                              'No real user state was opened or modified; no language-derived writes.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'baseline': result['baseline'], 'combined': result['combined']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
