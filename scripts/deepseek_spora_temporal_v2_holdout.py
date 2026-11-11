"""Compare temporal heads on authored template-disjoint development contrasts."""
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
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_spora_boundary_runtime_pilot import ENROLLMENTS
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_temporal_gate_v2 import trainer

OUTPUT = ROOT / 'runs/deepseek-spora-temporal-v2-holdout-20261112.json'
TEMPLATES = {
    0: ('{name}如今爱喝的饮料是什么？', '现在记下的{name}饮品是哪一种？',
        '你记得{name}眼下喜欢喝什么吗？'),
    1: ('你还记得{name}过去爱喝什么吗？', '早前记下的{name}饮料喜好是什么？',
        '告诉我{name}曾经喝的饮品。'),
    2: ('后来更新的{name}饮料是什么？', '更改偏好之后{name}喜欢喝什么？',
        '新记录里{name}改喝了什么？'),
    3: ('别说出{name}原来喜欢的饮品。', '如果{name}爱喝茶，请讲故事。',
        '把{name}现在爱喝的饮料改为水。'),
}


def decide(head, vector, threshold):
    with torch.no_grad():
        probs = head(vector).softmax(-1)[0]
        winner = int(probs.argmax())
        prediction = winner if winner != 3 and float(probs[:3].max()) >= threshold else 3
    return prediction, [float(x) for x in probs]


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    added = set(trainer.TRAIN_ABSTAIN)
    for choices in trainer.EXTRA_TRAIN.values():
        added.update(choices)
    if set(t for group in TEMPLATES.values() for t in group) & added:
        raise ValueError('Candidate training template overlap')
    path = trainer.OUTPUT / 'temporal.pt'
    info = json.loads((trainer.OUTPUT / 'result.json').read_text(encoding='utf-8'))
    if runtime.sha(path) != info['temporal_sha256']:
        raise ValueError('Candidate digest mismatch')
    candidate = torch.load(path, map_location='cpu', weights_only=True)
    source_sha = runtime.sha(runtime.CHECKPOINT)
    torch.set_num_threads(2)
    start = time.perf_counter()
    rows = []
    cases = [{'label': label, 'question': template.format(name=name), 'key': key}
             for key, name, _ in ENROLLMENTS[:3] for label, options in TEMPLATES.items() for template in options]
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        if any(p.is_meta for p in chat.base.parameters()) or any(b.is_meta for b in chat.base.buffers()):
            raise ValueError('Unmaterialized layer')
        if (candidate['base_sha256'] != chat.temporal_data['base_sha256']
                or candidate['feature_variant_sha256'] != chat.temporal_data['feature_variant_sha256']
                or candidate['classes'] != chat.temporal_data['classes']):
            raise ValueError('Candidate provenance mismatch')
        head = nn.Linear(chat.temporal.in_features, 4).eval()
        head.load_state_dict(candidate['state'], strict=True)
        for case in cases:
            features, masks = runtime.extract(chat.base, chat.tokenizer, [case['question']], batch_size=1)
            vector = pool(features, masks).float()
            old, old_probs = decide(chat.temporal, vector, chat.temporal_data['threshold'])
            new, new_probs = decide(head, vector, candidate['threshold'])
            rows.append({**case, 'baseline': old, 'candidate': new,
                         'baseline_probs': old_probs, 'candidate_probs': new_probs})
            print(json.dumps({'index': len(rows), 'label': case['label'], 'baseline': old, 'candidate': new}), flush=True)
    if runtime.sha(runtime.CHECKPOINT) != source_sha:
        raise ValueError('Protected checkpoint changed')
    summary = {}
    for label in TEMPLATES:
        subset = [r for r in rows if r['label'] == label]
        summary[str(label)] = {'cases': len(subset), 'baseline_correct': sum(r['baseline'] == label for r in subset),
                               'candidate_correct': sum(r['candidate'] == label for r in subset),
                               'baseline_abstain': sum(r['baseline'] == 3 for r in subset),
                               'candidate_abstain': sum(r['candidate'] == 3 for r in subset)}
    result = {'status': 'temporal_v2_template_disjoint_development', 'summary': summary, 'rows': rows,
              'source_checkpoint_sha256': source_sha, 'candidate_sha256': info['temporal_sha256'],
              'seconds': time.perf_counter() - start,
              'limitations': ['Three known names and twelve authored templates in beverage domain.',
                              'Exact overlap checked only against candidate added templates, not all historical artifacts.',
                              'Temporal head alone; does not evaluate full routing, memory or answer.',
                              'Same or ambiguous semantic labels can depend on user context.',
                              'No default model or actual individual changed.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
