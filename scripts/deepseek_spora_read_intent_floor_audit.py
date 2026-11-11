"""Measure early read-intent floor tradeoffs on authored contrasts; no runtime change."""
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
from torch.nn import functional as F
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_learned_name_span import single_span_mask
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_spora_boundary_runtime_pilot import ENROLLMENTS
from scripts.deepseek_spora_full_stream_loader import load_full_streamed

OUTPUT = ROOT / 'runs/deepseek-spora-read-intent-floor-audit-20261112.json'
POSITIVE = ('你记得{name}以前喜欢喝什么吗？', '请回忆{name}曾经爱喝的饮料。',
            '过去{name}的饮料偏好是什么？', '你知道{name}之前想喝什么吗？',
            '{name}先前喜欢喝什么？', '告诉我记录里{name}以前喝什么。')
NEGATIVE = ('给{name}推荐一种饮料。', '假设{name}喜欢喝茶，写一个故事。',
            '不要告诉我{name}喜欢喝什么。', '猜猜{name}以前喜欢喝什么。',
            '把{name}喜欢喝咖啡写进记忆。', '请修改{name}的饮料偏好。',
            '如果{name}以前喜欢喝茶，现在会怎样？', '为什么{name}应该少喝饮料？')
FLOORS = (2.0, 1.9, 1.0, 0.0, -1.0)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = runtime.sha(runtime.CHECKPOINT)
    torch.set_num_threads(2)
    started = time.perf_counter()
    cases = [{'kind': label, 'question': template.format(name=name), 'key': key}
             for key, name, _ in ENROLLMENTS[:3]
             for label, templates in (('positive', POSITIVE), ('negative', NEGATIVE))
             for template in templates]
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        if any(p.is_meta for p in chat.base.parameters()) or any(b.is_meta for b in chat.base.buffers()):
            raise ValueError('Unmaterialized layer')
        for case in cases:
            question = case['question']
            features, masks = runtime.extract(chat.base, chat.tokenizer, [question], batch_size=1)
            qvec = pool(features, masks).float()
            vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],))
                                 for i in range(2)], dim=-1)
            locator = nn.Linear(vectors.shape[-1], 1).eval()
            locator.load_state_dict(chat.router['locator_state'], strict=True)
            entity = nn.Linear(qvec.shape[-1], 1).eval()
            entity.load_state_dict(chat.entity_data['gate_state'], strict=True)
            intent = nn.Linear(qvec.shape[-1], 1).eval()
            intent.load_state_dict(chat.intent_data['state'], strict=True)
            with torch.no_grad():
                _, single = single_span_mask(locator(vectors).squeeze(-1).sigmoid()[0], masks[0],
                                             chat.router['locator_threshold'])
                logit = float(intent(qvec).squeeze(-1)[0])
                entity_score = float(entity(qvec).squeeze(-1)[0].sigmoid())
                probs = chat.temporal(qvec).softmax(-1)[0]
                winner = int(probs.argmax())
                temporal = winner if winner != 3 and float(probs[:3].max()) >= chat.temporal_data['threshold'] else 3
            entity_ok = entity_score >= chat.entity_data['gate_threshold']
            anchor = any(word in question for word in runtime.MEMORY_DOMAIN_ANCHORS)
            eligible = anchor and entity_ok and single and temporal != 3
            passes = {str(floor): bool(eligible and (logit >= chat.intent_data['threshold'] or logit >= floor))
                      for floor in FLOORS}
            rows.append({**case, 'logit': logit, 'entity_score': entity_score, 'single': single,
                         'temporal_intent': temporal, 'anchor': anchor, 'early_pass': passes})
            print(json.dumps({'index': len(rows), 'kind': case['kind'], 'logit': round(logit, 3),
                              'pass_2': passes['2.0']}), flush=True)
    if runtime.sha(runtime.CHECKPOINT) != sha:
        raise ValueError('Protected checkpoint changed')
    summary = {}
    for floor in FLOORS:
        positive = [r for r in rows if r['kind'] == 'positive']
        negative = [r for r in rows if r['kind'] == 'negative']
        summary[str(floor)] = {'positive_early_pass': sum(r['early_pass'][str(floor)] for r in positive),
                               'positive_total': len(positive),
                               'negative_early_pass': sum(r['early_pass'][str(floor)] for r in negative),
                               'negative_total': len(negative)}
    result = {'status': 'authored_early_gate_floor_audit', 'summary': summary, 'rows': rows,
              'source_checkpoint_sha256': sha, 'seconds': time.perf_counter() - started,
              'limitations': ['Authored small set, one domain and three known names; not independent benchmark.',
                              'Early-gate approximation only: no name binding, state lookup, answer or persistence.',
                              'Negation, hypothetical and guess labels are task-specific and can be ambiguous.',
                              'Floor candidates are exploratory, not calibrated deployment thresholds.',
                              'No behavior, model weights or individual state changed.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
