"""Compare old and candidate intent gates on new authored wording."""
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
from scripts.deepseek_spora_read_intent_gate_v6 import trainer

OUTPUT = ROOT / 'runs/deepseek-spora-intent-v6-holdout-20261112.json'
POSITIVE = ('你还记得{name}过去爱喝什么吗？', '翻看一下{name}早些年喜欢的饮料。',
            '请查出{name}以前钟爱的饮品。', '早前记下的{name}饮料喜好是什么？',
            '把{name}过去的饮料偏好告诉我。', '记忆中{name}从前常喝什么？')
NEGATIVE = ('你猜{name}过去可能喝什么饮料？', '别说出{name}以前的饮品喜好。',
            '替{name}挑选一款饮料送礼。', '假如{name}爱喝水，编个故事。',
            '把{name}的饮料记录更改为果汁。', '帮我保存{name}喜欢牛奶这件事。',
            '如果{name}以前喜欢喝茶，为什么？', '分析{name}应不应该少喝饮料。')


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if set(POSITIVE + NEGATIVE) & set(trainer.EXTRA_READ + trainer.EXTRA_NON_READ):
        raise ValueError('Candidate training template overlap')
    model_path = trainer.OUTPUT / 'intent.pt'
    info = json.loads((trainer.OUTPUT / 'result.json').read_text(encoding='utf-8'))
    if runtime.sha(model_path) != info['intent_sha256']:
        raise ValueError('Candidate digest mismatch')
    candidate = torch.load(model_path, map_location='cpu', weights_only=True)
    checkpoint_sha = runtime.sha(runtime.CHECKPOINT)
    torch.set_num_threads(2)
    start = time.perf_counter()
    cases = [{'label': label, 'question': template.format(name=name), 'key': key}
             for key, name, _ in ENROLLMENTS[:3]
             for label, templates in (('read', POSITIVE), ('other', NEGATIVE)) for template in templates]
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        if any(p.is_meta for p in chat.base.parameters()) or any(b.is_meta for b in chat.base.buffers()):
            raise ValueError('Unmaterialized layer')
        if (candidate['feature_variant_sha256'] != chat.intent_data['feature_variant_sha256']
                or candidate['router_sha256'] != chat.intent_data['router_sha256']
                or candidate['entity_gate_sha256'] != chat.intent_data['entity_gate_sha256']):
            raise ValueError('Candidate provenance mismatch')
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
            heads = {}
            for name, data in (('baseline', chat.intent_data), ('candidate', candidate)):
                head = nn.Linear(qvec.shape[-1], 1).eval()
                head.load_state_dict(data['state'], strict=True)
                heads[name] = head
            with torch.no_grad():
                _, single = single_span_mask(locator(vectors).squeeze(-1).sigmoid()[0], masks[0],
                                             chat.router['locator_threshold'])
                entity_score = float(entity(qvec).squeeze(-1)[0].sigmoid())
                probs = chat.temporal(qvec).softmax(-1)[0]
                winner = int(probs.argmax())
                temporal = winner if winner != 3 and float(probs[:3].max()) >= chat.temporal_data['threshold'] else 3
                logits = {name: float(head(qvec).squeeze(-1)[0]) for name, head in heads.items()}
            eligible = (any(anchor in question for anchor in runtime.MEMORY_DOMAIN_ANCHORS)
                        and single and entity_score >= chat.entity_data['gate_threshold'] and temporal != 3)
            passes = {name: bool(eligible and (logits[name] >= data['threshold']
                                              or logits[name] >= runtime.READ_INTENT_RECOVERY_FLOOR))
                      for name, data in (('baseline', chat.intent_data), ('candidate', candidate))}
            rows.append({**case, 'logits': logits, 'single': single, 'entity_score': entity_score,
                         'temporal_intent': temporal, 'early_pass': passes})
            print(json.dumps({'index': len(rows), 'label': case['label'], **passes}), flush=True)
    if runtime.sha(runtime.CHECKPOINT) != checkpoint_sha:
        raise ValueError('Protected source changed')
    summary = {}
    for name in ('baseline', 'candidate'):
        summary[name] = {label: {'passed': sum(r['early_pass'][name] for r in rows if r['label'] == label),
                                 'total': sum(r['label'] == label for r in rows)} for label in ('read', 'other')}
    report = {'status': 'authored_intent_v6_early_holdout', 'summary': summary, 'rows': rows,
              'source_checkpoint_sha256': checkpoint_sha, 'candidate_sha256': info['intent_sha256'],
              'seconds': time.perf_counter() - start,
              'limitations': ['Template-disjoint from v6 extras, but related authored examples and known names.',
                              'Only checks exact overlap with v6 added templates, not all prior artifacts.',
                              'Early gates only: no name binding, individual read, full answer or writes.',
                              'Guess and hypothetical labels are task-specific and may be ambiguous.',
                              'Candidate threshold from small calibration and same existing recovery floor.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
