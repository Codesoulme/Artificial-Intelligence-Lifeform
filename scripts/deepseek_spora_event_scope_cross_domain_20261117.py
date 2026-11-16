"""Fixed, authored cross-domain challenge for the frozen event-scope head.

This is not an acceptance set: it is small, authored, and contains no transactions.
Do not tune the classifier or its threshold on these cases.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch import nn

from scripts import deepseek_spora_event_scope_probe_20261117 as scope
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.language_training import frozen_digest

SOURCE = ROOT / 'runs/deepseek-spora-event-scope-probe-20261117'
OUT = ROOT / 'runs/deepseek-spora-event-scope-cross-domain-20261117.json'

# Labels describe one requested snapshot, not whether an entity exists or a
# given date is present. Multi-snapshot, hypothetical, commands, and specific
# dates are deliberately outside the four-class head's expressible scope.
CASES = (
    ('current', '工作', '陈悦现在在哪座城市工作？'),
    ('current', '住址', '旧地址不用管，周雨目前住在哪里？'),
    ('current', '宠物', '请告诉我李明如今养的宠物是什么。'),
    ('current', '职业', '王芳最新一次登记的职业是什么？'),
    ('current', '饮食', '赵岩此刻的饮食偏好是什么？'),
    ('current', '称呼', '现在应该怎么称呼许文？'),
    ('first', '工作', '陈悦最早工作的城市是哪座？'),
    ('first', '住址', '最初周雨住在哪里，后来搬去哪儿不用说。'),
    ('first', '宠物', '李明最开始养的宠物是哪一种？'),
    ('first', '职业', '王芳第一次登记的职业是什么？'),
    ('first', '饮食', '赵岩起初的饮食偏好是什么？'),
    ('first', '称呼', '最早我应该怎么称呼许文？'),
    ('previous', '工作', '陈悦换工作地点之前在哪座城市？'),
    ('previous', '住址', '最近一次搬家以前，周雨住在哪？'),
    ('previous', '宠物', '李明养现在这只宠物之前养的是什么？'),
    ('previous', '职业', '最近一次职业变更前，王芳是做什么的？'),
    ('previous', '饮食', '赵岩最近一次改变饮食之前的偏好是什么？'),
    ('previous', '称呼', '许文改称呼之前叫做什么？'),
    ('abstain', '多个时间', '分别说出陈悦最初与现在工作的城市。'),
    ('abstain', '否定', '不要告诉我周雨过去住在哪里。'),
    ('abstain', '假设', '假如李明以前养过猫，请写首诗。'),
    ('abstain', '修改', '将王芳以前的职业记录删除。'),
    ('abstain', '指定日期', '赵岩2024年5月的饮食偏好是什么？'),
    ('abstain', '无关', '计算十八除以三。'),
)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if len({question for _, _, question in CASES}) != len(CASES):
        raise ValueError('Duplicate challenge question')
    head_path = SOURCE / 'event_scope.pt'
    info = json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))
    head_sha = runtime.sha(head_path)
    payload = torch.load(head_path, map_location='cpu', weights_only=True)
    if (head_sha != info['head_sha256'] or payload.get('format') != 'spora-research-event-scope-v1'
            or tuple(payload['classes']) != scope.LABELS):
        raise ValueError('Unexpected head provenance or semantics')
    paths = {'base': runtime.MODEL / 'model.safetensors', 'integrated': runtime.CHECKPOINT,
             'head': head_path}
    hashes = {key: runtime.sha(path) for key, path in paths.items()}
    if hashes['base'] != runtime.WEIGHT_SHA or payload['base_sha256'] != runtime.WEIGHT_SHA:
        raise ValueError('Base provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix='spora-scope-domain-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        if payload['feature_variant_sha256'] != chat.temporal_data['feature_variant_sha256']:
            raise ValueError('Feature mismatch')
        digest = frozen_digest(chat.organism, chat.state)
        head = nn.Linear(chat.temporal.in_features, len(scope.LABELS)).eval()
        head.load_state_dict(payload['state'], strict=True)
        for label, domain, question in CASES:
            x, mask = runtime.extract(chat.base, chat.tokenizer, [question], batch_size=1)
            with torch.inference_mode():
                logits = head(pool(x, mask).float())
                probabilities = logits.softmax(-1)[0]
                predicted = int(scope.choose(logits, payload['threshold'])[0])
            rows.append({'label': label, 'domain': domain, 'question': question,
                         'predicted': scope.LABELS[predicted],
                         'unthresholded': scope.LABELS[int(probabilities.argmax())],
                         'probabilities': [float(v) for v in probabilities]})
            print(json.dumps({'done': len(rows), 'label': label, 'predicted': scope.LABELS[predicted]},
                             ensure_ascii=False), flush=True)
        if frozen_digest(chat.organism, chat.state) != digest:
            raise ValueError('Individual state changed')
    if any(runtime.sha(path) != hashes[key] for key, path in paths.items()):
        raise ValueError('Protected input changed')
    by_label = {}
    for label in scope.LABELS:
        group = [r for r in rows if r['label'] == label]
        by_label[label] = {'cases': len(group), 'correct': sum(r['predicted'] == label for r in group),
                           'abstain': sum(r['predicted'] == 'abstain' for r in group),
                           'wrong_concrete': sum(r['predicted'] not in (label, 'abstain') for r in group)}
    result = {'status': 'research_only_not_deployed', 'threshold': payload['threshold'],
              'by_label': by_label, 'rows': rows, 'source_sha256': hashes,
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['24 authored questions; no independently collected dialogue.',
                              'Fixed before running; not used to train or calibrate this candidate.',
                              'No entity checks, event retrieval, answer generation, or persistence.',
                              'Specific dates are unsupported, not a claim that all date queries should abstain.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(by_label), flush=True)


if __name__ == '__main__':
    main()
