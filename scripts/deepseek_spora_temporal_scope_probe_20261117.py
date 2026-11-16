"""Bounded temporal scope probe; never updates the integrated individual."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch import nn
from torch.nn import functional as F
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.language_training import atomic_save, frozen_digest

OUT = ROOT / 'runs/deepseek-spora-temporal-scope-probe-20261117'
SEED = 117031
NAMES = {'train': ('小苗', '阿昆', '小枫', '阿亭'), 'cal': ('小霜', '阿岚'), 'test': ('小蕊', '阿树')}
TEMPLATES = {
    'train': {
        0: ('不要旧的，{n}现在爱喝什么饮料？', '过去的不查，{n}当前爱喝什么饮料？',
            '{n}现今的饮品偏好是什么？'),
        1: ('{n}还没更改偏好时喜欢什么饮料？', '{n}变更以前偏爱什么饮品？',
            '不问新的，{n}之前喝什么饮料？'),
        2: ('{n}更改以后喜欢喝什么饮料？', '不说以前的，{n}后来偏爱什么饮品？',
            '记录改变之后{n}爱喝什么？'),
        3: ('{n}更改前后分别爱喝什么饮料？', '不要查询{n}此前喝什么饮料。',
            '把{name}以前的饮品记录改掉。'),
    },
    'cal': {
        0: ('不用讲过去，{n}眼下喜欢什么饮料？', '{n}此刻喝什么饮品？'),
        1: ('改变之前{n}偏好哪种饮料？', '{n}最早记录的饮品是什么？'),
        2: ('更新完成后{n}喜欢哪种饮料？', '请说{name}后来爱喝的饮品。'),
        3: ('{n}前后两次饮料分别是什么？', '不要说出{n}的旧饮品。'),
    },
    'test': {
        0: ('旧记录忽略，{n}目前喜欢什么饮品？', '{n}最新的饮料偏好是什么？',
            '不看往日，只说{name}现在喝什么。'),
        1: ('还没改口时{name}偏爱什么饮料？', '改动前{name}最爱喝什么？',
            '新偏好暂且不说，{n}原先爱喝什么？'),
        2: ('老偏好不用讲，{n}改口后喜欢什么？', '{n}更新后偏爱的饮品是什么？',
            '发生变动以后{name}喜欢喝哪一种？'),
        3: ('比较{name}以前与现在的饮品。', '请勿读取{name}原来的饮料记录。',
            '如果{name}之前喜欢茶，请写个故事。'),
    },
}


def build(part):
    return [{'question': template.format(n=name, name=name), 'label': label, 'template': template}
            for name in NAMES[part] for label, options in TEMPLATES[part].items() for template in options]


def predict(logits, threshold):
    probs = logits.softmax(-1)
    winner = probs.argmax(-1)
    return torch.where((winner != 3) & (probs[:, :3].max(-1).values >= threshold),
                       winner, torch.full_like(winner, 3))


def metrics(prediction, labels):
    return {'cases': len(labels), 'correct': int((prediction == labels).sum()),
            'wrong_concrete': int(((prediction != 3) & (prediction != labels)).sum()),
            'false_abstain': int(((prediction == 3) & (labels != 3)).sum())}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    splits = {part: build(part) for part in ('train', 'cal', 'test')}
    for key in TEMPLATES:
        flat = [x for options in TEMPLATES[key].values() for x in options]
        if len(flat) != len(set(flat)):
            raise ValueError('Duplicate templates')
    if any(set(x for v in TEMPLATES[a].values() for x in v) & set(x for v in TEMPLATES[b].values() for x in v)
           for a, b in (('train', 'cal'), ('train', 'test'), ('cal', 'test'))):
        raise ValueError('Template split overlap')
    paths = {'base': runtime.MODEL / 'model.safetensors', 'integrated': runtime.CHECKPOINT,
             'temporal': runtime.TEMPORAL}
    source = {k: runtime.sha(v) for k, v in paths.items()}
    if source['base'] != runtime.WEIGHT_SHA:
        raise ValueError('Base hash mismatch')
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='spora-temporal-scope-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        digest = frozen_digest(chat.organism, chat.state)
        vectors = {}
        for part, rows in splits.items():
            batches = []
            for row in rows:
                features, mask = runtime.extract(chat.base, chat.tokenizer, [row['question']], batch_size=1)
                batches.append(pool(features, mask).float())
            vectors[part] = torch.cat(batches)
            print(json.dumps({'features': part, 'count': len(rows)}), flush=True)
        labels = {part: torch.tensor([r['label'] for r in rows], dtype=torch.long)
                  for part, rows in splits.items()}
        head = nn.Linear(vectors['train'].shape[-1], 4)
        head.load_state_dict(chat.temporal.state_dict(), strict=True)
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.0003, weight_decay=0.02)
        for _ in range(500):
            loss = F.cross_entropy(head(vectors['train']), labels['train']) + .01 * head.weight.square().sum()
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite temporal loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.inference_mode():
            cal_prob = head(vectors['cal']).softmax(-1)
            cal_labels = labels['cal']
            negatives = cal_prob[cal_labels == 3]
            threshold = max((float(x[:3].max()) for x in negatives if int(x.argmax()) != 3), default=-1) + 1e-6
            threshold = max(0., threshold)
            if threshold > 1:
                raise ValueError('No threshold can avoid calibration false reads')
            old_logits = chat.temporal(vectors['test'])
            new_logits = head(vectors['test'])
            old = predict(old_logits, chat.temporal_data['threshold'])
            new = predict(new_logits, threshold)
            # Runtime keyword constraints are reported separately from raw head decisions.
            old_runtime = torch.tensor([runtime.constrain_temporal_intent(r['question'], int(old[i]))
                                        for i, r in enumerate(splits['test'])])
            new_runtime = torch.tensor([runtime.constrain_temporal_intent(r['question'], int(new[i]))
                                        for i, r in enumerate(splits['test'])])
        rows = [{**row, 'baseline': int(old[i]), 'candidate': int(new[i]),
                 'baseline_runtime_rule': int(old_runtime[i]), 'candidate_runtime_rule': int(new_runtime[i])}
                for i, row in enumerate(splits['test'])]
        if frozen_digest(chat.organism, chat.state) != digest:
            raise ValueError('Individual changed')
        candidate = {'format': 'spora-bounded-temporal-gate-v1', 'base_sha256': source['base'],
                     'feature_variant_sha256': chat.temporal_data['feature_variant_sha256'],
                     'threshold': threshold, 'state': head.state_dict(),
                     'classes': ['current', 'past', 'later', 'abstain']}
    if any(runtime.sha(path) != source[k] for k, path in paths.items()):
        raise ValueError('Protected source changed')
    OUT.mkdir()
    atomic_save(candidate, OUT / 'temporal.pt')
    summary = {name: metrics(value, labels['test']) for name, value in
               (('baseline', old), ('candidate', new), ('baseline_runtime_rule', old_runtime),
                ('candidate_runtime_rule', new_runtime))}
    result = {'status': 'isolated_scope_probe_not_deployed', 'summary': summary, 'rows': rows,
              'counts': {k: len(v) for k, v in splits.items()}, 'threshold': threshold,
              'source_sha256': source, 'candidate_sha256': runtime.sha(OUT / 'temporal.pt'),
              'seconds': time.perf_counter()-start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Small correlated authored templates and two test names; names are not identity tests.',
                              'Training designed from observed failures; no assurance of semantic disjointness.',
                              'Temporal classification only; no actual routing, generated answers or persistent writes.',
                              'Keyword-adjusted values simulate the runtime rule; no claim of complete end-to-end behavior.',
                              'Candidate is isolated, not installed or approved for user data.']}
    with (OUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
