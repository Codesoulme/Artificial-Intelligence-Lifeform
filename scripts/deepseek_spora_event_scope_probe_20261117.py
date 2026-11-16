"""Research-only event-order classifier on frozen DeepSeek/SPORA features."""
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

OUT = ROOT / 'runs/deepseek-spora-event-scope-probe-20261117'
LABELS = ('current', 'first', 'previous', 'abstain')
SEED = 117211
NAMES = {'train': ('阿启', '小诚', '小沛', '阿婷', '小章', '阿松'),
         'cal': ('小淼', '阿照'), 'test': ('小羽', '阿容')}
TEMPLATES = {
    'train': {
        0: ('{n}现在爱喝什么饮料？', '只说{n}当前的饮品喜好。',
            '过去的不必提，{n}如今喜欢什么饮料？', '忽略旧值，{n}最新记录的饮品是什么？'),
        1: ('{n}最初喜欢喝什么饮料？', '请说{name}第一次记录的饮品。',
            '不说后来的，{n}最早喜欢什么饮料？', '{n}最开始的饮品偏好是什么？'),
        2: ('{n}最近一次改口之前喜欢什么饮料？', '{n}改动前的上一种饮品是什么？',
            '不问最初，请说{name}上一次的饮品。', '{n}当前记录的前一个饮料是什么？'),
        3: ('{n}以前和现在分别喜欢什么饮料？', '不要读取{name}的饮料偏好。',
            '{n}改变饮品的具体日期是哪天？', '假设{name}过去喝茶，编个故事。'),
    },
    'cal': {
        0: ('只查询{name}眼下的饮料。', '旧偏好不用说，{n}此时喝什么？',
            '{n}现在的饮料是哪一种？'),
        1: ('不要之后的，{n}最初的饮品是什么？', '{n}第一次选的饮料叫什么？',
            '请查询{name}一开始喜欢喝的饮品。'),
        2: ('最新的前一项，{n}喜欢喝什么饮料？', '不问第一个，请查{name}上次的饮品。',
            '{n}最近变更前的饮品是什么？'),
        3: ('{n}每次变动的饮料都是什么？', '把{name}以前喜欢的饮料改掉。',
            '不要告诉我{name}当前喜欢喝什么。'),
    },
    'test': {
        0: ('最早的别说，{n}目前偏好哪种饮料？', '请告诉我{name}最新爱喝什么饮品。',
            '{n}此刻喜欢喝的饮品名称是什么？', '历史先不看，{n}现在喜欢喝哪种饮料？'),
        1: ('{n}最早记下的饮料喜好是什么？', '不看最新的，{n}首次喜欢的饮品是什么？',
            '问一下{name}起初喜欢哪种饮品。', '{n}初次记录的饮料是哪一种？'),
        2: ('{n}最后一次更新前喝什么饮料？', '先别讲第一条，{n}上一条饮品偏好是什么？',
            '现在之前的那次，{n}爱喝什么饮品？', '{n}前一版记下的饮料是什么？'),
        3: ('{n}最初和最近分别喜欢什么饮料？', '不要查{name}以前喝什么。',
            '请撤销{name}最初的饮料记录。', '如果{name}现在爱喝茶，写一首诗。'),
    },
}


def build(part):
    return [{'question': template.format(n=name, name=name), 'label': label, 'template': template}
            for name in NAMES[part] for label, options in TEMPLATES[part].items() for template in options]


def choose(logits, threshold):
    probs = logits.softmax(-1)
    best = probs.argmax(-1)
    return torch.where((best != 3) & (probs[:, :3].max(-1).values >= threshold), best, 3)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    split = {part: build(part) for part in ('train', 'cal', 'test')}
    sets = {part: {row['question'] for row in rows} for part, rows in split.items()}
    if any(len(sets[part]) != len(rows) for part, rows in split.items()) or any(
            sets[a] & sets[b] for a, b in (('train', 'cal'), ('train', 'test'), ('cal', 'test'))):
        raise ValueError('Repeated or overlapping examples')
    template_sets = {part: {row['template'] for row in rows} for part, rows in split.items()}
    if any(template_sets[a] & template_sets[b] for a, b in (('train', 'cal'), ('train', 'test'), ('cal', 'test'))):
        raise ValueError('Template overlap')
    paths = {'base': runtime.MODEL / 'model.safetensors', 'integrated': runtime.CHECKPOINT}
    source = {key: runtime.sha(path) for key, path in paths.items()}
    if source['base'] != runtime.WEIGHT_SHA:
        raise ValueError('Base provenance mismatch')
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='spora-event-scope-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        initial = frozen_digest(chat.organism, chat.state)
        vectors = {}
        for part, rows in split.items():
            features = []
            for row in rows:
                x, mask = runtime.extract(chat.base, chat.tokenizer, [row['question']], batch_size=1)
                features.append(pool(x, mask).float())
            vectors[part] = torch.cat(features)
            print(json.dumps({'part': part, 'count': len(rows)}), flush=True)
        targets = {part: torch.tensor([row['label'] for row in rows]) for part, rows in split.items()}
        head = nn.Linear(vectors['train'].shape[-1], len(LABELS))
        opt = torch.optim.AdamW(head.parameters(), lr=.0003, weight_decay=.02)
        for _ in range(500):
            loss = F.cross_entropy(head(vectors['train']), targets['train']) + .01*head.weight.square().sum()
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite event-scope loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        with torch.inference_mode():
            cal = head(vectors['cal']).softmax(-1)
            negative = cal[targets['cal'] == 3]
            wrong = [float(row[:3].max()) for row in negative if int(row.argmax()) != 3]
            threshold = max(wrong, default=-1.) + 1e-6
            threshold = max(0., threshold)
            if threshold > 1:
                raise ValueError('Cannot reject calibration negatives')
            logits = head(vectors['test'])
            predictions = choose(logits, threshold)
            probabilities = logits.softmax(-1)
        rows = [{**row, 'predicted': int(predictions[i]),
                 'probabilities': [float(x) for x in probabilities[i]]}
                for i, row in enumerate(split['test'])]
        if frozen_digest(chat.organism, chat.state) != initial:
            raise ValueError('Individual state changed')
        variant = chat.temporal_data['feature_variant_sha256']
    if any(runtime.sha(path) != source[key] for key, path in paths.items()):
        raise ValueError('Protected source changed')
    OUT.mkdir()
    path = OUT / 'event_scope.pt'
    atomic_save({'format': 'spora-research-event-scope-v1', 'classes': LABELS, 'threshold': threshold,
                 'base_sha256': source['base'], 'feature_variant_sha256': variant,
                 'state': head.state_dict()}, path)
    groups = {}
    for label, name in enumerate(LABELS):
        group = [r for r in rows if r['label'] == label]
        groups[name] = {'cases': len(group), 'correct': sum(r['predicted'] == label for r in group),
                        'abstain': sum(r['predicted'] == 3 for r in group),
                        'wrong_concrete': sum(r['predicted'] != 3 and r['predicted'] != label for r in group)}
    result = {'status': 'isolated_event_scope_probe_not_deployed', 'groups': groups, 'rows': rows,
              'counts': {name: len(rows) for name, rows in split.items()}, 'threshold': threshold,
              'calibration_incorrect_negative_before_threshold': len(wrong), 'source_sha256': source,
              'feature_variant_sha256': variant, 'head_sha256': runtime.sha(path),
              'seconds': time.perf_counter()-start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Synthetic correlated templates, different names and templates but no semantic-isolation proof.',
                              'Small calibration set; zero calibration false reads is not a safety guarantee.',
                              'No actual entity binding, event selection, reads, answers, durable writes or natural data.',
                              'Different label semantics from the deployed temporal head; never load as its replacement.',
                              'Only one seed; candidate is research-only and not installed.']}
    with (OUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(groups), flush=True)


if __name__ == '__main__':
    main()
