"""Detect all name spans on original-layer features and audit a read veto."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch import nn
from torch.nn import functional as F
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_name_span_probe import oracle_span_mask
from scripts.deepseek_named_binding_probe import NAMES
from scripts.deepseek_pooled_address import extract
from spora.named_address import feature_files_sha256

GATES = ROOT / 'runs/deepseek-spora-original-layer-request-gates-v2-20261114'
CALIBRATION = ROOT / 'runs/deepseek-spora-original-layer-joint-calibration-20261114.json'
OUT = ROOT / 'runs/deepseek-spora-original-multi-span-pilot-20261114'
SEED = 77412
SINGLE_TRAIN = ('{a}爱喝哪种饮料？', '请查询{a}的饮品喜好。', '你还记得{a}喜欢什么吗？')
MULTI_TRAIN = ('{a}和{b}各自喜欢喝什么？', '请比较{a}与{b}的饮料喜好。',
               '说说{a}的饮品，也说说{b}的。', '{a}爱喝什么，{b}又偏爱什么？')
NO_TRAIN = ('请讲个故事。', '怎样培养学习兴趣？', '今天适合喝什么？', '解释一下记忆。')
SINGLE_CAL = ('请说出{a}目前最喜欢的饮料。',)
MULTI_CAL = ('关于{a}和{b}，请分别说出饮品。',)
NO_CAL = ('一周有多少天？',)
UNSEEN = (
    ('single', '请回忆小凯的饮品偏好。', 1),
    ('single', '阿乐喜欢喝哪种饮料？', 1),
    ('single', '帮我查一下小妍爱喝什么。', 1),
    ('single', '小辰记录的饮料喜好是什么？', 1),
    ('multi', '小凯喜欢什么饮品，阿乐又喜欢什么？', 2),
    ('multi', '分别告诉我小妍和小辰的饮品。', 2),
    ('multi', '小辰与小凯各自的饮料是什么？', 2),
    ('multi', '阿乐的偏好是什么？小妍呢？', 2),
    ('ordinary', '今天的气温如何？', 0),
    ('ordinary', '请给我写一句短诗。', 0),
)


def spans_from_mask(mask, offsets, text, question):
    if (not isinstance(mask, torch.Tensor) or mask.ndim != 1 or mask.dtype != torch.bool
            or not isinstance(offsets, torch.Tensor) or offsets.shape != (len(mask), 2)):
        raise ValueError('Invalid span mask or offsets')
    start = text.find(question)
    if not question or start < 0 or text.find(question, start + 1) >= 0:
        raise ValueError('Question must occur exactly once')
    spans, current = [], []
    for index in mask.nonzero(as_tuple=True)[0].tolist():
        left, right = offsets[index].tolist()
        if left < start or right > start + len(question) or left >= right:
            raise ValueError('Selected token is outside question')
        if current and index != current[-1][0] + 1:
            spans.append(current)
            current = []
        current.append((index, left - start, right - start))
    if current:
        spans.append(current)
    return [question[min(x[1] for x in span):max(x[2] for x in span)] for span in spans]


def examples():
    rows = []
    for split, names, singles, multis, ordinary in (
        ('train', NAMES[:4], SINGLE_TRAIN, MULTI_TRAIN, NO_TRAIN),
        ('calibration', NAMES[4:6], SINGLE_CAL, MULTI_CAL, NO_CAL),
    ):
        for name in names:
            rows.extend({'split': split, 'group': 'single', 'question': t.format(a=name), 'names': [name],
                         'expected_count': 1} for t in singles)
        for i, name in enumerate(names):
            other = names[(i + 1) % len(names)]
            rows.extend({'split': split, 'group': 'multi', 'question': t.format(a=name, b=other),
                         'names': [name, other], 'expected_count': 2} for t in multis)
        rows.extend({'split': split, 'group': 'ordinary', 'question': q, 'names': [], 'expected_count': 0}
                    for q in ordinary)
    return rows


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    base_sha, checkpoint_sha = p.sha(p.MODEL / 'model.safetensors'), p.sha(p.runtime.CHECKPOINT)
    gate_path, source_path = GATES / 'gates.pt', GATES / 'result.json'
    gate_sha, source_sha, cal_sha = p.sha(gate_path), p.sha(source_path), p.sha(CALIBRATION)
    source = json.loads(source_path.read_text(encoding='utf-8'))
    calibration = json.loads(CALIBRATION.read_text(encoding='utf-8'))
    if (base_sha != p.WEIGHT_SHA or source['base_sha256'] != base_sha
            or source['checkpoint_sha256'] != checkpoint_sha or source['gates_sha256'] != gate_sha
            or calibration['gates_sha256'] != gate_sha or calibration['source_sha256'] != source_sha):
        raise ValueError('Source provenance mismatch')
    labeled = examples()
    review = []
    for row in source['rows']:
        if row['split'] not in ('prior_name_development', 'new_challenge'):
            continue
        group = row['group']
        count = 2 if group in ('multi', 'fresh_multi') else 0 if group in ('ordinary', 'fresh_ordinary') else 1
        review.append({'split': row['split'], 'group': group, 'question': row['question'],
                       'expected_count': count, 'source': row})
    review += [{'split': 'new_names', 'group': group, 'question': q, 'expected_count': count}
               for group, q, count in UNSEEN]
    rows = labeled + review
    if len({r['question'] for r in rows}) != len(rows):
        raise ValueError('Examples overlap')
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': r['question']}],
                                           tokenize=False, add_generation_prompt=True) for r in rows]
    enc = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False, return_offsets_mapping=True)
    truth = torch.zeros_like(enc['attention_mask'], dtype=torch.bool)
    crossed = set()
    for i, row in enumerate(labeled):
        for name in row['names']:
            span, crossing = oracle_span_mask(enc['offset_mapping'][i:i+1], enc['attention_mask'][i:i+1],
                                              [texts[i]], [row['question']], [name])
            truth[i] |= span[0]
            if crossing[0]:
                crossed.add(i)
    base = p.load_full_streamed().eval()
    features, allowed = extract(base, tokenizer, [r['question'] for r in rows], batch_size=1)
    if truth.shape != allowed.shape or bool((truth & ~allowed).any()):
        raise ValueError('Feature mask mismatch')
    tokens = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
    train_indices = [i for i, r in enumerate(rows) if r['split'] == 'train' and i not in crossed]
    x, y = tokens[train_indices][allowed[train_indices]], truth[train_indices][allowed[train_indices]].float()
    if not bool(y.any()) or bool(y.all()):
        raise ValueError('Both positive and negative token labels required')
    locator = nn.Linear(x.shape[-1], 1)
    optimizer = torch.optim.AdamW(locator.parameters(), lr=0.0005, weight_decay=0.01)
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(locator(x).squeeze(-1), y,
                                                   pos_weight=(y.numel() - y.sum()) / y.sum())
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite locator loss')
        loss.backward()
        optimizer.step()
    locator.eval()
    with torch.inference_mode():
        predicted = (locator(tokens).squeeze(-1).sigmoid() >= 0.5) & allowed
    thresholds = calibration['thresholds']
    results = []
    for i, row in enumerate(rows):
        spans = spans_from_mask(predicted[i], enc['offset_mapping'][i], texts[i], row['question'])
        result = {k: row[k] for k in ('split', 'group', 'question', 'expected_count')}
        result.update({'spans': spans, 'count_correct': len(spans) == row['expected_count'],
                       'representable': i not in crossed if i < len(labeled) else None,
                       'dropped_training': row['split'] == 'train' and i in crossed})
        source_row = row.get('source')
        if source_row:
            prior = source_row.get('prior')
            if prior and prior['guarded_accepted']:
                result['prior_guard_span'] = prior['guard_span']['selected_text']
            result['detector_agrees_guard'] = bool(prior and prior['guarded_accepted'] and len(spans) == 1
                                                    and spans[0] == prior['guard_span']['selected_text'])
            read_pass = source_row['read_score'] >= thresholds['read']
            for label, threshold in (('strict', thresholds['single_current']), ('joint', thresholds['single_joint'])):
                gate_pass = source_row['single_score'] >= threshold and read_pass
                result[label + '_gate_accepted'] = gate_pass
                result[label + '_detector_accepted'] = gate_pass and len(spans) == 1
                if prior:
                    result[label + '_prior_accepted'] = gate_pass and prior['guarded_accepted']
                    result[label + '_combined_accepted'] = gate_pass and result['detector_agrees_guard']
                    result[label + '_combined_correct'] = result[label + '_combined_accepted'] and prior['guarded_correct_key']
        results.append(result)
    if p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(p.runtime.CHECKPOINT) != checkpoint_sha or p.sha(gate_path) != gate_sha or p.sha(source_path) != source_sha or p.sha(CALIBRATION) != cal_sha:
        raise ValueError('Protected input changed')
    summary = {}
    for split in ('calibration', 'prior_name_development', 'new_challenge', 'new_names'):
        summary[split] = {}
        for group in sorted({r['group'] for r in results if r['split'] == split}):
            subset = [r for r in results if r['split'] == split and r['group'] == group]
            item = {'cases': len(subset), 'count_correct': sum(r['count_correct'] for r in subset),
                    'one_span': sum(len(r['spans']) == 1 for r in subset)}
            if split in ('prior_name_development', 'new_challenge'):
                item.update({key: sum(r.get(key, False) for r in subset) for key in
                             ('strict_gate_accepted', 'joint_gate_accepted', 'strict_detector_accepted',
                              'joint_detector_accepted')})
            if split == 'prior_name_development':
                item.update({key: sum(r.get(key, False) for r in subset) for key in
                             ('strict_prior_accepted', 'joint_prior_accepted', 'strict_combined_accepted',
                              'joint_combined_accepted', 'strict_combined_correct', 'joint_combined_correct')})
            summary[split][group] = item
    OUT.mkdir()
    path = OUT / 'locator.pt'
    torch.save({'format': 'spora-original-multi-span-research-v1', 'state': locator.state_dict(),
                'threshold': 0.5, 'base_sha256': base_sha, 'feature_config_sha256': feature_files_sha256(p.MODEL),
                'hidden_indices': [6, 18], 'seed': SEED}, path)
    report = {'summary': summary, 'rows': results, 'dropped_train': [rows[i]['question'] for i in sorted(crossed)
                                                                     if rows[i]['split'] == 'train'],
              'base_sha256': base_sha, 'checkpoint_sha256': checkpoint_sha, 'gates_sha256': gate_sha,
              'calibration_sha256': cal_sha, 'locator_sha256': p.sha(path), 'loss': float(loss.detach()),
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Synthetic token labels and small development examples with possible historical overlap.',
                              'Expected counts supplied by the experiment; no open-world identity or same-name guarantees.',
                              'Adjacent or token-merged names can collapse into one span; span count is not proof of one person.',
                              'Heads share underlying model features, so agreement is not independent.',
                              'Prior router results combined offline, not a real memory route or generation.',
                              'Threshold fixed before this evaluation; later tuning would make these examples development data.']}
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
