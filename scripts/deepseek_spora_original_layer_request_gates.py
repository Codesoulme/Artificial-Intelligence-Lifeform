"""Train original-layer entity/intent gates and combine prior name-route results."""
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
from scripts.deepseek_named_binding_probe import NAMES
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_original_layer_name_router import OUT as ROUTER, HELD_NEGATIVE
from spora.named_address import feature_files_sha256

OUT = ROOT / 'runs/deepseek-spora-original-layer-request-gates-20261114'
SEED = 39428
TRAIN_TEMPLATES = {
    'read': ('{name}现在喜欢喝哪种饮料？', '回忆一下{name}的饮品偏好。',
             '告诉我{name}喜欢的饮料是什么。'),
    'attribute': ('{name}喜欢的饮料是什么颜色？', '{name}爱喝的饮品卖多少钱？'),
    'negative': ('不要告诉我{name}喜欢喝什么饮品。', '不要查{name}的饮料喜好。'),
    'write': ('把{name}的饮品偏好改成茶。', '记住{name}喜欢喝咖啡。'),
    'hypothetical': ('假设{name}喜欢喝水，请讲故事。', '猜一下{name}喜欢喝什么。'),
    'recommend': ('给{name}推荐一种饮品。',),
}
CAL_TEMPLATES = {
    'read': ('请查询{name}现在最爱喝的饮品。', '记忆中{name}偏好哪种饮料？'),
    'attribute': ('{name}喜欢的饮品有多甜？',),
    'negative': ('我不想知道{name}喜欢喝什么。',),
    'write': ('删除{name}的饮料偏好。',),
    'hypothetical': ('推测{name}可能喝什么饮品。',),
    'recommend': ('给{name}选一种新的饮料。',),
}
MULTI_TRAIN = ('{a}和{b}分别喜欢喝什么饮料？', '比较{a}与{b}的饮品喜好。',
               '请说出{a}以及{b}各自喜欢喝的东西。')
MULTI_CAL = ('{a}和{b}的饮料选择分别是什么？', '告诉我{a}与{b}各自爱喝什么。')
NO_TRAIN = ('今天天气如何？', '给我讲一个故事。', '饮水有什么好处？', '今天想喝什么？')
NO_CAL = ('一周有多少天？', '请写一句鼓励学习的话。')
CHALLENGE = (
    ('attribute', '请说出小岚喜欢的饮品是什么颜色。', 1, 0),
    ('negative', '不要透露阿明的饮料喜好。', 1, 0),
    ('write', '请把小岚爱喝的饮品换成果汁。', 1, 0),
    ('hypothetical', '猜一猜阿明可能喜欢什么饮品。', 1, 0),
    ('recommend', '替小岚选一种饮料。', 1, 0),
    ('multi', '阿明喜欢喝什么，小岚又喜欢哪种饮品？', 0, 1),
    ('multi', '小岚和阿明各自偏爱什么？', 0, 1),
    ('ordinary', '请说明一天有多少小时。', 0, 0),
)


def build_rows(prior_rows):
    rows = []
    for split, names, templates, multis, no_names in (
        ('train', NAMES[:4], TRAIN_TEMPLATES, MULTI_TRAIN, NO_TRAIN),
        ('calibration', NAMES[4:6], CAL_TEMPLATES, MULTI_CAL, NO_CAL),
    ):
        for name in names:
            for group, variants in templates.items():
                rows.extend({'split': split, 'group': group, 'question': template.format(name=name),
                             'single': 1, 'read': int(group == 'read')} for template in variants)
        for index, name in enumerate(names):
            other = names[(index + 1) % len(names)]
            rows.extend({'split': split, 'group': 'multi', 'question': t.format(a=name, b=other),
                         'single': 0, 'read': 1} for t in multis)
        rows.extend({'split': split, 'group': 'ordinary', 'question': question, 'single': 0, 'read': 0}
                    for question in no_names)
    for prior in prior_rows:
        group = prior['group']
        if group == 'no_or_multiple':
            is_multi = prior['question'] in HELD_NEGATIVE[2:]
            single, read = 0, int(is_multi)
            group = 'multi' if is_multi else 'ordinary'
        else:
            single, read = 1, 1
        rows.append({'split': 'prior_name_development', 'group': group, 'question': prior['question'],
                     'single': single, 'read': read, 'prior': prior})
    rows.extend({'split': 'new_challenge', 'group': group, 'question': question,
                 'single': single, 'read': read} for group, question, single, read in CHALLENGE)
    questions = [r['question'] for r in rows]
    if len(set(questions)) != len(questions):
        raise ValueError('Training, calibration and held questions must be distinct')
    return rows


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    router_path = ROUTER / 'router.pt'
    router_sha = p.sha(router_path)
    prior_path = ROUTER / 'result.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    if (base_sha != p.WEIGHT_SHA or prior['base_sha256'] != base_sha
            or prior['checkpoint_sha256'] != checkpoint_sha or prior['router_sha256'] != router_sha):
        raise ValueError('Router or base provenance mismatch')
    rows = build_rows(prior['rows'])
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = p.load_full_streamed().eval()
    features, masks = extract(base, tokenizer, [row['question'] for row in rows], batch_size=1)
    vectors = pool(features, masks).float()
    train = torch.tensor([row['split'] == 'train' for row in rows])
    calibration = torch.tensor([row['split'] == 'calibration' for row in rows])
    states, thresholds, losses = {}, {}, {}
    for field in ('single', 'read'):
        targets = torch.tensor([row[field] for row in rows], dtype=torch.float32)
        model = nn.Linear(vectors.shape[-1], 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.02)
        weight = (int(train.sum()) - targets[train].sum()) / targets[train].sum()
        for _ in range(300):
            optimizer.zero_grad(set_to_none=True)
            scores = model(vectors[train]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(scores, targets[train], pos_weight=weight)
            loss = loss + 0.01 * model.weight.square().sum()
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite gate loss')
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            scores = model(vectors).squeeze(-1)
        negatives = scores[calibration & (targets == 0)]
        threshold = float(torch.nextafter(negatives.max(), torch.tensor(float('inf'))))
        states[field] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        thresholds[field] = threshold
        losses[field] = float(loss.detach())
        for i, row in enumerate(rows):
            row[field + '_score'] = float(scores[i])
            row[field + '_accepted'] = row[field + '_score'] >= threshold
    for row in rows:
        row['gates_accepted'] = row['single_accepted'] and row['read_accepted']
        if 'prior' in row:
            row['combined_accepted'] = row['gates_accepted'] and row['prior']['guarded_accepted']
            row['combined_correct_key'] = row['gates_accepted'] and row['prior']['guarded_correct_key']
    if p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(p.runtime.CHECKPOINT) != checkpoint_sha or p.sha(router_path) != router_sha:
        raise ValueError('Protected file changed')
    groups = {}
    for split in ('train', 'calibration', 'prior_name_development', 'new_challenge'):
        groups[split] = {}
        for group in sorted({r['group'] for r in rows if r['split'] == split}):
            subset = [r for r in rows if r['split'] == split and r['group'] == group]
            summary = {'cases': len(subset), 'single_accepted': sum(r['single_accepted'] for r in subset),
                       'read_accepted': sum(r['read_accepted'] for r in subset),
                       'gates_accepted': sum(r['gates_accepted'] for r in subset)}
            if split == 'prior_name_development':
                summary.update({'prior_guarded_accepted': sum(r['prior']['guarded_accepted'] for r in subset),
                                'combined_accepted': sum(r['combined_accepted'] for r in subset),
                                'combined_correct_key': sum(r['combined_correct_key'] for r in subset)})
            groups[split][group] = summary
    OUT.mkdir()
    gate_path = OUT / 'gates.pt'
    torch.save({'format': 'spora-original-layer-request-gates-research-v1', 'states': states,
                'thresholds': thresholds, 'base_sha256': base_sha, 'checkpoint_sha256': checkpoint_sha,
                'router_sha256': router_sha, 'feature_config_sha256': feature_files_sha256(p.MODEL),
                'hidden_indices': [6, 18]}, gate_path)
    report = {'groups': groups, 'rows': rows, 'thresholds': thresholds, 'losses': losses, 'seed': SEED,
              'base_sha256': base_sha, 'checkpoint_sha256': checkpoint_sha, 'router_sha256': router_sha,
              'gates_sha256': p.sha(gate_path), 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Artificial samples and prior router development rows; not an independent population estimate.',
                              'Thresholds chosen from small calibration negatives; not calibrated risk guarantees.',
                              'Separate offline classifier results combined with previously saved router outputs.',
                              'No live identity, temporal selection, memory reading, generation or crash recovery.',
                              'Names and templates may overlap project history; no general semantic or security claim.']}
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(groups), flush=True)


if __name__ == '__main__':
    main()
