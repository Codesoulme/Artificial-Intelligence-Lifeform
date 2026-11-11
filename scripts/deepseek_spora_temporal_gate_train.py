"""Train a bounded temporal router with an explicit abstention class."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_named_binding_probe import NAMES, UNKNOWN
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_composed_router_answer_eval import NEGATIVE
from scripts.deepseek_spora_gate_fresh_intent_audit import KNOWN
from scripts.deepseek_spora_memory_answer_distill import _load_candidate
from scripts.deepseek_spora_recalibrate_name_router import SOURCE, OUTPUT as ROUTER
from scripts.deepseek_temporal_intent_router import TRAIN, HELD, examples
from spora.integrated_checkpoint import replacement_feature_variant_sha256
from spora.language_training import atomic_save
from spora.named_address import feature_files_sha256
from spora.temporal_routing import CURRENT, PAST, LATER

OUTPUT = ROOT/'runs/deepseek-spora-temporal-gate-train-20261111'
ABSTAIN = 3
EXTRA_TRAIN = {
    CURRENT: ('查看{name}当前记下的饮品。', '请查询{name}此刻喜欢喝什么。'),
    PAST: ('查看{name}更早的饮品记录。', '请说出{name}早先喜欢的饮料。',
           '查询先前保存的{name}饮品偏好。'),
    LATER: ('更新之后{name}改爱什么饮品？', '查询更正后{name}的饮料记录。'),
}
TRAIN_ABSTAIN = ('不要告诉我{name}的饮料偏好。', '把{name}的饮品记录改成水。',
                 '谁告诉你{name}喜欢喝什么？', '如果{name}喜欢茶就提醒我。',
                 '{name}是在哪一次更换饮料的？', '{name}的饮料记录发生过几次变化？',
                 '给{name}写一句温暖的问候。')
CAL_POSITIVE = {
    CURRENT: ('{name}如今记载的饮料是什么？', '{name}当前记录中爱喝什么？'),
    PAST: ('请回忆{name}以前记录的饮品。', '{name}更早最爱哪种饮料？'),
    LATER: ('记录更新以后{name}喜欢什么？', '{name}修改后爱喝的饮品是什么？'),
}
CAL_ABSTAIN = ('暂时别查{name}的饮料。', '请把{name}喜欢的饮品更新为茶。',
               '{name}的偏好消息来自哪里？', '假如{name}喜欢咖啡怎么办？',
               '你知道{name}具体何时改过喜好吗？', '给{name}写一首诗。')
FRESH = {
    CURRENT: ('目前的记录显示{name}喜欢喝哪一种？',),
    PAST: ('在这次更正发生之前，{name}喜欢什么饮品？',),
    LATER: ('记下新偏好以后，{name}改喝什么？',),
}
FRESH_ABSTAIN = ('别读取{name}的饮品记录。', '请为{name}编写一句祝福。',
                 '把{name}的饮品喜好调整为牛奶。', '{name}的饮品偏好是在哪一天变化的？')
STEPS = 500
SEED = 510247


def by_name(templates, names):
    return [{'question': t.format(name=name), 'label': label, 'group': 'read'}
            for name in names for label, choices in templates.items() for t in choices]


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base model mismatch')
    router_report = json.loads((ROUTER/'result.json').read_text(encoding='utf-8'))
    if sha(ROUTER/'router.pt') != router_report['router_sha256']:
        raise ValueError('Name router digest mismatch')
    router = torch.load(ROUTER/'router.pt', map_location='cpu', weights_only=True)
    config_hash = feature_files_sha256(MODEL)
    if router['base_sha256'] != WEIGHT_SHA or router['feature_config_sha256'] != config_hash:
        raise ValueError('Router feature provenance mismatch')
    train_names, cal_names = NAMES[:6], NAMES[6:]+UNKNOWN[:2]
    train = []
    for name in train_names:
        train += [{'question': row['text'].replace(NAMES[0], name), 'label': row['label'], 'group': 'read'}
                  for row in examples(TRAIN)]
    train += by_name(EXTRA_TRAIN, train_names)
    train += [{'question': t.format(name=name), 'label': ABSTAIN, 'group': 'abstain'}
              for name in train_names for t in TRAIN_ABSTAIN]
    calibration = by_name(CAL_POSITIVE, cal_names)
    calibration += [{'question': t.format(name=name), 'label': ABSTAIN, 'group': 'abstain'}
                    for name in cal_names for t in CAL_ABSTAIN]
    held = [{'question': row['text'], 'label': row['label'], 'group': 'composed_development'} for row in examples(HELD)]
    held += [{'question': KNOWN[0].format(name=NAMES[0]), 'label': CURRENT, 'group': 'composed_development'},
             {'question': KNOWN[1].format(name=NAMES[0]), 'label': PAST, 'group': 'composed_development'}]
    held += [{'question': q, 'label': PAST if '小青' in q else ABSTAIN,
              'group': 'composed_unbound_time' if '小青' in q else 'composed_negative'} for q in NEGATIVE]
    fresh = by_name(FRESH, NAMES+UNKNOWN[:2])
    for row in fresh:
        row['group'] = 'fresh_read'
    held += fresh
    held += [{'question': t.format(name=name), 'label': ABSTAIN, 'group': 'fresh_abstain'}
             for name in NAMES+UNKNOWN[:2] for t in FRESH_ABSTAIN]
    if ({r['question'] for r in train} & ({r['question'] for r in calibration} | {r['question'] for r in held})
            or {r['question'] for r in calibration} & {r['question'] for r in held}):
        raise ValueError('Train, calibration and held text overlap')
    rows = train+calibration+held
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    blocks = {i: _load_candidate(SOURCE/f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    if replacement_feature_variant_sha256(MODEL, blocks) != router['feature_variant_sha256']:
        raise ValueError('Structural variant mismatch')
    for i, block in blocks.items():
        base.model.layers[i] = block
    features, masks = extract(base, tokenizer, [r['question'] for r in rows], batch_size=4)
    vectors = pool(features, masks).float()
    n_train, n_cal = len(train), len(calibration)
    labels = torch.tensor([r['label'] for r in train], dtype=torch.long)
    model = nn.Linear(vectors.shape[-1], 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
    trace = []
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(vectors[:n_train]), labels) + 0.01*model.weight.square().sum()
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite temporal gate loss')
        loss.backward()
        optimizer.step()
        if step in (0, 99, 249, 499):
            trace.append({'step': step+1, 'loss': float(loss.detach())})
    model.eval()
    with torch.inference_mode():
        logits = model(vectors)
        probs = logits.softmax(-1)
        cal_probs = probs[n_train:n_train+n_cal]
        cal_winner = cal_probs.argmax(-1)
        negative_scores = [float(cal_probs[i, :3].max()) for i, row in enumerate(calibration)
                           if row['label'] == ABSTAIN and int(cal_winner[i]) != ABSTAIN]
        threshold = max(negative_scores)+1e-6 if negative_scores else 0.0
        outcomes = []
        for i, row in enumerate(held):
            p = probs[n_train+n_cal+i]
            winner = int(p.argmax())
            predicted = winner if winner != ABSTAIN and float(p[:3].max()) >= threshold else ABSTAIN
            outcomes.append({**row, 'winner': winner, 'read_confidence': float(p[:3].max()),
                             'predicted': predicted, 'correct': predicted == row['label']})
    groups = {}
    for group in sorted({r['group'] for r in held}):
        subset = [r for r in outcomes if r['group'] == group]
        groups[group] = {'total': len(subset), 'correct': sum(r['correct'] for r in subset),
                         'false_read': sum(r['label'] == ABSTAIN and r['predicted'] != ABSTAIN for r in subset),
                         'false_abstain': sum(r['label'] != ABSTAIN and r['predicted'] == ABSTAIN for r in subset)}
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT/'temporal.pt'
    atomic_save({'format': 'spora-bounded-temporal-gate-v1', 'base_sha256': WEIGHT_SHA,
                 'feature_variant_sha256': router['feature_variant_sha256'], 'threshold': threshold,
                 'state': model.state_dict(), 'classes': ['current', 'past', 'later', 'abstain']}, checkpoint)
    report = {'groups': groups, 'rows': outcomes, 'base_sha256': WEIGHT_SHA,
              'feature_variant_sha256': router['feature_variant_sha256'], 'temporal_sha256': sha(checkpoint),
              'threshold': threshold, 'calibration_false_read_before_threshold': len(negative_scores),
              'train_count': n_train, 'calibration_count': n_cal, 'held_count': len(held), 'trace': trace,
              'elapsed_seconds': time.perf_counter()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Synthetic finite-domain temporal questions and previously inspected development cases.',
                              'Threshold uses small calibration negatives and is not a calibrated probability.',
                              'Labels past/later assume a two-event experiment; no arbitrary timeline resolution.',
                              'No memory reads, state writes, or actual answer generation in this probe.',
                              'No identity, unrestricted names or open-world safety guarantees.']}
    with (OUTPUT/'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'threshold': threshold, 'groups': groups}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
