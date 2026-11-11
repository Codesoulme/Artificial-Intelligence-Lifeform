"""Train a restricted memory-read intent gate and audit its composition."""
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
from scripts.deepseek_learned_name_span import TRAIN_NEGATIVE, single_span_mask
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_named_binding_migrate import OUTPUT as NAMED
from scripts.deepseek_named_binding_probe import KEYS, NAMES, UNKNOWN, TRAIN_TEMPLATES, pooled, predict
from scripts.deepseek_named_incremental_probe import NEW_KEYS
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_memory_answer_distill import _load_candidate
from scripts.deepseek_spora_recalibrate_name_router import SOURCE, OUTPUT as ROUTER
from scripts.deepseek_spora_gate_fresh_intent_audit import OUTPUT as AUDIT
from scripts.deepseek_spora_single_entity_gate import MULTI_TRAIN
from scripts.deepseek_temporal_intent_router import TRAIN as TIME_TRAIN
from spora.growing import load
from spora.integrated_checkpoint import replacement_feature_variant_sha256
from spora.language_training import atomic_save
from spora.named_address import address_basis_sha256, feature_files_sha256

GATE = ROOT/'runs/deepseek-spora-single-entity-gate-v2-20261111'
OUTPUT = ROOT/'runs/deepseek-spora-read-intent-gate-20261111'
TRAIN_NON_READ = ('为{name}写一首小诗。', '如果{name}遇到困难，我该如何鼓励？',
                  '请替{name}设计一张明信片。', '不要告诉我{name}喜欢什么饮料。',
                  '我不想知道{name}喜欢喝什么。', '记住{name}喜欢喝水。',
                  '请修改{name}喜欢的饮料。', '假如{name}喜欢茶，就说一声。',
                  '如果{name}爱喝果汁，请提醒我。', '谁告诉你{name}的饮品喜好？')
CAL_READ = ('我想查{name}目前的饮料记录。', '请回忆{name}过去喜欢喝什么。',
            '{name}后来最喜欢的饮品是什么？', '查询{name}现在保存的数字。')
CAL_NON_READ = ('给{name}写一句安慰的话。', '帮{name}想一个礼物。',
                '别告诉我{name}最爱的饮品。', '我不想查询{name}以前的饮料。',
                '把{name}的偏好更新为茶。', '记录{name}现在喜欢果汁。',
                '若{name}爱喝牛奶，请通知我。', '{name}的饮品消息来源是谁？')
FRESH_NON_READ = ('为{name}编写一句节日祝福。', '帮我想想怎样安慰{name}。',
                  '别读取{name}的饮料记录。', '我不想知道{name}过去喝什么。',
                  '请把{name}偏好的饮料改成水。', '是谁提供了{name}的饮料偏好信息？')
CAL_NO_NAME = ('请讲个笑话。', '我有点累，聊聊天。', '怎么安排周末？', '解释一下记忆。')
STEPS = 450
SEED = 29018
EXTRA_READ = ()
EXTRA_NON_READ = ()
INIT_CHECKPOINT = None
LEARNING_RATE = 0.0006


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base model mismatch')
    gate_report = json.loads((GATE/'result.json').read_text(encoding='utf-8'))
    router_report = json.loads((ROUTER/'result.json').read_text(encoding='utf-8'))
    if sha(GATE/'gate.pt') != gate_report['gate_sha256'] or sha(ROUTER/'router.pt') != router_report['router_sha256']:
        raise ValueError('Existing gate or router digest mismatch')
    gate_data = torch.load(GATE/'gate.pt', map_location='cpu', weights_only=True)
    router = torch.load(ROUTER/'router.pt', map_location='cpu', weights_only=True)
    if gate_data['router_sha256'] != sha(ROUTER/'router.pt') or gate_data['feature_variant_sha256'] != router['feature_variant_sha256']:
        raise ValueError('Gate/router provenance mismatch')
    organism, _, _ = load(NAMED/'individual.pt', 'cpu')
    config_hash = feature_files_sha256(MODEL)
    if (router['base_sha256'] != WEIGHT_SHA or router['feature_config_sha256'] != config_hash
            or router['basis_sha256'] != address_basis_sha256(organism)):
        raise ValueError('Router feature provenance mismatch')
    bases = torch.stack([F.normalize(site.current_basis.detach().float(), dim=-1) for site in organism.memories])
    train, calibration = [], []
    for name, key in zip(NAMES[:6], KEYS[:6]):
        for templates in TIME_TRAIN.values():
            train += [{'question': t.replace(NAMES[0], name), 'label': 1, 'key': key, 'group': 'read'} for t in templates]
        train += [{'question': t.format(name=name), 'label': 1, 'key': key, 'group': 'read'} for t in TRAIN_TEMPLATES]
        train += [{'question': t.format(name=name), 'label': 1, 'key': key, 'group': 'read'} for t in EXTRA_READ]
        train += [{'question': t.format(name=name), 'label': 0, 'key': None, 'group': 'non_read'} for t in TRAIN_NON_READ]
        train += [{'question': t.format(name=name), 'label': 0, 'key': None, 'group': 'non_read'} for t in EXTRA_NON_READ]
    train += [{'question': q, 'label': 0, 'key': None, 'group': 'no_name'} for q in TRAIN_NEGATIVE]
    train += [{'question': t.format(a=name, b=NAMES[(i+1)%len(NAMES)]), 'label': 0, 'key': None, 'group': 'multiple'}
              for i, name in enumerate(NAMES) for t in MULTI_TRAIN]
    for name, key in zip(NAMES[6:]+UNKNOWN[:2], KEYS[6:]+NEW_KEYS):
        calibration += [{'question': t.format(name=name), 'label': 1, 'key': key, 'group': 'read'} for t in CAL_READ]
        calibration += [{'question': t.format(name=name), 'label': 0, 'key': None, 'group': 'non_read'} for t in CAL_NON_READ]
    calibration += [{'question': q, 'label': 0, 'key': None, 'group': 'no_name'} for q in CAL_NO_NAME]
    audit = json.loads(AUDIT.read_text(encoding='utf-8'))
    held = [{'question': r['question'], 'label': int(r['group'] == 'known_memory'),
             'key': r['expected_key'], 'group': r['group']} for r in audit['rows']]
    for name in NAMES+UNKNOWN[:2]:
        held += [{'question': t.format(name=name), 'label': 0, 'key': None, 'group': 'fresh_non_read'}
                 for t in FRESH_NON_READ]
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
    question_vectors = pool(features, masks).float()
    token_vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
    n_train, n_cal = len(train), len(calibration)
    labels = torch.tensor([r['label'] for r in train], dtype=torch.float32)
    intent = nn.Linear(question_vectors.shape[-1], 1)
    if INIT_CHECKPOINT is not None:
        initial = torch.load(INIT_CHECKPOINT, map_location='cpu', weights_only=True)
        if initial.get('format') != 'spora-finite-read-intent-v1' or initial.get('base_sha256') != WEIGHT_SHA:
            raise ValueError('Intent initialization provenance mismatch')
        intent.load_state_dict(initial['state'], strict=True)
    optimizer = torch.optim.AdamW(intent.parameters(), lr=LEARNING_RATE, weight_decay=0.02)
    pos_weight = (len(labels)-labels.sum())/labels.sum()
    trace = []
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        logits = intent(question_vectors[:n_train]).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight) + 0.01*intent.weight.square().sum()
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite intent loss')
        loss.backward()
        optimizer.step()
        if step in (0, 99, 249, 449):
            trace.append({'step': step+1, 'loss': float(loss.detach())})
    intent.eval()
    entity = nn.Linear(question_vectors.shape[-1], 1).eval()
    entity.load_state_dict(gate_data['gate_state'], strict=True)
    locator = nn.Linear(token_vectors.shape[-1], 1).eval()
    locator.load_state_dict(router['locator_state'], strict=True)
    with torch.inference_mode():
        intent_logits = intent(question_vectors).squeeze(-1)
        cal_negative = intent_logits[n_train:n_train+n_cal][torch.tensor([r['label'] == 0 for r in calibration])]
        if not len(cal_negative) or not bool(torch.isfinite(cal_negative).all()):
            raise ValueError('Missing finite calibration negatives')
        threshold = float(torch.nextafter(cal_negative.max(), torch.tensor(float('inf'))))
        entity_scores = entity(question_vectors).squeeze(-1).sigmoid()
        token_scores = locator(token_vectors).squeeze(-1).sigmoid()
        outcomes = []
        for j, row in enumerate(held):
            index = n_train+n_cal+j
            selected, single = single_span_mask(token_scores[index], masks[index], router['locator_threshold'])
            result = {**row, 'intent_logit': float(intent_logits[index]), 'entity_score': float(entity_scores[index]),
                      'single_span': single, 'indices': None, 'address_score': None, 'entity_only_accepted': False,
                      'combined_accepted': False}
            if single:
                vectors = pooled(features[index:index+1], selected.unsqueeze(0))
                indices, scores = predict(vectors, router['projection'], bases)
                result['indices'], result['address_score'] = indices[0].tolist(), float(scores[0].min())
                result['entity_only_accepted'] = bool(result['entity_score'] >= gate_data['gate_threshold']
                                                      and indices[0, 0] == indices[0, 1]
                                                      and result['address_score'] >= gate_data['address_threshold'])
                result['combined_accepted'] = result['entity_only_accepted'] and result['intent_logit'] >= threshold
            result['correct_address'] = row['key'] is not None and result['combined_accepted'] and result['indices'] == [row['key']]*2
            outcomes.append(result)
    groups = {}
    for group in sorted({r['group'] for r in held}):
        subset = [r for r in outcomes if r['group'] == group]
        groups[group] = {'total': len(subset), 'entity_only_accepted': sum(r['entity_only_accepted'] for r in subset),
                         'combined_accepted': sum(r['combined_accepted'] for r in subset),
                         'correct_address': sum(r['correct_address'] for r in subset)}
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT/'intent.pt'
    atomic_save({'format': 'spora-finite-read-intent-v1', 'base_sha256': WEIGHT_SHA,
                 'feature_variant_sha256': router['feature_variant_sha256'], 'router_sha256': sha(ROUTER/'router.pt'),
                 'entity_gate_sha256': sha(GATE/'gate.pt'), 'threshold': threshold, 'state': intent.state_dict()}, checkpoint)
    report = {'groups': groups, 'rows': outcomes, 'base_sha256': WEIGHT_SHA, 'intent_sha256': sha(checkpoint),
              'initial_intent_sha256': sha(INIT_CHECKPOINT) if INIT_CHECKPOINT is not None else None,
              'learning_rate': LEARNING_RATE, 'steps': STEPS,
              'router_sha256': sha(ROUTER/'router.pt'), 'entity_gate_sha256': sha(GATE/'gate.pt'),
              'feature_variant_sha256': router['feature_variant_sha256'], 'intent_threshold': threshold,
              'train_count': n_train, 'calibration_count': n_cal, 'held_count': len(held), 'trace': trace,
              'extra_read_templates': list(EXTRA_READ), 'extra_non_read_templates': list(EXTRA_NON_READ),
              'calibration_rows': [{**row, 'logit': float(intent_logits[n_train+i]),
                                    'accepted': float(intent_logits[n_train+i]) >= threshold}
                                   for i, row in enumerate(calibration)],
              'elapsed_seconds': time.perf_counter()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Synthetic templates; earlier audit questions were already inspected during development.',
                              'Threshold calibrated on a small, nonrepresentative set of negatives.',
                              'Time intent, actual memory state, generation and writes are not tested here.',
                              'Unsupported, adversarial, cross-language and identity requests are not covered.',
                              'The gate is not an access-control boundary or a general language understanding system.']}
    with (OUTPUT/'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'threshold': threshold, 'groups': groups}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
