"""Train a separate single-entity gate for the experimental memory router."""
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
from scripts.deepseek_spora_name_threshold_calibration import (CAL_TEMPLATES, CAL_NO_NAME, CAL_MULTIPLE,
                                                               OUTPUT as CALIBRATION)
from scripts.deepseek_spora_recalibrate_name_router import SOURCE, OUTPUT as ROUTER
from scripts.deepseek_temporal_intent_router import TRAIN as TIME_TRAIN
from spora.growing import load
from spora.integrated_checkpoint import replacement_feature_variant_sha256
from spora.language_training import atomic_save
from spora.named_address import address_basis_sha256, feature_files_sha256

OUTPUT = ROOT/'runs/deepseek-spora-single-entity-gate-20261111'
MULTI_TRAIN = ('请比较{a}和{b}的饮料偏好。', '{a}与{b}分别爱喝什么？',
               '{a}和{b}的记录是不是一样？', '告诉我{a}以及{b}各自的饮品。')
FRESH_MULTI = ('关于小林及阿宁，请分别说出两人的饮品。', '小雨与阿哲，各自最喜欢喝什么？',
               '给我小梅、阿远两个人的饮料记录。', '小岚爱喝什么，阿明又喜欢什么？',
               '请同时回忆阿松与小雪的饮料。', '小青和阿禾谁喝什么？',
               '对照阿宁以及小雨的偏好。', '分别说说阿哲和小梅的饮品。')
STEPS = 400
SEED = 88102
NUMERIC_TEMPLATES = TRAIN_TEMPLATES[:1]


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base model mismatch')
    previous = json.loads((ROUTER/'result.json').read_text(encoding='utf-8'))
    if sha(ROUTER/'router.pt') != previous['router_sha256']:
        raise ValueError('Router checkpoint mismatch')
    router = torch.load(ROUTER/'router.pt', map_location='cpu', weights_only=True)
    prior_calibration = json.loads((CALIBRATION/'result.json').read_text(encoding='utf-8'))
    if prior_calibration['router_sha256'] != sha(ROUTER/'router.pt'):
        raise ValueError('Calibration/router mismatch')
    organism, _, _ = load(NAMED/'individual.pt', 'cpu')
    config_hash = feature_files_sha256(MODEL)
    if (router.get('base_sha256') != WEIGHT_SHA or router.get('feature_config_sha256') != config_hash
            or router.get('basis_sha256') != address_basis_sha256(organism)):
        raise ValueError('Router provenance mismatch')
    bases = torch.stack([F.normalize(site.current_basis.detach().float(), dim=-1) for site in organism.memories])
    training = []
    for name, key in zip(NAMES+UNKNOWN[:2], KEYS+NEW_KEYS):
        for choices in TIME_TRAIN.values():
            training.extend({'question': q.replace(NAMES[0], name), 'expected_key': key, 'group': 'single'}
                            for q in choices[:2])
        training.extend({'question': template.format(name=name), 'expected_key': key, 'group': 'single'}
                        for template in NUMERIC_TEMPLATES)
    for i, name in enumerate(NAMES):
        other = NAMES[(i+1)%len(NAMES)]
        training.extend({'question': t.format(a=name, b=other), 'expected_key': None, 'group': 'multiple_names'}
                        for t in MULTI_TRAIN)
    training += [{'question': q, 'expected_key': None, 'group': 'no_name'} for q in TRAIN_NEGATIVE]
    calibration = []
    for name, key in zip(NAMES+UNKNOWN[:2], KEYS+NEW_KEYS):
        calibration += [{'question': t.format(name=name), 'expected_key': key, 'group': 'known'} for t in CAL_TEMPLATES]
    for name in UNKNOWN[2:]:
        calibration += [{'question': t.format(name=name), 'expected_key': None, 'group': 'unbound'} for t in CAL_TEMPLATES]
    calibration += [{'question': q, 'expected_key': None, 'group': 'no_name'} for q in CAL_NO_NAME]
    calibration += [{'question': q, 'expected_key': None, 'group': 'multiple_names'} for q in CAL_MULTIPLE]
    held = [{'question': r['question'], 'expected_key': r['expected_key'], 'group': r['group']}
            for r in previous['results']['recalibrated_router']['rows']]
    held += [{'question': q, 'expected_key': None, 'group': 'fresh_multiple_names'} for q in FRESH_MULTI]
    if ({x['question'] for x in training} & ({x['question'] for x in calibration} | {x['question'] for x in held})
            or {x['question'] for x in calibration} & {x['question'] for x in held}):
        raise ValueError('Train, calibration and held text overlap')
    rows = training+calibration+held
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
        raise ValueError('Router and structural variant mismatch')
    for i, block in blocks.items():
        base.model.layers[i] = block
    features, masks = extract(base, tokenizer, [row['question'] for row in rows], batch_size=4)
    question_vectors = pool(features, masks).float()
    n_train, n_cal = len(training), len(calibration)
    targets = torch.tensor([row['group'] == 'single' for row in training], dtype=torch.float32)
    gate = nn.Linear(question_vectors.shape[-1], 1)
    opt = torch.optim.AdamW(gate.parameters(), lr=0.0007, weight_decay=0.02)
    pos_weight = (len(targets)-targets.sum())/targets.sum()
    trace = []
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(gate(question_vectors[:n_train]).squeeze(-1), targets,
                                                  pos_weight=pos_weight) + 0.01*gate.weight.square().sum()
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite gate loss')
        loss.backward()
        opt.step()
        if step in (0, 99, 199, 399):
            trace.append({'step': step+1, 'loss': float(loss.detach())})
    gate.eval()
    token_vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
    locator = nn.Linear(token_vectors.shape[-1], 1).eval()
    locator.load_state_dict(router['locator_state'], strict=True)
    with torch.inference_mode():
        gate_scores = gate(question_vectors).squeeze(-1).sigmoid()
        token_scores = locator(token_vectors).squeeze(-1).sigmoid()
        scored = []
        for index, row in enumerate(rows):
            selected, single = single_span_mask(token_scores[index], masks[index], router['locator_threshold'])
            result = {**row, 'gate_score': float(gate_scores[index]), 'single_span': single,
                      'indices': None, 'address_score': None}
            if single:
                vectors = pooled(features[index:index+1], selected.unsqueeze(0))
                indices, sims = predict(vectors, router['projection'], bases)
                result['indices'], result['address_score'] = indices[0].tolist(), float(sims[0].min())
            scored.append(result)
    cal_rows = scored[n_train:n_train+n_cal]
    held_rows = scored[n_train+n_cal:]
    negative_scores = [x['gate_score'] for x in cal_rows if x['group'] in ('no_name', 'multiple_names')]
    if not negative_scores:
        raise ValueError('Missing calibration negatives')
    gate_threshold = max(negative_scores)+1e-6
    eligible_unbound = [x['address_score'] for x in cal_rows if x['group'] == 'unbound'
                        and x['gate_score'] >= gate_threshold and x['single_span']
                        and x['indices'][0] == x['indices'][1]]
    address_floor = float(previous['results']['prior_router']['threshold'])
    address_threshold = max(address_floor, max(eligible_unbound)+1e-6 if eligible_unbound else address_floor)

    def summarize(items, gated):
        groups = {}
        for source in items:
            row = dict(source)
            gate_pass = not gated or row['gate_score'] >= gate_threshold
            row['accepted'] = bool(gate_pass and row['single_span'] and row['indices'][0] == row['indices'][1]
                                   and row['address_score'] >= address_threshold)
            row['correct'] = row['expected_key'] is not None and row['accepted'] and row['indices'] == [row['expected_key']]*2
            bucket = groups.setdefault(row['group'], {'total': 0, 'accepted': 0, 'correct': 0, 'rows': []})
            bucket['total'] += 1
            bucket['accepted'] += int(row['accepted'])
            bucket['correct'] += int(row['correct'])
            bucket['rows'].append(row)
        return groups

    results = {'ungated': {'calibration': summarize(cal_rows, False), 'held': summarize(held_rows, False)},
               'gated': {'calibration': summarize(cal_rows, True), 'held': summarize(held_rows, True)}}
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT/'gate.pt'
    atomic_save({'format': 'spora-single-entity-gate-v1', 'router_sha256': sha(ROUTER/'router.pt'),
                 'feature_variant_sha256': router['feature_variant_sha256'], 'gate_threshold': gate_threshold,
                 'address_threshold': address_threshold, 'gate_state': gate.state_dict()}, checkpoint)
    report = {'results': results, 'base_sha256': WEIGHT_SHA, 'router_sha256': sha(ROUTER/'router.pt'),
              'gate_sha256': sha(checkpoint), 'feature_variant_sha256': router['feature_variant_sha256'],
              'gate_threshold': gate_threshold, 'address_threshold': address_threshold,
              'calibration_negatives': len(negative_scores), 'eligible_unbound': len(eligible_unbound),
              'training_examples': len(training), 'calibration_examples': n_cal, 'held_examples': len(held),
              'numeric_training_templates': list(NUMERIC_TEMPLATES),
              'trace': trace, 'elapsed_seconds': time.perf_counter()-started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Synthetic questions and small calibration set; earlier held data was inspected.',
                              'Gate distinguishes single from multi/no-name, not memory intent or identity.',
                              'Unseen names and entity aliases are not robustly established.',
                              'No answer generation, automatic writes, or open-world safety validated.']}
    with (OUTPUT/'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'gate_threshold': gate_threshold, 'address_threshold': address_threshold,
                      'held': {g: {k: v for k, v in x.items() if k != 'rows'}
                               for g, x in results['gated']['held'].items()}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
