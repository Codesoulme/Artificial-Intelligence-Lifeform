"""Recalibrate the finite named-address router on SPORA-pair features."""
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
from scripts.deepseek_learned_name_span import (OUTPUT as LOCATOR, TRAIN_NEGATIVE, HELD_NEGATIVE,
                                                HELD_MULTIPLE, single_span_mask)
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_name_span_probe import OUTPUT as PROJECTION, oracle_span_mask
from scripts.deepseek_named_binding_migrate import OUTPUT as NAMED
from scripts.deepseek_named_binding_probe import (NAMES, KEYS, UNKNOWN, TRAIN_TEMPLATES, HELD_TEMPLATES,
                                                 pooled, predict, fit_binding)
from scripts.deepseek_named_incremental_probe import NEW_KEYS
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, _load_candidate
from scripts.deepseek_temporal_intent_router import TRAIN as TIME_TRAIN, HELD as TIME_HELD, examples
from spora.growing import load
from spora.integrated_checkpoint import replacement_feature_variant_sha256
from spora.language_training import atomic_save
from spora.named_address import address_basis_sha256, feature_files_sha256

SOURCE = ROOT/'runs/deepseek-spora-diverse-memory-replay-20261111'
OUTPUT = ROOT/'runs/deepseek-spora-recalibrate-name-router-20261111'
FRESH_TEMPLATES = ('在保存的信息中，{name}目前爱喝哪种饮料？',
                   '修改记录以前，{name}最喜欢哪种饮品？',
                   '修改之后，{name}更喜欢喝什么？')
STEPS = 400
SEED = 99342


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base model mismatch')
    prior = json.loads((SOURCE/'result.json').read_text(encoding='utf-8'))
    for i in (6, 12):
        if isinstance(prior.get('layers'), dict):
            info = prior['layers'][str(i)]
            valid = sha(SOURCE/f'layer-{i}.pt') == info['candidate_sha256'] and sha(PAIR/f'layer-{i}.pt') == info['root_sha256']
        elif 'candidate_checkpoints' in prior:
            info = prior['candidate_checkpoints'][str(i)]
            valid = sha(SOURCE/f'layer-{i}.pt') == info['sha256']
        else:
            raise ValueError('Unsupported candidate provenance report')
        if not valid:
            raise ValueError('Candidate provenance mismatch')
    loc_info = json.loads((LOCATOR/'result.json').read_text(encoding='utf-8'))
    proj_info = json.loads((PROJECTION/'result.json').read_text(encoding='utf-8'))
    if sha(LOCATOR/'locator.pt') != loc_info['locator_sha256'] or sha(PROJECTION/'projections.pt') != proj_info['projection_sha256']:
        raise ValueError('Prior router digest mismatch')
    old_loc_data = torch.load(LOCATOR/'locator.pt', map_location='cpu', weights_only=True)
    old_proj = torch.load(PROJECTION/'projections.pt', map_location='cpu', weights_only=True)
    config_hash = feature_files_sha256(MODEL)
    if any(x.get('weight_sha256') != WEIGHT_SHA or x.get('feature_config_sha256') != config_hash
           for x in (old_loc_data, old_proj)):
        raise ValueError('Prior router feature fingerprint mismatch')
    organism, _, _ = load(NAMED/'individual.pt', 'cpu')
    basis_hash = address_basis_sha256(organism)
    bases = torch.stack([F.normalize(site.current_basis.detach().float(), dim=-1) for site in organism.memories])
    bound = list(zip(NAMES+UNKNOWN[:2], KEYS+NEW_KEYS))
    training = []
    for name, key in bound:
        training += [{'text': t.format(name=name), 'name': name, 'key': key, 'group': 'train_positive'}
                     for t in TRAIN_TEMPLATES[:(4 if name in NAMES else 2)]]
        for choices in TIME_TRAIN.values():
            training += [{'text': t.replace(NAMES[0], name), 'name': name, 'key': key, 'group': 'train_positive'}
                         for t in choices[:2]]
    training += [{'text': q, 'name': None, 'key': None, 'group': 'train_negative'} for q in TRAIN_NEGATIVE]
    training += [{'text': f'比较{NAMES[i]}与{NAMES[(i+1)%len(NAMES)]}各自的饮品偏好。',
                  'name': None, 'key': None, 'group': 'train_multiple'} for i in range(len(NAMES))]
    held = []
    for name, key in bound:
        held += [{'text': t.format(name=name), 'name': name, 'key': key, 'group': 'numeric_known'} for t in HELD_TEMPLATES]
        held += [{'text': t.format(name=name), 'name': name, 'key': key, 'group': 'fresh_beverage'} for t in FRESH_TEMPLATES]
    held += [{'text': x['text'], 'name': NAMES[0], 'key': KEYS[0], 'group': 'temporal_development'}
             for x in examples(TIME_HELD)]
    for name in UNKNOWN[2:]:
        held += [{'text': t.format(name=name), 'name': name, 'key': None, 'group': 'unbound'} for t in FRESH_TEMPLATES]
    held += [{'text': q, 'name': None, 'key': None, 'group': 'no_name'} for q in HELD_NEGATIVE]
    held += [{'text': q, 'name': None, 'key': None, 'group': 'multiple_names'} for q in HELD_MULTIPLE]
    if {x['text'] for x in training} & {x['text'] for x in held}:
        raise ValueError('Training and held text overlap')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    def label_rows(rows):
        questions = [row['text'] for row in rows]
        texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False,
                                               add_generation_prompt=True) for q in questions]
        encoded = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False,
                            return_offsets_mapping=True)
        positive = [i for i, row in enumerate(rows) if row['name'] is not None]
        spans, crossings = oracle_span_mask(encoded['offset_mapping'][positive], encoded['attention_mask'][positive],
                                            [texts[i] for i in positive], [questions[i] for i in positive],
                                            [rows[i]['name'] for i in positive])
        return questions, encoded, positive, spans, {positive[i] for i, crosses in enumerate(crossings) if crosses}

    rows = training+held
    questions, encoded, positive, spans, crossed = label_rows(rows)
    dropped_train = [training[i]['text'] for i in sorted(crossed) if i < len(training)]
    if dropped_train:
        training = [row for i, row in enumerate(training) if i not in crossed]
        rows = training+held
        questions, encoded, positive, spans, crossed = label_rows(rows)
        if any(i < len(training) for i in crossed):
            raise ValueError('Unrepresentable training labels remain')
    truth = torch.zeros_like(encoded['attention_mask'], dtype=torch.bool)
    truth[positive] = spans
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    blocks = {i: _load_candidate(SOURCE/f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    variant = replacement_feature_variant_sha256(MODEL, blocks)
    for i, block in blocks.items():
        base.model.layers[i] = block
    features, masks = extract(base, tokenizer, questions, batch_size=4)
    if features.shape[2] != truth.shape[1] or bool((truth & ~masks).any()):
        raise ValueError('Feature/token span mismatch')
    n_train = len(training)
    vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
    train_x = vectors[:n_train][masks[:n_train]]
    train_y = truth[:n_train][masks[:n_train]].float()
    old_locator = nn.Linear(train_x.shape[-1], 1).eval()
    old_locator.load_state_dict(old_loc_data['state'], strict=True)
    locator = nn.Linear(train_x.shape[-1], 1)
    locator.load_state_dict(old_loc_data['state'], strict=True)
    optimizer = torch.optim.AdamW(locator.parameters(), lr=0.0003, weight_decay=0.01)
    positive_weight = (train_y.numel()-train_y.sum())/train_y.sum()
    trace = []
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(locator(train_x).squeeze(-1), train_y, pos_weight=positive_weight)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite locator loss')
        loss.backward()
        optimizer.step()
        if step in (0, 99, 199, 399):
            trace.append({'step': step+1, 'loss': float(loss.detach())})
    locator.eval()
    positive_train = [i for i, row in enumerate(training) if row['key'] is not None]
    name_vectors = pooled(features[positive_train], truth[positive_train])
    targets = torch.stack([bases[:, training[i]['key']] for i in positive_train])
    adapted_weights = fit_binding(name_vectors, targets)
    _, train_scores = predict(name_vectors, adapted_weights, bases)
    adapted_threshold = float(train_scores.min())-1e-6
    if not 0 <= adapted_threshold <= 1:
        raise ValueError('Invalid learned threshold')
    methods = (('prior_router', old_locator, old_proj['weights']['oracle_name_span'],
                proj_info['results']['oracle_name_span']['threshold']),
               ('recalibrated_router', locator, adapted_weights, adapted_threshold))
    results = {}
    with torch.inference_mode():
        for method, model, weights, threshold in methods:
            logits = model(vectors[n_train:]).squeeze(-1).sigmoid()
            outcomes = []
            for j, row in enumerate(held):
                selected, single = single_span_mask(logits[j], masks[n_train+j], old_loc_data['threshold'])
                item = {'question': row['text'], 'group': row['group'], 'expected_key': row['key'],
                        'span_exact': (bool(torch.equal(selected, truth[n_train+j]))
                                       if row['name'] and n_train+j not in crossed else None),
                        'span_representable': n_train+j not in crossed if row['name'] else None,
                        'single_span': single, 'accepted': False}
                if single:
                    span_vector = pooled(features[n_train+j:n_train+j+1], selected.unsqueeze(0))
                    indices, sims = predict(span_vector, weights, bases)
                    item['indices'], item['similarities'] = indices[0].tolist(), sims[0].tolist()
                    item['accepted'] = bool((sims[0] >= threshold).all() and indices[0, 0] == indices[0, 1])
                item['correct_address'] = row['key'] is not None and item['accepted'] and item.get('indices') == [row['key']]*2
                outcomes.append(item)
            groups = {}
            for group in sorted({row['group'] for row in held}):
                subset = [row for row in outcomes if row['group'] == group]
                groups[group] = {'total': len(subset), 'representable': sum(row['span_representable'] is True for row in subset),
                                 'exact_span': sum(row['span_exact'] is True for row in subset),
                                 'accepted': sum(row['accepted'] for row in subset),
                                 'correct_address': sum(row['correct_address'] for row in subset)}
            results[method] = {'threshold': threshold, 'groups': groups, 'rows': outcomes}
            print(json.dumps({'method': method, 'groups': groups}, ensure_ascii=False), flush=True)
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT/'router.pt'
    atomic_save({'format': 'spora-recalibrated-name-router-v1', 'base_sha256': WEIGHT_SHA,
                 'feature_config_sha256': config_hash, 'feature_variant_sha256': variant,
                 'basis_sha256': basis_hash, 'feature_layers': [6, 18], 'threshold': adapted_threshold,
                 'locator_threshold': old_loc_data['threshold'],
                 'locator_state': {k: v.detach().cpu() for k, v in locator.state_dict().items()},
                 'projection': adapted_weights.detach().cpu()}, checkpoint)
    report = {'results': results, 'base_sha256': WEIGHT_SHA, 'feature_config_sha256': config_hash,
              'feature_variant_sha256': variant, 'basis_sha256': basis_hash,
              'candidate_sha256': {str(i): sha(SOURCE/f'layer-{i}.pt') for i in (6, 12)},
              'old_locator_sha256': sha(LOCATOR/'locator.pt'), 'old_projection_sha256': sha(PROJECTION/'projections.pt'),
              'router_sha256': sha(checkpoint), 'train_positive': len(positive_train),
              'train_negative': n_train-len(positive_train), 'dropped_unrepresentable_training': dropped_train,
              'unrepresentable_held': [rows[i]['text'] for i in sorted(crossed) if i >= n_train],
              'held': len(held), 'trace': trace,
              'elapsed_seconds': time.perf_counter()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['All ten bound names and their addresses were supervised during training.',
                              'Most names and some held templates were used in prior development.',
                              'Fresh beverage templates are synthetic and only one fact domain is tested.',
                              'Multiple-name and unbound samples are few; not a safe open-world router.',
                              'No generation, event-time selection, autonomous writing or conversation evaluated.',
                              'Threshold selected on training similarities, not calibrated uncertainty.']}
    with (OUTPUT/'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    run()
