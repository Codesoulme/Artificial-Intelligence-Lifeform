"""Audit the experimental name gate on fresh wording and non-memory intents."""
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
from scripts.deepseek_spora_single_entity_gate import MULTI_TRAIN
from scripts.deepseek_temporal_intent_router import TRAIN as TIME_TRAIN
from spora.growing import load
from spora.integrated_checkpoint import replacement_feature_variant_sha256
from spora.named_address import address_basis_sha256, feature_files_sha256

GATE = ROOT/'runs/deepseek-spora-single-entity-gate-v2-20261111'
OUTPUT = ROOT/'runs/deepseek-spora-gate-fresh-intent-audit-20261111.json'
KNOWN = ('我想查阅{name}现在的饮品偏好。', '说说更早记录的{name}喜欢什么饮料。')
NON_MEMORY = ('给{name}写一句鼓励的话。', '如果{name}向你打招呼，你会怎么回应？')
UNBOUND_NAMES = ('林默', '桐雨', '星河', '柳鸣', '若竹', '柯岚')
MULTI = ('先说{a}爱喝什么，再说{b}爱喝什么。', '请分别查询{a}与{b}的饮品偏好。')
NO_NAME = ('你能不能讲一件有趣的小事？', '告诉我今天的计划。', '之前我们聊过哪些话题？',
           '现在记录的内容是什么？', '给我一些睡前放松的方法。', '解释一下如何记忆。')


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base model mismatch')
    gate_info = json.loads((GATE/'result.json').read_text(encoding='utf-8'))
    if sha(GATE/'gate.pt') != gate_info['gate_sha256']:
        raise ValueError('Gate digest mismatch')
    gate_data = torch.load(GATE/'gate.pt', map_location='cpu', weights_only=True)
    if sha(ROUTER/'router.pt') != gate_data['router_sha256']:
        raise ValueError('Gate and name router mismatch')
    router = torch.load(ROUTER/'router.pt', map_location='cpu', weights_only=True)
    config_hash = feature_files_sha256(MODEL)
    organism, _, _ = load(NAMED/'individual.pt', 'cpu')
    if (router['base_sha256'] != WEIGHT_SHA or router['feature_config_sha256'] != config_hash
            or router['basis_sha256'] != address_basis_sha256(organism)
            or gate_data['feature_variant_sha256'] != router['feature_variant_sha256']):
        raise ValueError('Router provenance mismatch')
    bases = torch.stack([F.normalize(site.current_basis.detach().float(), dim=-1) for site in organism.memories])
    rows = []
    for name, key in zip(NAMES+UNKNOWN[:2], KEYS+NEW_KEYS):
        rows += [{'question': t.format(name=name), 'group': 'known_memory', 'expected_key': key} for t in KNOWN]
        rows += [{'question': t.format(name=name), 'group': 'known_non_memory', 'expected_key': None} for t in NON_MEMORY]
    for name in UNBOUND_NAMES:
        rows += [{'question': t.format(name=name), 'group': 'unbound_memory', 'expected_key': None} for t in KNOWN]
    for i, name in enumerate(NAMES[:6]):
        rows += [{'question': t.format(a=name, b=NAMES[(i+1)%len(NAMES)]), 'group': 'multiple_names',
                  'expected_key': None} for t in MULTI]
    rows += [{'question': q, 'group': 'no_name', 'expected_key': None} for q in NO_NAME]
    training = set(TRAIN_NEGATIVE)
    for name in NAMES+UNKNOWN[:2]:
        training.update(t.format(name=name) for t in TRAIN_TEMPLATES)
        for choices in TIME_TRAIN.values():
            training.update(t.replace(NAMES[0], name) for t in choices[:2])
    for i, name in enumerate(NAMES):
        training.update(t.format(a=name, b=NAMES[(i+1)%len(NAMES)]) for t in MULTI_TRAIN)
    prior = json.loads((GATE/'result.json').read_text(encoding='utf-8'))
    prior_questions = {row['question'] for part in ('calibration', 'held')
                       for group in prior['results']['gated'][part].values() for row in group['rows']}
    if {r['question'] for r in rows} & (training | prior_questions):
        raise ValueError('Fresh audit questions overlap known training or earlier evaluation text')
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
        raise ValueError('Model variant mismatch')
    for i, block in blocks.items():
        base.model.layers[i] = block
    features, masks = extract(base, tokenizer, [row['question'] for row in rows], batch_size=4)
    question_vectors = pool(features, masks).float()
    token_vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
    gate = nn.Linear(question_vectors.shape[-1], 1).eval()
    gate.load_state_dict(gate_data['gate_state'], strict=True)
    locator = nn.Linear(token_vectors.shape[-1], 1).eval()
    locator.load_state_dict(router['locator_state'], strict=True)
    outcomes = []
    with torch.inference_mode():
        gate_scores = gate(question_vectors).squeeze(-1).sigmoid()
        loc_scores = locator(token_vectors).squeeze(-1).sigmoid()
        for index, row in enumerate(rows):
            selected, single = single_span_mask(loc_scores[index], masks[index], router['locator_threshold'])
            item = {**row, 'gate_score': float(gate_scores[index]), 'single_span': single,
                    'indices': None, 'address_score': None, 'accepted': False}
            if single:
                vector = pooled(features[index:index+1], selected.unsqueeze(0))
                indices, scores = predict(vector, router['projection'], bases)
                item['indices'], item['address_score'] = indices[0].tolist(), float(scores[0].min())
                item['accepted'] = bool(item['gate_score'] >= gate_data['gate_threshold']
                                        and indices[0, 0] == indices[0, 1]
                                        and item['address_score'] >= gate_data['address_threshold'])
            item['correct_address'] = row['expected_key'] is not None and item['accepted'] and item['indices'] == [row['expected_key']]*2
            outcomes.append(item)
    groups = {}
    for group in sorted({r['group'] for r in rows}):
        subset = [r for r in outcomes if r['group'] == group]
        groups[group] = {'total': len(subset), 'accepted': sum(r['accepted'] for r in subset),
                         'correct_address': sum(r['correct_address'] for r in subset)}
    report = {'groups': groups, 'rows': outcomes, 'base_sha256': WEIGHT_SHA,
              'router_sha256': sha(ROUTER/'router.pt'), 'gate_sha256': sha(GATE/'gate.pt'),
              'feature_variant_sha256': router['feature_variant_sha256'],
              'elapsed_seconds': time.perf_counter()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Synthetic fresh wording, known names and model weights already used during development.',
                              'Known-name non-memory requests are out of the single-entity gate objective.',
                              'No temporal classifier, native memory, generation or write path in this probe.',
                              'No claims about open-world entities, identity or security.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(groups, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
