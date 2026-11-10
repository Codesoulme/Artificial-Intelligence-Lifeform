"""Measure old language-to-memory routers after SPORA layer replacement."""
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
from scripts.deepseek_learned_name_span import OUTPUT as LOCATOR_DIR, HELD_MULTIPLE, HELD_NEGATIVE, single_span_mask
from scripts.deepseek_name_span_probe import OUTPUT as PROJECTION_DIR, oracle_span_mask
from scripts.deepseek_named_binding_migrate import OUTPUT as NAMED_DIR
from scripts.deepseek_named_binding_probe import HELD_TEMPLATES, KEYS, NAMES, UNKNOWN, pooled, predict
from scripts.deepseek_named_incremental_probe import NEW_KEYS
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, _load_candidate
from scripts.deepseek_temporal_intent_router import TRAIN as TIME_TRAIN, HELD as TIME_HELD, examples
from spora.growing import load
from spora.named_address import feature_files_sha256

NEW = ROOT/'runs/deepseek-spora-diverse-memory-replay-20261111'
OUTPUT = ROOT/'runs/deepseek-spora-router-feature-shift-20261111.json'


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base weight hash mismatch')
    replay = json.loads((NEW/'result.json').read_text(encoding='utf-8'))
    for i in (6, 12):
        meta = replay['layers'][str(i)]
        if sha(NEW/f'layer-{i}.pt') != meta['candidate_sha256'] or sha(PAIR/f'layer-{i}.pt') != meta['root_sha256']:
            raise ValueError('Candidate provenance mismatch')
    locator_path, projection_path = LOCATOR_DIR/'locator.pt', PROJECTION_DIR/'projections.pt'
    locator_info = json.loads((LOCATOR_DIR/'result.json').read_text(encoding='utf-8'))
    projection_info = json.loads((PROJECTION_DIR/'result.json').read_text(encoding='utf-8'))
    if sha(locator_path) != locator_info['locator_sha256'] or sha(projection_path) != projection_info['projection_sha256']:
        raise ValueError('Existing router checkpoint digest mismatch')
    locator_data = torch.load(locator_path, map_location='cpu', weights_only=True)
    projection = torch.load(projection_path, map_location='cpu', weights_only=True)
    fingerprint = feature_files_sha256(MODEL)
    if any(data.get('weight_sha256') != WEIGHT_SHA or data.get('feature_config_sha256') != fingerprint
           for data in (locator_data, projection)):
        raise ValueError('Existing router feature provenance mismatch')
    organism, _, _ = load(NAMED_DIR/'individual.pt', 'cpu')
    bases = torch.stack([F.normalize(site.current_basis.detach().float(), dim=-1) for site in organism.memories])
    locator = nn.Linear(2*locator_data['feature_width'], 1).eval()
    locator.load_state_dict(locator_data['state'], strict=True)
    weights = projection['weights']['oracle_name_span']
    threshold = projection_info['results']['oracle_name_span']['threshold']
    time_train, time_held = examples(TIME_TRAIN), examples(TIME_HELD)
    singles = [(name, key, t.format(name=name)) for name, key in zip(NAMES+UNKNOWN[:2], KEYS+NEW_KEYS)
               for t in HELD_TEMPLATES]
    singles += [(name, None, t.format(name=name)) for name in UNKNOWN[2:] for t in HELD_TEMPLATES]
    questions = [x['text'] for x in time_train+time_held] + [x[2] for x in singles] + list(HELD_NEGATIVE) + list(HELD_MULTIPLE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False,
                                           add_generation_prompt=True) for q in questions]
    encoded = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False, return_offsets_mapping=True)
    off = len(time_train)+len(time_held)
    truth, crossings = oracle_span_mask(encoded['offset_mapping'][off:off+len(singles)],
        encoded['attention_mask'][off:off+len(singles)], texts[off:off+len(singles)],
        questions[off:off+len(singles)], [x[0] for x in singles])
    if any(crossings):
        raise ValueError('Tokenizer crosses a reference name boundary')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    original = {i: base.model.layers[i] for i in (6, 12)}
    old_pair = {i: _load_candidate(PAIR/f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    new_pair = {i: _load_candidate(NEW/f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    features = {}
    masks = None
    for mode, blocks in (('original', original), ('old_pair', old_pair), ('new_pair', new_pair)):
        for i in (6, 12):
            base.model.layers[i] = blocks[i]
        current, current_mask = extract(base, tokenizer, questions, batch_size=2)
        if masks is not None and not torch.equal(masks, current_mask):
            raise ValueError('Feature tokenization changed between modes')
        features[mode], masks = current, current_mask
        print(json.dumps({'mode': mode, 'feature_examples': len(questions)}), flush=True)
    train_features = pool(features['original'][:len(time_train)], masks[:len(time_train)]).float()
    labels = torch.tensor([row['label'] for row in time_train], dtype=torch.long)
    torch.manual_seed(4876)
    router = nn.Linear(train_features.shape[-1], 3)
    optimizer = torch.optim.AdamW(router.parameters(), lr=0.003, weight_decay=0.02)
    for _ in range(600):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(router(train_features), labels) + 0.02*router.weight.square().sum()
        loss.backward()
        optimizer.step()
    router.eval()
    results = {}
    with torch.inference_mode():
        for mode, tensors in features.items():
            time_x = pool(tensors[len(time_train):off], masks[len(time_train):off]).float()
            predictions = router(time_x).argmax(-1).tolist()
            time_rows = [{'question': row['text'], 'expected': row['label'], 'predicted': prediction,
                          'correct': prediction == row['label']} for row, prediction in zip(time_held, predictions)]
            vectors = torch.cat([F.layer_norm(tensors[:, i].float(), (tensors.shape[-1],)) for i in range(2)], dim=-1)
            scores = locator(vectors[off:]).squeeze(-1).sigmoid()
            rows = []
            for j, question in enumerate(questions[off:]):
                selected, single = single_span_mask(scores[j], masks[off+j], locator_data['threshold'])
                if j < len(singles):
                    _, expected, _ = singles[j]
                    group = 'known' if expected is not None else 'unbound'
                    exact = bool(torch.equal(selected, truth[j]))
                else:
                    expected, exact = None, None
                    group = 'no_name' if j < len(singles)+len(HELD_NEGATIVE) else 'multiple_names'
                result = {'question': question, 'group': group, 'expected_key': expected,
                          'span_exact': exact, 'single_span': single, 'accepted': False}
                if single:
                    name_vector = pooled(tensors[off+j:off+j+1], selected.unsqueeze(0))
                    indices, similarity = predict(name_vector, weights, bases)
                    result['indices'], result['similarities'] = indices[0].tolist(), similarity[0].tolist()
                    result['accepted'] = bool((similarity[0] >= threshold).all() and indices[0, 0] == indices[0, 1])
                result['correct_address'] = (expected is not None and result['accepted']
                                             and result.get('indices') == [expected, expected])
                rows.append(result)
            grouped = {}
            for group in ('known', 'unbound', 'no_name', 'multiple_names'):
                subset = [row for row in rows if row['group'] == group]
                grouped[group] = {'total': len(subset), 'exact_span': sum(row['span_exact'] is True for row in subset),
                                  'accepted': sum(row['accepted'] for row in subset),
                                  'correct_address': sum(row['correct_address'] for row in subset)}
            results[mode] = {'temporal_correct': sum(row['correct'] for row in time_rows),
                             'temporal_total': len(time_rows), 'temporal_rows': time_rows,
                             'name_groups': grouped, 'name_rows': rows}
            print(json.dumps({'mode': mode, 'time': results[mode]['temporal_correct'],
                              'names': grouped['known']['correct_address']}), flush=True)
    report = {'results': results, 'base_sha256': WEIGHT_SHA, 'feature_config_sha256': fingerprint,
              'locator_sha256': sha(locator_path), 'projection_sha256': sha(projection_path),
              'old_pair_sha256': {str(i): sha(PAIR/f'layer-{i}.pt') for i in (6, 12)},
              'new_pair_sha256': {str(i): sha(NEW/f'layer-{i}.pt') for i in (6, 12)},
              'named_individual_sha256': sha(NAMED_DIR/'individual.pt'),
              'elapsed_seconds': time.perf_counter()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['All templates and names were previously used during development.',
                              'Temporal classifier is retrained on original features only in this run.',
                              'No fine-tuning on replacement features; tests feature compatibility.',
                              'Read-only feature tests with no SPORA memory injection, writes or answer generation.',
                              'Unknown and multiple-name examples are small and do not establish safety.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    run()
