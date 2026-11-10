"""Test learned name/time routing through addressed SPORA answer generation."""
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
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_learned_name_span import OUTPUT as LOCATOR, single_span_mask
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_name_span_probe import OUTPUT as PROJECTION
from scripts.deepseek_named_binding_probe import KEYS, NAMES, pooled, predict
from scripts.deepseek_named_binding_migrate import OUTPUT as NAMED
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, SOURCE_CHECKPOINT, _load_candidate, _memory_args
from scripts.deepseek_temporal_intent_router import TRAIN, HELD, examples
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.current_memory_status import current_memory_status
from spora.growing import load
from spora.integrated_checkpoint import load_integrated
from spora.internal_memory import observe_sourced
from spora.named_address import address_basis_sha256, feature_files_sha256
from spora.temporal_routing import CURRENT, PAST, LATER, route_spec

NEW = ROOT/'runs/deepseek-spora-diverse-memory-replay-20261111'
OUTPUT = ROOT/'runs/deepseek-spora-predicted-route-answer-probe-20261111.json'
NEGATIVE = ('我今天有点累，能陪我聊聊吗？', '小青以前喜欢喝什么？', '小林和阿宁各喜欢什么饮料？')


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base model hash mismatch')
    replay = json.loads((NEW/'result.json').read_text(encoding='utf-8'))
    for i in (6, 12):
        entry = replay['layers'][str(i)]
        if sha(NEW/f'layer-{i}.pt') != entry['candidate_sha256'] or sha(PAIR/f'layer-{i}.pt') != entry['root_sha256']:
            raise ValueError('Candidate provenance mismatch')
    loc_data = torch.load(LOCATOR/'locator.pt', map_location='cpu', weights_only=True)
    proj_data = torch.load(PROJECTION/'projections.pt', map_location='cpu', weights_only=True)
    loc_report = json.loads((LOCATOR/'result.json').read_text(encoding='utf-8'))
    proj_report = json.loads((PROJECTION/'result.json').read_text(encoding='utf-8'))
    fingerprint = feature_files_sha256(MODEL)
    if (sha(LOCATOR/'locator.pt') != loc_report['locator_sha256']
            or sha(PROJECTION/'projections.pt') != proj_report['projection_sha256']
            or any(x.get('weight_sha256') != WEIGHT_SHA or x.get('feature_config_sha256') != fingerprint
                   for x in (loc_data, proj_data))):
        raise ValueError('Name router provenance mismatch')
    named, _, _ = load(NAMED/'individual.pt', 'cpu')
    named_basis = address_basis_sha256(named)
    locator = nn.Linear(2*loc_data['feature_width'], 1).eval()
    locator.load_state_dict(loc_data['state'], strict=True)
    weights = proj_data['weights']['oracle_name_span']
    threshold = proj_report['results']['oracle_name_span']['threshold']
    train, held = examples(TRAIN), examples(HELD)
    questions = [row['text'] for row in train+held] + list(NEGATIVE)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    integrated, organism, _ = load_integrated(SOURCE_CHECKPOINT, base, MODEL, individual_device='cuda')
    if address_basis_sha256(organism) != named_basis:
        raise ValueError('Name binding and target individual address basis mismatch')
    bases = torch.stack([F.normalize(site.current_basis.detach().cpu().float(), dim=-1) for site in organism.memories])
    state = fresh_controlled_state(organism)
    for value in (1, 6):
        state, decision = observe_sourced(organism, state, KEYS[0], value, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('Memory write rejected')
    original = {i: base.model.layers[i] for i in (6, 12)}
    old_pair = {i: _load_candidate(PAIR/f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    new_pair = {i: _load_candidate(NEW/f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    all_features, masks = {}, None
    for mode, blocks in (('original', original), ('old_pair', old_pair), ('new_pair', new_pair)):
        for i in (6, 12):
            base.model.layers[i] = blocks[i]
        feats, mask = extract(base, tokenizer, questions, batch_size=2)
        if masks is not None and not torch.equal(mask, masks):
            raise ValueError('Question tokenization changed')
        all_features[mode], masks = feats, mask
    train_x = pool(all_features['original'][:len(train)], masks[:len(train)]).float()
    labels = torch.tensor([r['label'] for r in train], dtype=torch.long)
    torch.manual_seed(4876)
    router = nn.Linear(train_x.shape[-1], 3)
    opt = torch.optim.AdamW(router.parameters(), lr=0.003, weight_decay=0.02)
    for _ in range(600):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(router(train_x), labels) + 0.02*router.weight.square().sum()
        loss.backward()
        opt.step()
    router.eval()
    results = {}
    with torch.inference_mode():
        for mode, blocks in (('original', original), ('old_pair', old_pair), ('new_pair', new_pair)):
            for i in (6, 12):
                base.model.layers[i] = blocks[i]
            feats = all_features[mode]
            vectors = torch.cat([F.layer_norm(feats[:, i].float(), (feats.shape[-1],)) for i in range(2)], dim=-1)
            loc_scores = locator(vectors[len(train):]).squeeze(-1).sigmoid()
            intents = router(pool(feats[len(train):], masks[len(train):]).float()).argmax(-1).tolist()
            rows = []
            for i, question in enumerate(questions[len(train):]):
                expected = held[i]['label'] if i < len(held) else None
                selected, contiguous = single_span_mask(loc_scores[i], masks[len(train)+i], loc_data['threshold'])
                row = {'question': question, 'expected_intent': expected, 'predicted_intent': intents[i],
                       'expected_key': KEYS[0] if expected is not None else None,
                       'accepted': False, 'reason': 'no_single_span', 'answer': None}
                if contiguous:
                    vec = pooled(feats[len(train)+i:len(train)+i+1], selected.unsqueeze(0))
                    indices, sims = predict(vec, weights, bases)
                    row['indices'], row['similarities'] = indices[0].tolist(), sims[0].tolist()
                    if not bool((sims[0] >= threshold).all() and indices[0, 0] == indices[0, 1]):
                        row['reason'] = 'name_binding_rejected'
                    else:
                        key = int(indices[0, 0])
                        if current_memory_status(organism, state, key)['status'] != 'known':
                            row['reason'] = 'unknown_memory'
                        else:
                            row['accepted'], row['reason'], row['predicted_key'] = True, 'routed', key
                            _, event = route_spec(intents[i])
                            intent = 'current' if event is None else 'past' if intents[i] == PAST else 'later'
                            args = _memory_args(organism, state, {'key': key, 'intent': intent,
                                                                   'event_time': 0 if event is None else event})
                            prefix = closed_think_inputs(tokenizer, question, 'cuda')
                            ids = integrated.generate(**prefix, **args, max_new_tokens=8, do_sample=False,
                                                      pad_token_id=tokenizer.pad_token_id,
                                                      eos_token_id=tokenizer.eos_token_id)
                            row['answer'] = tokenizer.decode(ids[0, prefix['input_ids'].shape[1]:],
                                                             skip_special_tokens=True).strip()
                row['route_correct'] = (expected is None and not row['accepted']) or (
                    expected is not None and row['accepted'] and row.get('predicted_key') == KEYS[0]
                    and row['predicted_intent'] == expected)
                answer = '咖啡' if expected == PAST else '水'
                row['answer_correct'] = expected is not None and row['answer'] in (answer, answer+'.', answer+'。')
                rows.append(row)
            results[mode] = {'route_correct': sum(x['route_correct'] for x in rows),
                             'answer_correct': sum(x['answer_correct'] for x in rows[:len(held)]),
                             'positive_total': len(held), 'negative_total': len(NEGATIVE), 'rows': rows}
            print(json.dumps({'mode': mode, 'route_correct': results[mode]['route_correct'],
                              'answer_correct': results[mode]['answer_correct']}), flush=True)
    report = {'results': results, 'base_sha256': WEIGHT_SHA, 'source_sha256': sha(SOURCE_CHECKPOINT),
              'locator_sha256': sha(LOCATOR/'locator.pt'), 'projection_sha256': sha(PROJECTION/'projections.pt'),
              'old_pair_sha256': {str(i): sha(PAIR/f'layer-{i}.pt') for i in (6, 12)},
              'new_pair_sha256': {str(i): sha(NEW/f'layer-{i}.pt') for i in (6, 12)},
              'basis_sha256': named_basis, 'feature_config_sha256': fingerprint,
              'elapsed_seconds': time.perf_counter()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Nine reused authored questions, one known entity and two programmed facts.',
                              'Only three negative queries, with no trained temporal abstention class.',
                              'The predicted temporal class is externally mapped to hardcoded event times 0 and 1.',
                              'Name and temporal routers were trained on original-base features, not replacement features.',
                              'Closed reasoning, eight-token answers; not a general conversation or autonomous write test.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    run()
