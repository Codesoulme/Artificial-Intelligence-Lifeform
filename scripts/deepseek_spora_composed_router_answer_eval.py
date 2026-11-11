"""Evaluate the experimental intent, entity and name gates with memory answers."""
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
from scripts.deepseek_learned_name_span import single_span_mask
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_named_binding_migrate import OUTPUT as NAMED
from scripts.deepseek_named_binding_probe import KEYS, NAMES, pooled, predict
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_gate_fresh_intent_audit import KNOWN
from scripts.deepseek_spora_memory_answer_distill import SOURCE_CHECKPOINT, _load_candidate, _memory_args
from scripts.deepseek_spora_recalibrate_name_router import SOURCE, OUTPUT as ROUTER
from scripts.deepseek_temporal_intent_router import TRAIN, HELD, examples
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.current_memory_status import current_memory_status
from spora.growing import load
from spora.integrated_checkpoint import load_integrated, replacement_feature_variant_sha256
from spora.internal_memory import observe_sourced
from scripts.spora_event_scan_selector import select_event
from spora.named_address import address_basis_sha256, feature_files_sha256
from spora.temporal_routing import CURRENT, PAST, LATER, route_spec

ENTITY = ROOT/'runs/deepseek-spora-single-entity-gate-v2-20261111'
INTENT = ROOT/'runs/deepseek-spora-read-intent-gate-v2-20261111'
OUTPUT = ROOT/'runs/deepseek-spora-composed-router-answer-eval-20261111.json'
TEMPORAL_CHECKPOINT = None
DYNAMIC_EVENT_SELECTION = False
NEGATIVE = ('不要告诉我小林现在喜欢什么饮料。', '把小林的饮料改成牛奶。', '给小林写一句鼓励的话。',
            '小林和阿宁分别爱喝什么？', '小青以前喜欢喝什么？', '你会说中文吗？')


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base model mismatch')
    router_report = json.loads((ROUTER/'result.json').read_text(encoding='utf-8'))
    entity_report = json.loads((ENTITY/'result.json').read_text(encoding='utf-8'))
    intent_report = json.loads((INTENT/'result.json').read_text(encoding='utf-8'))
    for path, expected in ((ROUTER/'router.pt', router_report['router_sha256']),
                           (ENTITY/'gate.pt', entity_report['gate_sha256']),
                           (INTENT/'intent.pt', intent_report['intent_sha256'])):
        if sha(path) != expected:
            raise ValueError('Router or gate digest mismatch')
    router = torch.load(ROUTER/'router.pt', map_location='cpu', weights_only=True)
    entity_data = torch.load(ENTITY/'gate.pt', map_location='cpu', weights_only=True)
    intent_data = torch.load(INTENT/'intent.pt', map_location='cpu', weights_only=True)
    config_hash = feature_files_sha256(MODEL)
    named, _, _ = load(NAMED/'individual.pt', 'cpu')
    if (router['base_sha256'] != WEIGHT_SHA or router['feature_config_sha256'] != config_hash
            or router['basis_sha256'] != address_basis_sha256(named)
            or entity_data['router_sha256'] != sha(ROUTER/'router.pt')
            or intent_data['router_sha256'] != sha(ROUTER/'router.pt')
            or intent_data['entity_gate_sha256'] != sha(ENTITY/'gate.pt')
            or entity_data['feature_variant_sha256'] != router['feature_variant_sha256']
            or intent_data['feature_variant_sha256'] != router['feature_variant_sha256']):
        raise ValueError('Router composition provenance mismatch')
    train, held = examples(TRAIN), examples(HELD)
    held += [{'text': KNOWN[0].format(name=NAMES[0]), 'label': CURRENT},
             {'text': KNOWN[1].format(name=NAMES[0]), 'label': PAST}]
    questions = [r['text'] for r in train+held] + list(NEGATIVE)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    integrated, organism, _ = load_integrated(SOURCE_CHECKPOINT, base, MODEL, individual_device='cuda')
    if address_basis_sha256(organism) != router['basis_sha256']:
        raise ValueError('Individual and name router address bases differ')
    bases = torch.stack([F.normalize(site.current_basis.detach().cpu().float(), dim=-1) for site in organism.memories])
    state = fresh_controlled_state(organism)
    for value in (1, 6):
        state, decision = observe_sourced(organism, state, KEYS[0], value, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('Controlled memory write rejected')
    original_features, original_mask = extract(base, tokenizer, [r['text'] for r in train], batch_size=2)
    train_x = pool(original_features, original_mask).float()
    labels = torch.tensor([r['label'] for r in train], dtype=torch.long)
    torch.manual_seed(4876)
    temporal = nn.Linear(train_x.shape[-1], 3)
    opt = torch.optim.AdamW(temporal.parameters(), lr=0.003, weight_decay=0.02)
    for _ in range(600):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(temporal(train_x), labels) + 0.02*temporal.weight.square().sum()
        loss.backward()
        opt.step()
    temporal.eval()
    blocks = {i: _load_candidate(SOURCE/f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    if replacement_feature_variant_sha256(MODEL, blocks) != router['feature_variant_sha256']:
        raise ValueError('Structural variant mismatch')
    temporal_data = None
    if TEMPORAL_CHECKPOINT is not None:
        temporal_report = json.loads((TEMPORAL_CHECKPOINT.parent/'result.json').read_text(encoding='utf-8'))
        if sha(TEMPORAL_CHECKPOINT) != temporal_report['temporal_sha256']:
            raise ValueError('Temporal gate checkpoint digest mismatch')
        temporal_data = torch.load(TEMPORAL_CHECKPOINT, map_location='cpu', weights_only=True)
        if (temporal_data.get('format') != 'spora-bounded-temporal-gate-v1'
                or temporal_data.get('base_sha256') != WEIGHT_SHA
                or temporal_data.get('feature_variant_sha256') != router['feature_variant_sha256']
                or temporal_data.get('classes') != ['current', 'past', 'later', 'abstain']):
            raise ValueError('Temporal gate provenance mismatch')
        temporal = nn.Linear(train_x.shape[-1], 4).eval()
        temporal.load_state_dict(temporal_data['state'], strict=True)
    for i, block in blocks.items():
        base.model.layers[i] = block
    features, masks = extract(base, tokenizer, questions[len(train):], batch_size=2)
    qvec = pool(features, masks).float()
    token_vec = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
    locator = nn.Linear(token_vec.shape[-1], 1).eval()
    locator.load_state_dict(router['locator_state'], strict=True)
    entity = nn.Linear(qvec.shape[-1], 1).eval()
    entity.load_state_dict(entity_data['gate_state'], strict=True)
    intent = nn.Linear(qvec.shape[-1], 1).eval()
    intent.load_state_dict(intent_data['state'], strict=True)
    results = []
    with torch.inference_mode():
        loc_scores = locator(token_vec).squeeze(-1).sigmoid()
        entity_scores = entity(qvec).squeeze(-1).sigmoid()
        intent_logits = intent(qvec).squeeze(-1)
        time_logits = temporal(qvec)
        if temporal_data is None:
            predicted_times = time_logits.argmax(-1).tolist()
        else:
            probs = time_logits.softmax(-1)
            winners = probs.argmax(-1)
            predicted_times = [int(winner) if int(winner) != 3 and float(probs[i, :3].max()) >= temporal_data['threshold']
                               else 3 for i, winner in enumerate(winners)]
        for i, question in enumerate(questions[len(train):]):
            expected = held[i]['label'] if i < len(held) else None
            selected, single = single_span_mask(loc_scores[i], masks[i], router['locator_threshold'])
            row = {'question': question, 'expected_intent': expected, 'predicted_intent': predicted_times[i],
                   'intent_logit': float(intent_logits[i]), 'entity_score': float(entity_scores[i]),
                   'accepted': False, 'reason': None, 'answer': None}
            if row['intent_logit'] < intent_data['threshold']:
                row['reason'] = 'not_confident_read_intent'
            elif row['entity_score'] < entity_data['gate_threshold']:
                row['reason'] = 'not_single_entity'
            elif predicted_times[i] == 3:
                row['reason'] = 'ambiguous_temporal'
            elif not single:
                row['reason'] = 'no_single_name_span'
            else:
                v = pooled(features[i:i+1], selected.unsqueeze(0))
                indices, sims = predict(v, router['projection'], bases)
                row['indices'], row['similarities'] = indices[0].tolist(), sims[0].tolist()
                if not bool(indices[0, 0] == indices[0, 1] and (sims[0] >= entity_data['address_threshold']).all()):
                    row['reason'] = 'name_binding_rejected'
                else:
                    key = int(indices[0, 0])
                    if current_memory_status(organism, state, key)['status'] != 'known':
                        row['reason'] = 'unknown_memory'
                    else:
                        _, event = route_spec(predicted_times[i])
                        mode = 'current' if event is None else 'past' if predicted_times[i] == PAST else 'later'
                        if DYNAMIC_EVENT_SELECTION and event is not None:
                            order = 'effective_previous' if predicted_times[i] == PAST else 'effective_latest'
                            selection = select_event(organism, state, key, order, as_of=state.cursor-1)
                            row['memory_event_selection'] = selection
                            if selection['status'] != 'experimental_candidate':
                                row['reason'] = 'memory_event_abstain'
                                row['route_correct'] = False
                                row['answer_correct'] = False
                                results.append(row)
                                continue
                            event = selection['event_time']
                        row['accepted'], row['reason'], row['predicted_key'] = True, 'routed', key
                        args = _memory_args(organism, state, {'key': key, 'intent': mode,
                                                               'event_time': 0 if event is None else event})
                        prefix = closed_think_inputs(tokenizer, question, 'cuda')
                        ids = integrated.generate(**prefix, **args, max_new_tokens=8, do_sample=False,
                                                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                        row['answer'] = tokenizer.decode(ids[0, prefix['input_ids'].shape[1]:],
                                                         skip_special_tokens=True).strip()
            row['route_correct'] = (expected is None and not row['accepted']) or (
                expected is not None and row['accepted'] and row.get('predicted_key') == KEYS[0]
                and row['predicted_intent'] == expected)
            target = '咖啡' if expected == PAST else '水'
            row['answer_correct'] = expected is not None and row['answer'] in (target, target+'.', target+'。')
            results.append(row)
    report = {'rows': results, 'positive_total': len(held), 'negative_total': len(NEGATIVE),
              'positive_routes_correct': sum(r['route_correct'] for r in results[:len(held)]),
              'positive_answers_correct': sum(r['answer_correct'] for r in results[:len(held)]),
              'negative_rejected': sum(not r['accepted'] for r in results[len(held):]),
              'base_sha256': WEIGHT_SHA, 'source_sha256': sha(SOURCE_CHECKPOINT),
              'router_sha256': sha(ROUTER/'router.pt'), 'entity_sha256': sha(ENTITY/'gate.pt'),
              'intent_sha256': sha(INTENT/'intent.pt'), 'feature_variant_sha256': router['feature_variant_sha256'],
              'temporal_sha256': sha(TEMPORAL_CHECKPOINT) if TEMPORAL_CHECKPOINT is not None else None,
              'candidate_sha256': {str(i): sha(SOURCE/f'layer-{i}.pt') for i in (6, 12)},
              'elapsed_seconds': time.perf_counter()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Reused authored development questions, one entity and two programmed facts.',
                              'Experimental code maps temporal classes to event times 0 and 1.',
                              'No autonomous writing, real dialogue history, identity or open-world validation.',
                              'Closed reasoning and eight-token answer cap.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('positive_total', 'positive_routes_correct',
                                               'positive_answers_correct', 'negative_total', 'negative_rejected')}), flush=True)


if __name__ == '__main__':
    run()
