"""Audit the composed SPORA reader on fresh Chinese formulations."""
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
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_composed_router_answer_eval import (
    ENTITY, INTENT, SOURCE_CHECKPOINT, _load_candidate, _memory_args)
from scripts.deepseek_spora_gate_fresh_intent_audit import KNOWN
from scripts.deepseek_spora_recalibrate_name_router import SOURCE, OUTPUT as ROUTER
from scripts.deepseek_temporal_intent_router import CURRENT, LATER, PAST, TRAIN, examples
from scripts.open_base_baseline import INDIVIDUAL_SHA
from scripts.spora_event_scan_selector import select_event
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.current_memory_status import current_memory_status
from spora.growing import load
from spora.integrated_checkpoint import load_integrated, replacement_feature_variant_sha256
from spora.internal_memory import observe_sourced
from spora.named_address import address_basis_sha256, feature_files_sha256

OUTPUT = ROOT/'runs/deepseek-spora-independent-route-audit-20261111.json'
TEMPORAL_CHECKPOINT = ROOT/'runs/deepseek-spora-temporal-gate-train-20261111/temporal.pt'
INTENT_CHECKPOINT_DIR = INTENT

# None labels are intentional negative controls. These formulations were not
# used in the existing gate-training files.
CASES = (
    {'text': '小林眼下偏好的饮品是哪一种？', 'label': CURRENT, 'value': 6},
    {'text': '现在小林通常喝什么？', 'label': CURRENT, 'value': 6},
    {'text': '我想查一下小林目前的饮料偏好。', 'label': CURRENT, 'value': 6},
    {'text': '小林此时最爱喝的东西是什么？', 'label': CURRENT, 'value': 6},
    {'text': '请告诉我小林当下记录的饮品。', 'label': CURRENT, 'value': 6},
    {'text': '小林最新留下的饮料是哪种？', 'label': CURRENT, 'value': 6},
    {'text': '小林在改口之前偏爱什么饮料？', 'label': PAST, 'value': 1},
    {'text': '较早的时候，小林爱喝哪一类饮品？', 'label': PAST, 'value': 1},
    {'text': '小林最初记录的饮料偏好是什么？', 'label': PAST, 'value': 1},
    {'text': '请翻出小林变更前的饮品信息。', 'label': PAST, 'value': 1},
    {'text': '小林早先常喝的是哪种？', 'label': PAST, 'value': 1},
    {'text': '在更新以前，小林喜欢什么饮料？', 'label': PAST, 'value': 1},
    {'text': '小林改了偏好以后选择了什么饮料？', 'label': LATER, 'value': 6},
    {'text': '后来的记录里小林爱喝什么？', 'label': LATER, 'value': 6},
    {'text': '小林在那次更正之后偏爱哪种饮品？', 'label': LATER, 'value': 6},
    {'text': '更新完成后，小林的饮料选择是什么？', 'label': LATER, 'value': 6},
    {'text': '小林后来改成喜欢喝什么了？', 'label': LATER, 'value': 6},
    {'text': '再往后的记录中，小林最爱哪种饮料？', 'label': LATER, 'value': 6},
    {'text': '请给小林写一句鼓励的话。', 'label': None},
    {'text': '不要查询小林的饮料记录。', 'label': None},
    {'text': '把小林喜欢的饮料改成牛奶。', 'label': None},
    {'text': '小林和阿宁分别喜欢喝什么？', 'label': None},
    {'text': '小雪以前喜欢喝什么？', 'label': None},
    {'text': '你能解释一下社会主义核心价值观吗？', 'label': None},
    {'text': '小梅后来喝什么？', 'label': None},
    {'text': '小林的饮品记录是谁写下的？', 'label': None},
    {'text': '帮我设计一个饮料广告。', 'label': None},
    {'text': '小林现在喜欢什么以及阿宁过去喜欢什么？', 'label': None},
    {'text': '请把小林和小雨的偏好都查出来。', 'label': None},
)


def _check_provenance():
    router_report = json.loads((ROUTER/'result.json').read_text(encoding='utf-8'))
    entity_report = json.loads((ENTITY/'result.json').read_text(encoding='utf-8'))
    intent_report = json.loads((INTENT_CHECKPOINT_DIR/'result.json').read_text(encoding='utf-8'))
    for path, expected in ((ROUTER/'router.pt', router_report['router_sha256']),
                           (ENTITY/'gate.pt', entity_report['gate_sha256']),
                           (INTENT_CHECKPOINT_DIR/'intent.pt', intent_report['intent_sha256'])):
        if sha(path) != expected:
            raise ValueError('Gate digest mismatch')
    router = torch.load(ROUTER/'router.pt', map_location='cpu', weights_only=True)
    entity_data = torch.load(ENTITY/'gate.pt', map_location='cpu', weights_only=True)
    intent_data = torch.load(INTENT_CHECKPOINT_DIR/'intent.pt', map_location='cpu', weights_only=True)
    named, _, _ = load(NAMED/'individual.pt', 'cpu')
    config_hash = feature_files_sha256(MODEL)
    if (router['base_sha256'] != WEIGHT_SHA or router['feature_config_sha256'] != config_hash
            or router['basis_sha256'] != address_basis_sha256(named)
            or entity_data['router_sha256'] != sha(ROUTER/'router.pt')
            or intent_data['router_sha256'] != sha(ROUTER/'router.pt')
            or intent_data['entity_gate_sha256'] != sha(ENTITY/'gate.pt')
            or entity_data['feature_variant_sha256'] != router['feature_variant_sha256']
            or intent_data['feature_variant_sha256'] != router['feature_variant_sha256']):
        raise ValueError('Gate composition provenance mismatch')
    return router, entity_data, intent_data


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base model mismatch')
    router_data, entity_data, intent_data = _check_provenance()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    integrated, organism, _ = load_integrated(SOURCE_CHECKPOINT, base, MODEL, individual_device='cuda')
    bases = torch.stack([F.normalize(site.current_basis.detach().cpu().float(), dim=-1)
                         for site in organism.memories])
    state = fresh_controlled_state(organism)
    for value in (1, 6):
        state, decision = observe_sourced(organism, state, KEYS[0], value, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('Controlled memory write rejected')
    train = examples(TRAIN)
    temporal_hidden, temporal_mask = extract(base, tokenizer, [r['text'] for r in train], batch_size=2)
    temporal_width = pool(temporal_hidden, temporal_mask).shape[-1]
    temporal = nn.Linear(temporal_width, 4).eval()
    temporal_report = json.loads((TEMPORAL_CHECKPOINT.parent/'result.json').read_text(encoding='utf-8'))
    if sha(TEMPORAL_CHECKPOINT) != temporal_report['temporal_sha256']:
        raise ValueError('Temporal gate digest mismatch')
    temporal_data = torch.load(TEMPORAL_CHECKPOINT, map_location='cpu', weights_only=True)
    if (temporal_data.get('format') != 'spora-bounded-temporal-gate-v1'
            or temporal_data.get('base_sha256') != WEIGHT_SHA
            or temporal_data.get('feature_variant_sha256') != router_data['feature_variant_sha256']
            or temporal_data.get('classes') != ['current', 'past', 'later', 'abstain']):
        raise ValueError('Temporal gate provenance mismatch')
    temporal.load_state_dict(temporal_data['state'], strict=True)
    blocks = {i: _load_candidate(SOURCE/f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    if replacement_feature_variant_sha256(MODEL, blocks) != router_data['feature_variant_sha256']:
        raise ValueError('Structural variant mismatch')
    for i, block in blocks.items():
        base.model.layers[i] = block
    questions = [case['text'] for case in CASES]
    features, masks = extract(base, tokenizer, questions, batch_size=2)
    qvec = pool(features, masks).float()
    token_vec = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
    locator = nn.Linear(token_vec.shape[-1], 1).eval()
    locator.load_state_dict(router_data['locator_state'], strict=True)
    entity = nn.Linear(qvec.shape[-1], 1).eval()
    entity.load_state_dict(entity_data['gate_state'], strict=True)
    intent = nn.Linear(qvec.shape[-1], 1).eval()
    intent.load_state_dict(intent_data['state'], strict=True)
    results = []
    with torch.inference_mode():
        loc_scores = locator(token_vec).squeeze(-1).sigmoid()
        entity_scores = entity(qvec).squeeze(-1).sigmoid()
        intent_logits = intent(qvec).squeeze(-1)
        temporal_logits = temporal(qvec)
        probs = temporal_logits.softmax(-1)
        winners = probs.argmax(-1)
        predicted_times = [int(winners[i]) if int(winners[i]) != 3 and
                           float(probs[i, :3].max()) >= temporal_data['threshold'] else 3
                           for i in range(len(CASES))]
        for i, case in enumerate(CASES):
            selected, single = single_span_mask(loc_scores[i], masks[i], router_data['locator_threshold'])
            expected = case['label']
            row = {'question': case['text'], 'expected_intent': expected,
                   'predicted_intent': predicted_times[i], 'expected_value': case.get('value'),
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
                indices, sims = predict(v, router_data['projection'], bases)
                row['indices'], row['similarities'] = indices[0].tolist(), sims[0].tolist()
                if not bool(indices[0, 0] == indices[0, 1] and
                            (sims[0] >= entity_data['address_threshold']).all()):
                    row['reason'] = 'name_binding_rejected'
                else:
                    key = int(indices[0, 0])
                    if current_memory_status(organism, state, key)['status'] != 'known':
                        row['reason'] = 'unknown_memory'
                    else:
                        mode = 'current' if predicted_times[i] == CURRENT else 'past' if predicted_times[i] == PAST else 'later'
                        order = None if mode == 'current' else 'effective_previous' if mode == 'past' else 'effective_latest'
                        selection = None if order is None else select_event(organism, state, key, order, as_of=state.cursor-1)
                        row['memory_event_selection'] = selection
                        if selection is not None and selection['status'] != 'experimental_candidate':
                            row['reason'] = 'memory_event_abstain'
                        else:
                            event = None if selection is None else selection['event_time']
                            row['accepted'], row['reason'], row['predicted_key'] = True, 'routed', key
                            args = _memory_args(organism, state, {'key': key, 'intent': mode,
                                                                   'event_time': 0 if event is None else event})
                            prefix = closed_think_inputs(tokenizer, case['text'], 'cuda')
                            ids = integrated.generate(**prefix, **args, max_new_tokens=8, do_sample=False,
                                                      pad_token_id=tokenizer.pad_token_id,
                                                      eos_token_id=tokenizer.eos_token_id)
                            row['answer'] = tokenizer.decode(ids[0, prefix['input_ids'].shape[1]:],
                                                             skip_special_tokens=True).strip()
                            row['answer_correct'] = expected is not None and row['answer'] in (
                                WORDS[case['value']], WORDS[case['value']]+'.', WORDS[case['value']]+'。')
            row['route_correct'] = (expected is None and not row['accepted']) or (
                expected is not None and row['accepted'] and row.get('predicted_key') == KEYS[0]
                and row['predicted_intent'] == expected)
            row.setdefault('answer_correct', False)
            results.append(row)
            print(json.dumps({'index': i, 'accepted': row['accepted'], 'route_correct': row['route_correct']}, ensure_ascii=False), flush=True)
    report = {'status': 'independent_authored_chinese_route_audit', 'base_sha256': WEIGHT_SHA,
              'intent_sha256': sha(INTENT_CHECKPOINT_DIR/'intent.pt'),
              'temporal_sha256': sha(TEMPORAL_CHECKPOINT),
              'individual_sha256': INDIVIDUAL_SHA, 'case_count': len(CASES),
              'positive_total': sum(c['label'] is not None for c in CASES),
              'negative_total': sum(c['label'] is None for c in CASES),
              'positive_routes_correct': sum(r['route_correct'] for r, c in zip(results, CASES) if c['label'] is not None),
              'positive_answers_correct': sum(r['answer_correct'] for r, c in zip(results, CASES) if c['label'] is not None),
              'negative_rejected': sum(not r['accepted'] for r, c in zip(results, CASES) if c['label'] is None),
              'rows': results, 'elapsed_seconds': time.perf_counter()-started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Fresh authored formulations, not a statistically independent corpus.',
                              'One bound entity and two controlled facts; negative names are not written.',
                              'Gates and temporal classifier were trained before this audit but reuse the same base.',
                              'Closed answer generation with an eight-token cap.',
                              'Does not establish open-world language, autonomous writes or general date semantics.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('positive_total', 'positive_routes_correct',
                                             'positive_answers_correct', 'negative_total',
                                             'negative_rejected')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
