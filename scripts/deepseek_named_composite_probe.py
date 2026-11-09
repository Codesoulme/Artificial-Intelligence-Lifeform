"""Read-only development test of learned name location and SPORA memory reads."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_adaptive_read_probe import GATE
from scripts.deepseek_chat_interference import SELECTOR
from scripts.deepseek_cluener_locator_train import OUTPUT as NEW_LOCATOR
from scripts.deepseek_learned_name_span import OUTPUT as OLD_LOCATOR, single_span_mask
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_memory_intent import IntentGate
from scripts.deepseek_name_span_probe import OUTPUT as PROJECTION
from scripts.deepseek_named_binding_probe import KEYS, NAMES, UNKNOWN, VALUES, pooled, predict
from scripts.deepseek_named_incremental_probe import NEW_KEYS, NEW_VALUES, OUTPUT as STATE_SOURCE
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_read_strength_probe import ScaledReadout
from scripts.deepseek_semantic_gate import hide_numbers
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.open_base_trajectory_comparison import numeric_correct
from spora.current_memory_status import current_memory_status
from spora.growing import query
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256
from spora.open_base_memory import memory_hooks
from spora.state_codec import restore

OUTPUT = ROOT / 'runs/deepseek-named-composite-20261110.json'
TEMPLATES = ('{name}现存的数字是多少？', '我想查{name}当前保存的数字。')
NEGATIVE = (
    ('negated', '请不要告诉我小林当前的数字。'), ('negated', '阿宁现在的值暂时保密。'),
    ('history', '小雨之前的数字是什么？'), ('history', '阿哲最早记住的数是多少？'),
    ('write', '把小梅当前的数字改掉。'), ('write', '请删除阿远的记录。'),
    ('source', '小岚当前的数字是谁提供的？'), ('source', '阿明的数值从哪里来？'),
    ('derived', '阿松的当前数字是否为偶数？'), ('derived', '小雪现在的值加倍是多少？'),
    ('multiple', '小林和阿宁各自的数字是多少？'), ('multiple', '比较小雨和阿哲现在的数字。'),
    ('none', '请告诉我当前的数字。'), ('none', '我想查现在保存的数值。'),
    ('chat', '你好，请给我一个放松建议。'), ('chat', '最近有什么值得学习的内容？'),
)


def replace_span(question, text, offsets, mask):
    start = text.find(question)
    if start < 0 or text.find(question, start + 1) >= 0:
        raise ValueError('Question must occur once')
    positions = torch.nonzero(mask).flatten()
    if not len(positions):
        raise ValueError('Selected span is empty')
    left = int(offsets[positions, 0].min()) - start
    right = int(offsets[positions, 1].max()) - start
    if not 0 <= left < right <= len(question):
        raise ValueError('Selected span outside question')
    return hide_numbers(question[:left] + '对象0' + question[right:]), [left, right]


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = {'state': STATE_SOURCE / 'research_individual.pt', 'state_report': STATE_SOURCE / 'result.json',
             'projection': PROJECTION / 'projections.pt', 'projection_report': PROJECTION / 'result.json',
             'old_locator': OLD_LOCATOR / 'locator.pt', 'old_report': OLD_LOCATOR / 'result.json',
             'new_locator': NEW_LOCATOR / 'locator.pt', 'new_report': NEW_LOCATOR / 'result.json',
             'gate': GATE, 'selector': SELECTOR, 'weight': MODEL / 'model.safetensors', 'individual': INDIVIDUAL}
    protected = {name: sha(path) for name, path in files.items()}
    if protected['weight'] != WEIGHT_SHA or protected['individual'] != INDIVIDUAL_SHA:
        raise ValueError('Protected model mismatch')
    reports = {}
    for key in ('state_report', 'projection_report', 'old_report', 'new_report'):
        with files[key].open(encoding='utf-8') as stream:
            reports[key] = json.load(stream)
    if (reports['state_report']['checkpoint_sha256'] != protected['state']
            or reports['projection_report']['projection_sha256'] != protected['projection']
            or reports['old_report']['locator_sha256'] != protected['old_locator']
            or reports['new_report']['locator_sha256'] != protected['new_locator']):
        raise ValueError('Research checkpoint provenance mismatch')
    config_sha = feature_files_sha256(MODEL)
    projection = torch.load(files['projection'], map_location='cpu', weights_only=True)
    gate_data = torch.load(files['gate'], map_location='cpu', weights_only=True)
    selector_data = torch.load(files['selector'], map_location='cpu', weights_only=True)
    locator_data = {name: torch.load(files[name + '_locator'], map_location='cpu', weights_only=True)
                    for name in ('old', 'new')}
    if (projection.get('format') != 'oracle-name-span-development-v1'
            or projection.get('weight_sha256') != WEIGHT_SHA or projection.get('feature_config_sha256') != config_sha
            or gate_data.get('format') != 'deepseek-spora-number-invariant-intent-v1'
            or selector_data.get('format') != 'deepseek-spora-question-pooled-finite-address-v1'
            or gate_data.get('threshold') != 0.5 or gate_data.get('layers') != [5, 17]
            or selector_data.get('layers') != [5, 17]
            or any(d.get('weight_sha256') != WEIGHT_SHA or d.get('individual_sha256') != INDIVIDUAL_SHA
                   for d in (gate_data, selector_data))
            or any(d.get('format') != 'deepseek-name-span-locator-development-v1'
                   or d.get('weight_sha256') != WEIGHT_SHA or d.get('feature_config_sha256') != config_sha
                   or d.get('layers') != [6, 18] or d.get('threshold') != 0.5 for d in locator_data.values())):
        raise ValueError('Component metadata mismatch')
    payload = torch.load(files['state'], map_location='cpu', weights_only=True)
    if payload.get('format') != 'named-online-ridge-research-v1':
        raise ValueError('Unexpected research state format')
    organism, state = restore(payload['schema'], payload['tensors'])
    before = frozen_digest(organism, state)
    all_keys, all_values = KEYS + NEW_KEYS, VALUES + NEW_VALUES
    key_tensor = torch.tensor(all_keys, dtype=torch.long)
    with torch.inference_mode():
        actual = query(organism, state, key_tensor, torch.zeros_like(key_tensor), torch.zeros_like(key_tensor)).argmax(-1)
    if actual.tolist() != list(all_values):
        raise ValueError('Controlled memory values do not match source')
    cases = [{'group': 'known', 'question': template.format(name=name), 'key': key, 'value': value}
             for name, key, value in zip(NAMES + UNKNOWN[:2], all_keys, all_values) for template in TEMPLATES]
    cases += [{'group': 'unbound', 'question': template.format(name=name), 'key': None, 'value': None}
              for name in UNKNOWN[2:] for template in TEMPLATES]
    cases += [{'group': group, 'question': question, 'key': None, 'value': None} for group, question in NEGATIVE]
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    questions = [r['question'] for r in cases]
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False,
                                           add_generation_prompt=True) for q in questions]
    encoded = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False, return_offsets_mapping=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    features, masks = extract(base, tokenizer, questions, batch_size=1)
    if features.shape[2] != encoded['offset_mapping'].shape[1]:
        raise ValueError('Tokenizer alignment mismatch')
    token_vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
    bases = torch.stack([F.normalize(site.current_basis.detach().float(), dim=-1) for site in organism.memories])
    threshold = reports['projection_report']['results']['oracle_name_span']['threshold']
    decisions, gate_texts = {}, set()
    with torch.inference_mode():
        for name, data in locator_data.items():
            locator = nn.Linear(features.shape[-1] * 2, 1).eval()
            locator.load_state_dict(data['state'], strict=True)
            token_scores = locator(token_vectors).squeeze(-1).sigmoid()
            rows = []
            for i, case in enumerate(cases):
                mask, single = single_span_mask(token_scores[i], masks[i])
                row = {'single_span': single, 'gate_input': None, 'predicted': None, 'similarities': None}
                if single:
                    vectors = pooled(features[i:i+1], mask.unsqueeze(0))
                    predicted, scores = predict(vectors, projection['weights']['oracle_name_span'], bases)
                    row['predicted'], row['similarities'] = predicted[0].tolist(), scores[0].tolist()
                    row['gate_input'], row['character_span'] = replace_span(case['question'], texts[i], encoded['offset_mapping'][i], mask)
                    gate_texts.add(row['gate_input'])
                rows.append(row)
            decisions[name] = rows
    gate = IntentGate(base.config.hidden_size).to('cuda').eval()
    gate.load_state_dict(gate_data['state'], strict=True)
    ordered = sorted(gate_texts)
    gate_lookup = {}
    if ordered:
        gate_features, gate_masks = extract(base, tokenizer, ordered, batch_size=1)
        with torch.inference_mode():
            for index, text in enumerate(ordered):
                gate_lookup[text] = float(gate(gate_features[index:index+1].to('cuda'),
                                               gate_masks[index:index+1].to('cuda')).sigmoid()[0])
    adapter = ScaledReadout(base.config.hidden_size, organism.config.memory_width, 0.5).to('cuda').eval()
    adapter.load_state_dict(selector_data['readout'], strict=True)
    answer_cache, results = {}, {}
    with torch.inference_mode():
        for name, decision_rows in decisions.items():
            rows = []
            for case, decision in zip(cases, decision_rows):
                route, score = 'defer', None
                if decision['single_span']:
                    score = gate_lookup[decision['gate_input']]
                    predicted, sims = decision['predicted'], decision['similarities']
                    if score >= 0.5 and min(sims) >= threshold and predicted[0] == predicted[1]:
                        status = current_memory_status(organism, state, predicted[0])
                        route = 'unknown' if status['status'] == 'unknown' else 'answer'
                row = dict(case, **decision, intent_score=score, route=route)
                row['route_correct'] = (route == 'answer' and decision['predicted'] == [case['key']] * 2
                                        if case['group'] == 'known' else route == 'defer')
                if route == 'answer' and case['group'] == 'known':
                    cache_key = (case['question'], decision['predicted'][0])
                    if cache_key not in answer_cache:
                        key = decision['predicted'][0]
                        addresses = [bases[site, key:key+1].to('cuda') for site in range(2)]
                        inputs = closed_think_inputs(tokenizer, case['question'], 'cuda')
                        with memory_hooks([base.model.layers[5], base.model.layers[17]], adapter, state,
                                          current_only=True, addresses=addresses):
                            generated = base.generate(**inputs, max_new_tokens=8, do_sample=False,
                                                      pad_token_id=tokenizer.eos_token_id)
                        tokens = generated[0, inputs['input_ids'].shape[1]:].tolist()
                        answer_cache[cache_key] = {'text': tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                                                   'ended_with_eos': tokenizer.eos_token_id in tokens}
                    row['answer'] = answer_cache[cache_key]
                    row['answer_correct'] = row['route_correct'] and numeric_correct(row['answer']['text'], case['value'])
                rows.append(row)
            summary = {}
            for group in ('known', 'unbound', 'negative'):
                group_rows = [r for r in rows if (r['group'] == group if group != 'negative' else
                                                   r['group'] not in ('known', 'unbound'))]
                summary[group] = {'examples': len(group_rows), 'route_correct': sum(r['route_correct'] for r in group_rows),
                                  'answer_correct': sum(r.get('answer_correct', False) for r in group_rows),
                                  'misrouted': sum(r['route'] != 'defer' for r in group_rows) if group != 'known' else None}
            results[name] = {'summary': summary, 'rows': rows}
            print(f'{name} locator checked', flush=True)
    if (frozen_digest(organism, state) != before or feature_files_sha256(MODEL) != config_sha
            or any(sha(path) != protected[name] for name, path in files.items())):
        raise ValueError('Protected input changed')
    report = {'status': 'named_composite_development_only', 'weight_sha256': WEIGHT_SHA,
              'state_sha256': protected['state'], 'projection_sha256': protected['projection'],
              'locator_sha256': {name: protected[name + '_locator'] for name in ('old', 'new')},
              'gate_sha256': protected['gate'], 'results': results, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['No true online training, persistence test, history or natural-language fact values.',
                              'Location and intent use learned components; gate input replaces the predicted span with a symbolic numbered placeholder.',
                              'Address projection was supervised on all ten known names and evaluated on development templates.',
                              'Requests and entity names are hand-authored; some names and patterns were seen during development.',
                              'Unbound and non-current requests are not generated even if misrouted.',
                              'Known answers use closed thinking, greedy generation, 8-token limit and fixed 0.5 injection.',
                              'No proof of safe routing, independent individual or general chat ability.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: v['summary'] for k, v in results.items()}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
