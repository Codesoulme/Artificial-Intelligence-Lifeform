"""Isolated research test of Chinese observation, binding, write and readback."""
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
from scripts.deepseek_cluener_locator_train import OUTPUT as LOCATOR
from scripts.deepseek_learned_name_span import single_span_mask
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_name_span_probe import OUTPUT as PROJECTION
from scripts.deepseek_named_binding_probe import KEYS, NAMES, UNKNOWN, pooled, predict
from scripts.deepseek_named_incremental_probe import NEW_KEYS
from scripts.deepseek_observation_factorized_probe import FORMAT, FactorizedProbe, combined_logits
from scripts.deepseek_observation_value_probe import ABSTAIN, WORDS, pool
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_read_strength_probe import ScaledReadout
from scripts.deepseek_semantic_value_pilot import word_correct
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.current_memory_status import current_memory_status
from spora.growing import load, query
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256
from spora.open_base_memory import memory_hooks

OUTPUT = ROOT / 'runs/deepseek-observation-memory-comparison-20261110.json'
VALUE = ROOT / 'runs/deepseek-observation-factorized-20261110'
READOUT = ROOT / 'runs/deepseek-semantic-value-balanced-20261110'


def build_cases():
    cases = []
    for i, (name, key) in enumerate(zip((NAMES[0], NAMES[1], UNKNOWN[0], UNKNOWN[1]),
                                        (KEYS[0], KEYS[1], NEW_KEYS[0], NEW_KEYS[1]))):
        first, second = ((1, 6) if i % 2 == 0 else (6, 1))
        cases.append({'group': 'correction', 'name': name, 'expected_key': key, 'expected_value': second,
                      'events': [f'{name}喜欢喝{WORDS[first]}。', f'{name}最喜欢的饮料是{WORDS[second]}。'],
                      'event_expected': [first, second], 'question': f'{name}喜欢的饮品是什么？'})
    cases += [
        {'group': 'negative', 'name': NAMES[0], 'expected_key': KEYS[0], 'expected_value': None,
         'events': [f'{NAMES[0]}不喜欢喝咖啡。'], 'event_expected': [None],
         'question': f'{NAMES[0]}喜欢的饮品是什么？'},
        {'group': 'negative', 'name': NAMES[1], 'expected_key': KEYS[1], 'expected_value': None,
         'events': [f'我只是猜测{NAMES[1]}喜欢喝水。'], 'event_expected': [None],
         'question': f'{NAMES[1]}喜欢的饮品是什么？'},
        {'group': 'unbound', 'name': UNKNOWN[2], 'expected_key': None, 'expected_value': None,
         'events': [f'{UNKNOWN[2]}喜欢喝咖啡。'], 'event_expected': [None],
         'question': f'{UNKNOWN[2]}喜欢的饮品是什么？'},
        {'group': 'multiple', 'name': NAMES[0], 'expected_key': KEYS[0], 'expected_value': None,
         'events': [f'{NAMES[1]}喜欢喝咖啡，但不知道{NAMES[0]}喜欢什么。'], 'event_expected': [None],
         'question': f'{NAMES[0]}喜欢的饮品是什么？'},
    ]
    return cases


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = {'weight': MODEL / 'model.safetensors', 'individual': INDIVIDUAL,
             'projection': PROJECTION / 'projections.pt', 'projection_report': PROJECTION / 'result.json',
             'locator': LOCATOR / 'locator.pt', 'locator_report': LOCATOR / 'result.json',
             'value': VALUE / 'classifier.pt', 'value_report': VALUE / 'result.json',
             'readout': READOUT / 'adapter.pt', 'readout_report': READOUT / 'result.json'}
    before = {name: sha(path) for name, path in files.items()}
    if before['weight'] != WEIGHT_SHA or before['individual'] != INDIVIDUAL_SHA:
        raise ValueError('Protected input mismatch')
    reports = {}
    for name in ('projection', 'locator', 'value', 'readout'):
        with files[name + '_report'].open(encoding='utf-8') as stream:
            reports[name] = json.load(stream)
    for name, field in (('projection', 'projection_sha256'), ('locator', 'locator_sha256'),
                        ('value', 'classifier_sha256'), ('readout', 'adapter_sha256')):
        if reports[name][field] != before[name]:
            raise ValueError('Component checkpoint mismatch')
    config_sha = feature_files_sha256(MODEL)
    data = {name: torch.load(files[name], map_location='cpu', weights_only=True)
            for name in ('projection', 'locator', 'value', 'readout')}
    if (data['projection'].get('format') != 'oracle-name-span-development-v1'
            or data['locator'].get('format') != 'deepseek-name-span-locator-development-v1'
            or data['value'].get('format') != FORMAT
            or data['readout'].get('format') != 'deepseek-spora-semantic-value-development-v1'
            or any(d.get('weight_sha256') != WEIGHT_SHA or d.get('individual_sha256') != INDIVIDUAL_SHA
                   for d in (data['projection'], data['value'], data['readout']))
            or any(d.get('feature_config_sha256') != config_sha for d in (data['projection'], data['locator'], data['value']))
            or data['locator'].get('weight_sha256') != WEIGHT_SHA or data['locator'].get('layers') != [6, 18]
            or data['locator'].get('threshold') != 0.5 or data['value'].get('hidden_indices') != [6, 18]
            or tuple(data['value'].get('vocabulary', ())) != WORDS or data['value'].get('abstain') != ABSTAIN
            or data['readout'].get('layers') != [5, 17] or data['readout'].get('strength') != 0.5):
        raise ValueError('Component provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    original_digest = frozen_digest(organism, original)
    bases = torch.stack([F.normalize(site.current_basis.detach().float(), dim=-1) for site in organism.memories])
    cases = build_cases()
    texts = list(dict.fromkeys(text for case in cases for text in case['events'] + [case['question']]))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    formatted = [tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False,
                                               add_generation_prompt=True) for text in texts]
    encoding = tokenizer(formatted, return_tensors='pt', padding=True, add_special_tokens=False,
                         return_offsets_mapping=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    features, masks = extract(base, tokenizer, texts, batch_size=1)
    if features.shape[2] != encoding['offset_mapping'].shape[1]:
        raise ValueError('Feature alignment mismatch')
    locator = nn.Linear(features.shape[-1] * 2, 1).eval()
    locator.load_state_dict(data['locator']['state'], strict=True)
    value = FactorizedProbe(features.shape[-1] * 2).eval()
    value.load_state_dict(data['value']['state'], strict=True)
    adapter = ScaledReadout(base.config.hidden_size, organism.config.memory_width, 0.5).to('cuda').eval()
    adapter.load_state_dict(data['readout']['state'], strict=True)
    threshold = reports['projection']['results']['oracle_name_span']['threshold']
    tokens = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
    with torch.no_grad():
        scores = locator(tokens).squeeze(-1).sigmoid()
        fact, value_logits = value(pool(features, masks))
        classes = combined_logits(fact, value_logits).argmax(-1)
    decisions = {}
    for i, text in enumerate(texts):
        mask, single = single_span_mask(scores[i], masks[i])
        row = {'single_span': single, 'key': None, 'similarities': None,
               'value': int(classes[i]), 'fact_probability': float(fact[i].sigmoid()),
               'value_prediction': int(value_logits[i].argmax())}
        if single:
            vectors = pooled(features[i:i + 1], mask.unsqueeze(0))
            indices, similarities = predict(vectors, data['projection']['weights']['oracle_name_span'], bases)
            ids, sims = indices[0].tolist(), similarities[0].tolist()
            row['indices'], row['similarities'] = ids, sims
            if ids[0] == ids[1] and min(sims) >= threshold:
                row['key'] = ids[0]
        decisions[text] = row
    results, baseline_answers = [], {}
    for case in cases:
        state = fresh_controlled_state(organism)
        events = []
        for text, expected in zip(case['events'], case['event_expected']):
            inferred = decisions[text]
            action = None
            if inferred['key'] is not None and inferred['value'] != ABSTAIN:
                state, action = observe_sourced(organism, state, inferred['key'], inferred['value'], 'direct', 1.0)
            events.append({'text': text, 'expected_value': expected, 'inferred': inferred, 'write': action,
                           'correct_write': expected is not None and inferred['key'] == case['expected_key']
                           and inferred['value'] == expected and action is not None and action['current_write'],
                           'unexpected_write': expected is None and action is not None and action['current_write']})
        read = decisions[case['question']]
        if case['question'] not in baseline_answers:
            inputs = closed_think_inputs(tokenizer, case['question'], 'cuda')
            with torch.no_grad():
                generated = base.generate(**inputs, max_new_tokens=8, do_sample=False,
                                          pad_token_id=tokenizer.eos_token_id)
            baseline_answers[case['question']] = tokenizer.decode(
                generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        output, status = None, None
        if read['key'] is not None:
            status = current_memory_status(organism, state, read['key'])['status']
            if status != 'unknown':
                inputs = closed_think_inputs(tokenizer, case['question'], 'cuda')
                addresses = [bases[site, read['key']:read['key'] + 1].to('cuda') for site in range(2)]
                with torch.no_grad(), memory_hooks([base.model.layers[5], base.model.layers[17]], adapter, state,
                                                   current_only=True, addresses=addresses):
                    generated = base.generate(**inputs, max_new_tokens=8, do_sample=False,
                                              pad_token_id=tokenizer.eos_token_id)
                output = tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        native = None
        if case['expected_key'] is not None:
            key = torch.tensor([case['expected_key']])
            with torch.no_grad():
                native = int(query(organism, state, key, torch.zeros_like(key), torch.zeros_like(key)).argmax(-1)[0])
        results.append({**case, 'events_result': events, 'read_decision': read, 'read_status': status,
                        'native_value': native, 'answer': output, 'baseline_answer': baseline_answers[case['question']],
                        'answer_correct': case['expected_value'] is not None and read['key'] == case['expected_key']
                        and output is not None and word_correct(output, WORDS[case['expected_value']]),
                        'baseline_correct': case['expected_value'] is not None
                        and word_correct(baseline_answers[case['question']], WORDS[case['expected_value']])})
    if (any(sha(path) != before[name] for name, path in files.items())
            or frozen_digest(organism, original) != original_digest or feature_files_sha256(MODEL) != config_sha):
        raise ValueError('Protected input changed')
    summary = {'correction_cases': sum(r['group'] == 'correction' for r in results),
               'correct_event_writes': sum(e['correct_write'] for r in results for e in r['events_result']),
               'expected_event_writes': sum(e is not None for r in results for e in r['event_expected']),
               'unexpected_writes': sum(e['unexpected_write'] for r in results for e in r['events_result']),
               'correct_answers': sum(r['answer_correct'] for r in results),
               'baseline_correct': sum(r['baseline_correct'] for r in results)}
    report = {'status': 'isolated_composite_research_only', 'summary': summary, 'rows': results,
              'component_sha256': before, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['All states are fresh temporary research copies; no persistent individual is changed.',
                              'Correct values and addresses are used only for scoring; predictions drive writes and reads.',
                              'The experiment invokes the read path for selected questions; no general intent router is tested.',
                              'The simulated source is always direct; natural language does not determine source trust.',
                              'Four synthetic correction sequences and four negatives are insufficient for safety or generalization.',
                              'Known names and many templates were used in component development; vocabulary is fixed.',
                              'A named projection and frozen encoder remain separate artifacts, not a unified model.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    run()
