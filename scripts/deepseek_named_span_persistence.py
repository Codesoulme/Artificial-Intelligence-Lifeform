"""Persist span-based name binding and isolated observations together."""
import argparse
import json
import subprocess
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

from scripts.deepseek_cluener_locator_train import OUTPUT as LOCATOR
from scripts.deepseek_learned_name_span import single_span_mask
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_name_span_probe import OUTPUT as PROJECTION
from scripts.deepseek_named_binding_probe import KEYS, NAMES, pooled
from scripts.deepseek_observation_factorized_probe import FORMAT, FactorizedProbe, combined_logits
from scripts.deepseek_observation_value_probe import ABSTAIN, WORDS, pool
from scripts.deepseek_pooled_address import extract
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import load, query, save
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest
from spora.named_address import NamedAddressBinding, address_basis_sha256, feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-named-span-persistence-20261110'
VALUE = ROOT / 'runs/deepseek-observation-factorized-20261110'
CONTRACT = 'name-span-mean-v1'


def verify(individual, audit):
    model, state, _ = load(individual, 'cpu')
    data = torch.load(audit, map_location='cpu', weights_only=True)
    feature_sha = feature_files_sha256(MODEL)
    with torch.inference_mode():
        binding = model.named_address_binding(data['features'], model, WEIGHT_SHA, feature_sha,
                                              feature_contract=CONTRACT)
        keys = torch.arange(model.config.key_count, dtype=torch.long)
        native = query(model, state, keys, torch.zeros_like(keys), torch.zeros_like(keys)).argmax(-1)
    if (not torch.equal(binding['indices'], data['indices'])
            or not torch.equal(binding['accepted'], data['accepted'])
            or not torch.allclose(binding['scores'], data['scores'], atol=1e-6, rtol=0)
            or not torch.equal(native, data['native'])):
        raise ValueError('Reloaded binding or memory differs')
    return {'binding_predictions': len(data['features']), 'accepted': int(binding['accepted'].sum()),
            'native_values': len(keys), 'cursor': state.cursor, 'state_digest': frozen_digest(model, state)}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = {'weight': MODEL / 'model.safetensors', 'individual': INDIVIDUAL,
             'projection': PROJECTION / 'projections.pt', 'projection_report': PROJECTION / 'result.json',
             'locator': LOCATOR / 'locator.pt', 'locator_report': LOCATOR / 'result.json',
             'value': VALUE / 'classifier.pt', 'value_report': VALUE / 'result.json'}
    before = {key: sha(path) for key, path in files.items()}
    if before['weight'] != WEIGHT_SHA or before['individual'] != INDIVIDUAL_SHA:
        raise ValueError('Protected input mismatch')
    reports = {}
    for name in ('projection', 'locator', 'value'):
        with files[name + '_report'].open(encoding='utf-8') as stream:
            reports[name] = json.load(stream)
    for name, field in (('projection', 'projection_sha256'), ('locator', 'locator_sha256'),
                        ('value', 'classifier_sha256')):
        if reports[name][field] != before[name]:
            raise ValueError('Component checkpoint mismatch')
    config_sha = feature_files_sha256(MODEL)
    data = {name: torch.load(files[name], map_location='cpu', weights_only=True)
            for name in ('projection', 'locator', 'value')}
    if (data['projection'].get('format') != 'oracle-name-span-development-v1'
            or data['locator'].get('format') != 'deepseek-name-span-locator-development-v1'
            or data['value'].get('format') != FORMAT
            or any(d.get('weight_sha256') != WEIGHT_SHA or d.get('feature_config_sha256') != config_sha
                   for d in data.values())
            or data['projection'].get('individual_sha256') != INDIVIDUAL_SHA
            or data['value'].get('individual_sha256') != INDIVIDUAL_SHA
            or data['locator'].get('layers') != [6, 18] or data['locator'].get('threshold') != 0.5
            or data['value'].get('hidden_indices') != [6, 18]
            or tuple(data['value'].get('vocabulary', ())) != WORDS or data['value'].get('abstain') != ABSTAIN):
        raise ValueError('Component provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    original_digest = frozen_digest(organism, original)
    state = fresh_controlled_state(organism)
    weights = data['projection']['weights']['oracle_name_span']
    threshold = reports['projection']['results']['oracle_name_span']['threshold']
    binding = NamedAddressBinding(weights.shape[1], weights.shape[2], WEIGHT_SHA, threshold,
                                  feature_contract=CONTRACT, basis_sha256=address_basis_sha256(organism),
                                  feature_config_sha256=config_sha)
    with torch.no_grad():
        binding.weights.copy_(weights)
    organism.named_address_binding = binding
    texts = [f'{NAMES[0]}喜欢喝咖啡。', f'{NAMES[1]}喜欢喝水。',
             f'{NAMES[0]}最喜欢的饮料是水。', f'{NAMES[1]}最喜欢的饮料是咖啡。',
             f'{NAMES[0]}喜欢的饮品是什么？', f'{NAMES[1]}喜欢的饮品是什么？']
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    features, masks = extract(base, tokenizer, texts, batch_size=1)
    vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
    locator = nn.Linear(vectors.shape[-1], 1).eval()
    locator.load_state_dict(data['locator']['state'], strict=True)
    value = FactorizedProbe(vectors.shape[-1]).eval()
    value.load_state_dict(data['value']['state'], strict=True)
    with torch.no_grad():
        token_scores = locator(vectors).squeeze(-1).sigmoid()
        fact, value_logits = value(pool(features, masks))
        classes = combined_logits(fact, value_logits).argmax(-1)
    span_vectors, selected = [], []
    for i in range(len(texts)):
        mask, single = single_span_mask(token_scores[i], masks[i])
        selected.append(single)
        span_vectors.append(pooled(features[i:i + 1], mask.unsqueeze(0))[0])
    spans = torch.stack(span_vectors)
    with torch.inference_mode():
        mapping = binding(spans, organism, WEIGHT_SHA, config_sha, feature_contract=CONTRACT)
    events = []
    for i, expected in enumerate((1, 6, 6, 1)):
        key = int(mapping['indices'][i, 0]) if bool(mapping['accepted'][i]) and selected[i] else None
        predicted = int(classes[i])
        decision = None
        if key is not None and predicted != ABSTAIN:
            state, decision = observe_sourced(organism, state, key, predicted, 'direct', 1.0)
        expected_key = (KEYS[0], KEYS[1], KEYS[0], KEYS[1])[i]
        events.append({'text': texts[i], 'predicted_key': key, 'predicted_value': predicted,
                       'decision': decision, 'expected_key': expected_key, 'expected_value': expected,
                       'correct_write': key == expected_key and predicted == expected and decision is not None
                       and decision['current_write']})
    keys = torch.arange(organism.config.key_count, dtype=torch.long)
    with torch.inference_mode():
        native = query(organism, state, keys, torch.zeros_like(keys), torch.zeros_like(keys)).argmax(-1)
    expected_digest = frozen_digest(organism, state)
    OUTPUT.mkdir(parents=False, exist_ok=False)
    individual, audit = OUTPUT / 'individual.pt', OUTPUT / 'features.pt'
    save(individual, organism, state, {'purpose': 'isolated_span_binding_and_observation_research',
                                       'source_individual_sha256': INDIVIDUAL_SHA,
                                       'projection_sha256': before['projection']})
    torch.save({'features': spans, 'indices': mapping['indices'], 'scores': mapping['scores'],
                'accepted': mapping['accepted'], 'native': native}, audit)
    same = verify(individual, audit)
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--verify', str(individual), str(audit)],
                           check=True, capture_output=True, text=True, encoding='utf-8', timeout=120)
    separate = json.loads(child.stdout.strip())
    if same != separate or same['state_digest'] != expected_digest:
        raise ValueError('Reload verification differs')
    del organism.named_address_binding
    if (frozen_digest(organism, original) != original_digest or feature_files_sha256(MODEL) != config_sha
            or any(sha(path) != before[name] for name, path in files.items())):
        raise ValueError('Protected input changed')
    report = {'status': 'isolated_span_binding_persistence_research_only', 'events': events,
              'query_indices': mapping['indices'][4:].tolist(), 'query_accepted': mapping['accepted'][4:].tolist(),
              'native_target_values': [int(native[KEYS[0]]), int(native[KEYS[1]])],
              'saved_individual_sha256': sha(individual), 'audit_features_sha256': sha(audit),
              'same_process': same, 'separate_process': separate, 'component_sha256': before,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Fresh temporary state, not a replacement for the original individual or its memories.',
                              'Four hand-authored events and two names were used in previous component development.',
                              'Source is explicitly simulated as direct; labels are used for scoring only.',
                              'The saved individual contains span projection and numeric state; encoder, locator, value head and readout remain separate.',
                              'Saved features are used for the independent-process verification; fresh encoding and generation were not tested there.',
                              'Ordinary reload is not a crash-recovery test or proof of safe automatic writing.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'events_correct': sum(e['correct_write'] for e in events), 'events': len(events),
                      'native_target_values': report['native_target_values'], 'reload': same}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', nargs=2, metavar=('INDIVIDUAL', 'AUDIT'))
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(*map(Path, args.verify))), flush=True)
    else:
        run()
