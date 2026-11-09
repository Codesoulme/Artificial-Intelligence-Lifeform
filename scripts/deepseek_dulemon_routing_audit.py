"""Audit false memory reads on a bounded external Chinese dialogue sample."""
import argparse
import hashlib
import json
import math
import sys
import time
import zipfile
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
from scripts.deepseek_integrated_checkpoint_probe import OUTPUT as CHECKPOINT
from scripts.deepseek_learned_name_span import single_span_mask
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_named_binding_probe import pooled
from scripts.deepseek_named_span_persistence import CONTRACT
from scripts.deepseek_observation_factorized_probe import FORMAT as VALUE_FORMAT, FactorizedProbe, combined_logits
from scripts.deepseek_observation_value_probe import ABSTAIN, WORDS, pool
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_preference_intent_probe import FORMAT
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.current_memory_status import current_memory_status
from spora.integrated_checkpoint import load_integrated
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

ARCHIVE = ROOT / 'runs/dulemon-source-20261110/DuLeMon.zip'
ARCHIVE_SHA = '797680b350dcb401cbe6255362c84e32c37363c983ffccaa0c0000bc75e3bde3'
GATE = ROOT / 'runs/deepseek-preference-intent-expanded-20261110'
VALUE = ROOT / 'runs/deepseek-observation-factorized-20261110'
OUTPUT = ROOT / 'runs/deepseek-dulemon-routing-audit-20261110.json'
EXCLUDE = ('喝', '饮', '酒', '茶', '咖啡', '可乐', '水', '豆浆', '酸奶', '果汁', '牛奶')
COUNT = 80


def sample_archive(path, topic=False):
    candidates = []
    with zipfile.ZipFile(path) as archive:
        with archive.open('DuLeMon/both/test.json') as stream:
            for session, line in enumerate(stream):
                if not line.strip():
                    continue
                data = json.loads(line)
                if not isinstance(data.get('conversation'), list):
                    raise ValueError('Invalid dialogue record')
                for turn, item in enumerate(data['conversation']):
                    if not isinstance(item, str) or not item.startswith('Usr:'):
                        continue
                    text = ''.join(item.split('\t', 1)[0][4:].split())
                    if not 4 <= len(text) <= 80:
                        continue
                    if topic and not (any(term in text for term in ('喝', '饮料', '饮品'))
                                      and any(term in text for term in ('喜欢', '爱'))):
                        continue
                    if not topic and any(term in text for term in EXCLUDE):
                        continue
                    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
                    candidates.append((digest, session, turn, text))
    selected, sessions, texts = [], set(), set()
    for digest, session, turn, text in sorted(candidates):
        if (not topic and session in sessions) or text in texts:
            continue
        selected.append({'text': text, 'text_sha256': digest, 'session': session, 'turn': turn})
        sessions.add(session)
        texts.add(text)
        if not topic and len(selected) == COUNT:
            break
    if not selected or (not topic and len(selected) != COUNT):
        raise ValueError('Insufficient distinct sessions')
    return selected


def run(output=OUTPUT, topic=False):
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT / 'runs').resolve()) or output.exists():
        raise ValueError('Expected a new output file under runs')
    files = {'archive': ARCHIVE, 'weight': MODEL / 'model.safetensors', 'original': INDIVIDUAL,
             'individual': CHECKPOINT / 'individual.pt', 'individual_report': CHECKPOINT / 'result.json',
             'locator': LOCATOR / 'locator.pt', 'locator_report': LOCATOR / 'result.json',
             'gate': GATE / 'gate.pt', 'gate_report': GATE / 'result.json'}
    if topic:
        files.update(value=VALUE / 'classifier.pt', value_report=VALUE / 'result.json')
    before = {name: sha(path) for name, path in files.items()}
    if before['archive'] != ARCHIVE_SHA or before['weight'] != WEIGHT_SHA or before['original'] != INDIVIDUAL_SHA:
        raise ValueError('Protected input mismatch')
    reports = {}
    for name in ('individual', 'locator', 'gate'):
        with files[name + '_report'].open(encoding='utf-8') as stream:
            reports[name] = json.load(stream)
    if topic:
        with files['value_report'].open(encoding='utf-8') as stream:
            reports['value'] = json.load(stream)
    if (reports['individual']['checkpoint_sha256'] != before['individual']
            or reports['locator']['locator_sha256'] != before['locator']
            or reports['gate']['gate_sha256'] != before['gate']):
        raise ValueError('Component checkpoint mismatch')
    if topic and reports['value']['classifier_sha256'] != before['value']:
        raise ValueError('Observation checkpoint mismatch')
    config_sha = feature_files_sha256(MODEL)
    locator_data = torch.load(files['locator'], map_location='cpu', weights_only=True)
    gate_data = torch.load(files['gate'], map_location='cpu', weights_only=True)
    value_data = torch.load(files['value'], map_location='cpu', weights_only=True) if topic else None
    threshold = gate_data.get('threshold_logit')
    if (locator_data.get('format') != 'deepseek-name-span-locator-development-v1'
            or locator_data.get('weight_sha256') != WEIGHT_SHA or locator_data.get('feature_config_sha256') != config_sha
            or locator_data.get('layers') != [6, 18] or locator_data.get('threshold') != 0.5
            or gate_data.get('format') != FORMAT or gate_data.get('weight_sha256') != WEIGHT_SHA
            or gate_data.get('individual_sha256') != INDIVIDUAL_SHA or gate_data.get('feature_config_sha256') != config_sha
            or gate_data.get('hidden_indices') != [6, 18] or type(threshold) is not float
            or not math.isfinite(threshold) or reports['gate']['threshold_logit'] != threshold):
        raise ValueError('Component provenance mismatch')
    if topic and (value_data.get('format') != VALUE_FORMAT or value_data.get('weight_sha256') != WEIGHT_SHA
                  or value_data.get('individual_sha256') != INDIVIDUAL_SHA
                  or value_data.get('feature_config_sha256') != config_sha or value_data.get('hidden_indices') != [6, 18]
                  or tuple(value_data.get('vocabulary', ())) != WORDS or value_data.get('abstain') != ABSTAIN):
        raise ValueError('Observation component provenance mismatch')
    samples = sample_archive(ARCHIVE, topic=topic)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    integrated, organism, state = load_integrated(files['individual'], base, MODEL)
    integrated.eval()
    digest = frozen_digest(organism, state)
    features, masks = extract(base, tokenizer, [r['text'] for r in samples], batch_size=2)
    token_vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
    locator = nn.Linear(token_vectors.shape[-1], 1).eval()
    locator.load_state_dict(locator_data['state'], strict=True)
    gate = nn.Linear(gate_data['hidden_width'], 1).eval()
    gate.load_state_dict(gate_data['state'], strict=True)
    with torch.no_grad():
        location_scores = locator(token_vectors).squeeze(-1).sigmoid()
        intent_scores = gate(pool(features, masks)).squeeze(-1)
        value_classes = None
        if topic:
            value_probe = FactorizedProbe(token_vectors.shape[-1]).eval()
            value_probe.load_state_dict(value_data['state'], strict=True)
            fact, value_logits = value_probe(pool(features, masks))
            value_classes = combined_logits(fact, value_logits).argmax(-1)
    bases = [F.normalize(site.current_basis.to('cuda').float(), dim=-1) for site in organism.memories]
    rows = []
    for i, sample in enumerate(samples):
        mask, single = single_span_mask(location_scores[i], masks[i])
        accepted = float(intent_scores[i]) >= threshold
        key, status, answer, baseline = None, None, None, None
        if accepted and single:
            vectors = pooled(features[i:i + 1], mask.unsqueeze(0))
            with torch.inference_mode():
                binding = organism.named_address_binding(vectors, organism, WEIGHT_SHA, config_sha,
                                                         feature_contract=CONTRACT)
            if bool(binding['accepted'][0]):
                key = int(binding['indices'][0, 0])
                status = current_memory_status(organism, state, key)['status']
                if status == 'known':
                    inputs = closed_think_inputs(tokenizer, sample['text'], 'cuda')
                    addresses = [basis[key:key + 1] for basis in bases]
                    with torch.no_grad():
                        plain = integrated.generate(**inputs, max_new_tokens=16, do_sample=False,
                                                    pad_token_id=tokenizer.eos_token_id)
                        generated = integrated.generate(**inputs, memory_state=state, memory_addresses=addresses,
                                                         current_only=True, max_new_tokens=16, do_sample=False,
                                                         pad_token_id=tokenizer.eos_token_id)
                    start = inputs['input_ids'].shape[1]
                    baseline = tokenizer.decode(plain[0, start:], skip_special_tokens=True).strip()
                    answer = tokenizer.decode(generated[0, start:], skip_special_tokens=True).strip()
        rows.append({**sample, 'intent_logit': float(intent_scores[i]), 'intent_accepted': accepted,
                     'single_span': single, 'predicted_key': key, 'status': status,
                     'memory_injected': status == 'known', 'baseline_answer': baseline, 'memory_answer': answer,
                     'observation_value': int(value_classes[i]) if topic else None})
    if (any(sha(path) != before[name] for name, path in files.items()) or feature_files_sha256(MODEL) != config_sha
            or frozen_digest(organism, state) != digest):
        raise ValueError('Protected input changed')
    summary = {'examples': len(rows), 'intent_accepted': sum(r['intent_accepted'] for r in rows),
               'accepted_single_span': sum(r['intent_accepted'] and r['single_span'] for r in rows),
               'accepted_address': sum(r['predicted_key'] is not None for r in rows),
               'memory_injected': sum(r['memory_injected'] for r in rows),
               'observation_nonabstain': sum(r['observation_value'] != ABSTAIN for r in rows) if topic else None}
    report = {'status': 'bounded_external_dialogue_development_audit', 'summary': summary, 'rows': rows,
              'archive_sha256': before['archive'], 'source_member': 'DuLeMon/both/test.json',
              'selection': {'distinct_sessions': not topic, 'text_deduplication': True, 'count': len(samples),
                            'sort': 'sha256 of whitespace-normalized utterance',
                            'topic_filter': topic, 'topic_terms': ['喝', '饮料', '饮品'] if topic else None,
                            'preference_terms': ['喜欢', '爱'] if topic else None,
                            'excluded_terms': None if topic else EXCLUDE},
              'component_sha256': before, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Keyword-filtered single user utterances are treated as out of scope; labels are not original dataset annotations.',
                              'Only selected dialogue turns are evaluated; dialogue context and persona annotations are not supplied.',
                              'No ground-truth supported named-entity drink recall or response quality is evaluated.',
                              'Observation predictions are diagnostic only: no writes or entity/source resolution are performed.',
                              'Name projection and gate were developed on synthetic data; base pretraining overlap is unknown.',
                              'Data usage rights of the external archive require separate verification before redistribution or training.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--topic', action='store_true')
    args = parser.parse_args()
    run(args.output, args.topic)
