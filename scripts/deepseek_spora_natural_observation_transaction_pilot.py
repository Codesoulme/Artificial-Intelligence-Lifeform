"""Run a natural-language observation through an isolated SPORA individual."""
import argparse
import gc
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

from scripts import deepseek_spora_runtime_chat as chat_runtime
from scripts import deepseek_spora_end_to_end_candidate_audit as route
from spora.observation_candidate import ObservationCandidate
from spora.growing import query


OUTPUT = ROOT / 'runs/deepseek-spora-natural-observation-transaction-pilot-v2-20261113'
WORKING = OUTPUT / 'working-individual.pt'
POLICY_ID = 'spora-polarity-calibration-20261111-v1'
CHECKPOINT = route.CHECKPOINT
NATURAL_HEADS = None

CASES = [
    {'kind': 'positive', 'text': '小梅偏爱的饮料是茶。', 'expected': '茶'},
    {'kind': 'negative', 'text': '小梅不喜欢喝茶。', 'expected': None},
    {'kind': 'unknown_value', 'text': '小梅偏爱的饮料是热巧克力。', 'expected': None},
    {'kind': 'correction', 'text': '小梅偏爱的饮料是咖啡。', 'expected': '咖啡'},
]


def build_components(chat):
    tokenizer = chat.tokenizer
    base = chat.base
    organism = chat.organism
    locator_data = torch.load(route.LOCATOR / 'locator.pt', map_location='cpu', weights_only=True)
    binding_data = torch.load(route.BINDING / 'binding.pt', map_location='cpu', weights_only=True)
    polarity_data = torch.load(route.POLARITY / 'gate.pt', map_location='cpu', weights_only=True)
    value_data = torch.load(route.VALUE, map_location='cpu', weights_only=True)
    locator = nn.Linear(locator_data['state']['weight'].shape[1], 1).eval()
    locator.load_state_dict(locator_data['state'], strict=True)
    bases = torch.stack([F.normalize(site.current_basis.detach().float().cpu(), dim=-1)
                         for site in organism.memories])
    if chat.natural_write_heads is not None:
        polarity = chat.natural_write_heads.polarity
        polarity_data = {'threshold': chat.natural_write_heads.polarity_threshold}
        value = chat.natural_write_heads.value
        value_data = dict(value_data, threshold=chat.natural_write_heads.value_threshold)
    else:
        polarity = nn.Linear(base.config.hidden_size * 2, len(route.CLASSES)).eval()
        polarity.load_state_dict(polarity_data['state'], strict=True)
        value = route.SpanHead(base.config.hidden_size).eval()
        value.load_state_dict(value_data['state'], strict=True)
    return tokenizer, locator, binding_data, polarity, polarity_data, value, value_data, bases


def parse_candidate(chat, text, components):
    tokenizer, locator, binding_data, polarity, polarity_data, value, value_data, bases = components
    rows = [{'text': text, 'start': 0, 'end': 0, 'positive': True}]
    vectors, masks, offsets, starts = route.feature_batch(chat.base, tokenizer, rows)
    token_vectors = torch.cat([F.layer_norm(vectors[:, i].float(), (vectors.shape[-1],))
                               for i in range(2)], dim=-1)
    with torch.inference_mode():
        subject_scores = locator(token_vectors).squeeze(-1).sigmoid()[0]
        probabilities = polarity(route.pool(vectors, masks)).softmax(-1)[0]
    value_rows = route.decode_values(rows, value, vectors, offsets, starts, masks,
                                     value_data['threshold'])
    selected, single = route.single_span_mask(subject_scores, masks[0])
    span = route.pooled(vectors, selected.unsqueeze(0)).detach()
    indices, similarities = route.predict(span.cpu(), binding_data['weights'], bases)
    polarity_index = int(probabilities.argmax())
    value_row = value_rows[0]
    address = indices[0].tolist()
    address_scores = similarities[0].tolist()
    address_ok = (single and address[0] == address[1]
                  and min(address_scores) >= binding_data['threshold'])
    value_text = value_row['value']
    value_index = route.WORDS.index(value_text) if value_text in route.WORDS else None
    polarity_ok = (polarity_index == 0
                   and float(probabilities[0]) >= polarity_data['threshold'])
    value_ok = bool(value_row['accepted'] and value_index is not None)
    candidate = ObservationCandidate(
        address[0] if address_ok else None, value_text, value_index,
        route.CLASSES[polarity_index], 'direct', 1.0, chat.state.cursor,
        float(probabilities[0]), text)
    gate = {'policy_id': POLICY_ID,
            'authorized': bool(address_ok and polarity_ok and value_ok),
            'confidence': 1.0 if address_ok and polarity_ok and value_ok else 0.0}
    return {'text': text, 'predicted_address': address, 'address_ok': bool(address_ok),
            'address_scores': address_scores, 'polarity': route.CLASSES[polarity_index],
            'polarity_probability': float(probabilities[0]), 'polarity_ok': polarity_ok,
            'value': value_text, 'value_index': value_index, 'value_ok': value_ok,
            'candidate': candidate, 'gate': gate}


def verify(path, key):
    chat_runtime.CHECKPOINT = CHECKPOINT
    chat_runtime.WORKING_CHECKPOINT = Path(path)
    chat = chat_runtime.load_chat(natural_write_heads_path=NATURAL_HEADS)
    device = next(chat.organism.parameters()).device
    if chat.state.cursor == 0:
        raise ValueError('Expected persisted observations after independent reload')
    value = int(query(chat.organism, chat.state,
                      torch.tensor([key], device=device),
                      torch.zeros(1, dtype=torch.long, device=device),
                      torch.zeros(1, dtype=torch.long, device=device)).argmax(-1)[0])
    return {'cursor': chat.state.cursor, 'key': key, 'current_value': value,
            'working_sha256': chat_runtime.sha(Path(path))}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=False, exist_ok=False)
    started = time.perf_counter()
    chat_runtime.CHECKPOINT = CHECKPOINT
    chat_runtime.WORKING_CHECKPOINT = WORKING
    chat = chat_runtime.load_chat(natural_write_heads_path=NATURAL_HEADS)
    components = build_components(chat)
    rows = []
    for case in CASES:
        parsed = parse_candidate(chat, case['text'], components)
        candidate = parsed.pop('candidate')
        gate = parsed.pop('gate')
        try:
            result = chat.observe_candidate(candidate, calibrated_gate=gate, policy_id=POLICY_ID)
            action = result.get('action', 'written') if isinstance(result, dict) else 'written'
        except Exception as exc:
            action = 'error'
            result = {'error': type(exc).__name__ + ': ' + str(exc)}
        rows.append({**case, **parsed, 'action': action, 'result': result,
                     'cursor_after': chat.state.cursor})
    observed_key = rows[0]['predicted_address'][0]
    if not rows[0]['address_ok']:
        raise ValueError('Initial natural-language observation did not resolve one entity')
    device = next(chat.organism.parameters()).device
    current_value = int(query(chat.organism, chat.state,
                              torch.tensor([observed_key], device=device),
                              torch.zeros(1, dtype=torch.long, device=device),
                              torch.zeros(1, dtype=torch.long, device=device)).argmax(-1)[0])
    before_reload = {'cursor': chat.state.cursor, 'key': observed_key,
                     'current_value': current_value, 'working_sha256': chat_runtime.sha(WORKING)}
    pre = {'status': 'awaiting_independent_reload', 'cases': rows,
           'before_reload': before_reload, 'correction_action': rows[-1]['action'],
           'seconds_before_process_exit': time.perf_counter() - started,
           'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
    (OUTPUT / 'pre-reload.json').write_text(json.dumps(pre, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    del chat, components
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({'actions': [row['action'] for row in rows],
                      'working_checkpoint': str(WORKING)}, ensure_ascii=False), flush=True)


def finalize():
    pre = json.loads((OUTPUT / 'pre-reload.json').read_text(encoding='utf-8'))
    key = int(pre['before_reload']['key'])
    after_reload = verify(WORKING, key)
    report = {'status': 'isolated_natural_observation_transaction',
              'cases': pre['cases'], 'before_reload': pre['before_reload'],
              'after_independent_reload': after_reload,
              'same_current_value': pre['before_reload']['current_value'] == after_reload['current_value'],
              'correction_action': pre['correction_action'],
              'seconds_before_process_exit': pre['seconds_before_process_exit'],
              'limitations': [
                  'Temporary working checkpoint; formal individual is untouched.',
                  'Finite beverage vocabulary and existing development gate artifacts.',
                  'The new sentences are a small authored challenge set, not open-domain language.',
                  'Natural-language parser and gate are not yet the default chat entry point.',
              ]}
    (OUTPUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'actions': [row['action'] for row in report['cases']],
                      'same_current_value': report['same_current_value']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', nargs=2)
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.verify[0], int(args.verify[1])), ensure_ascii=False), flush=True)
    elif args.finalize:
        finalize()
    else:
        run()
