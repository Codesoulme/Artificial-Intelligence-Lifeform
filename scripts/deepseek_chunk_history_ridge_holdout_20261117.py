"""Fixed-ridge comparison on new authored-benchmark inputs and write orders.

This is an in-memory development probe, not a natural-language memory benchmark.
All addresses, event times, identities and ground-truth tokens are supplied.
"""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch

from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_chunk_history_replay_study_20261117 import solve_replay
from scripts.deepseek_chunk_variable_crash_pilot_20261117 import SOURCE, INPUT, NAMES, resources
from scripts.deepseek_spora_chunked_memory_pilot_20261117 import first_user_turn
from spora.language_training import frozen_digest
from spora.research_chunk_history_store import ResearchChunkHistoryStore
from spora.research_chunked_text import encode_chunks

OUT = ROOT / 'runs/deepseek-spora-chunk-ridge-holdout-20261117.json'
SEEDS = (1941, 1942, 1943)
RIDGES = (0.01, 0.001)  # fixed before inspecting this data
CHECKPOINTS = (30, 60, 62, 64)


def fixed_inputs(path, tokenizer):
    """First 11 distinct eligible user turns with case number 200 or higher."""
    selected, seen = [], set()
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            case = json.loads(line)
            number = int(case['case_id'].split('-')[1])
            if number < 200:
                continue
            content = first_user_turn(case)
            if content is None:
                continue
            text_sha = hashlib.sha256(content.encode('utf-8')).hexdigest()
            tokens = tokenizer.encode(content, add_special_tokens=False)
            if len(tokens) < 24 or text_sha in seen:
                continue
            selected.append({'case_id': case['case_id'], 'tokens': tokens[:24], 'text_sha256': text_sha})
            seen.add(text_sha)
            if len(selected) == 11:
                break
    if len(selected) != 11:
        raise ValueError('Insufficient eligible inputs')
    return selected


def build_events(inputs, identity_ids, codec, seed):
    """Two shuffled 30-packet waves, then four fixed continuation packets."""
    waves = []
    for version in (0, 1):
        wave = []
        for person in range(5):
            tokens = inputs[person + 5*version]['tokens']
            packets = encode_chunks(codec, identity_ids[person], tokens, max_chunks=6)
            if len(packets) != 6:
                raise ValueError('Expected six packets')
            for chunk, packet in enumerate(packets):
                wave.append({'key': person*6+chunk, 'person': person, 'version': version,
                             'chunk': chunk, 'packet': packet.packet.detach(), 'metadata': packet.metadata,
                             'identity': identity_ids[person], 'fact': tokens[chunk*4:chunk*4+4]})
        random.Random(seed + version*10000).shuffle(wave)
        waves.extend(wave)
    extra = inputs[10]['tokens'][:16]
    packets = encode_chunks(codec, identity_ids[0], extra, max_chunks=4)
    for chunk, packet in enumerate(packets):
        waves.append({'key': (chunk if chunk < 2 else 33+chunk-2), 'person': 0, 'version': 2,
                      'chunk': chunk, 'packet': packet.packet.detach(), 'metadata': packet.metadata,
                      'identity': identity_ids[0], 'fact': extra[chunk*4:chunk*4+4]})
    if len(waves) != 64:
        raise ValueError('Unexpected stream length')
    return waves


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = {'source': SOURCE, 'input': INPUT, 'base': runtime.MODEL / 'model.safetensors'}
    hashes = {name: runtime.sha(path) for name, path in paths.items()}
    if hashes['base'] != runtime.WEIGHT_SHA:
        raise ValueError('Base mismatch')
    tokenizer, codec = resources()
    inputs = fixed_inputs(INPUT, tokenizer)
    identities = [tokenizer.encode(name, add_special_tokens=False) for name in NAMES]
    store = ResearchChunkHistoryStore.open(SOURCE, codec)
    start = time.perf_counter()
    results = []
    try:
        model = store.model.eval()
        digest = frozen_digest(model, store.state)
        if len(model.memories) != 2 or model.config.memory_width != 90 or model.config.time_count < 64:
            raise ValueError('Unexpected model dimensions')
        for seed in SEEDS:
            events = build_events(inputs, identities, codec, seed)
            fasts, velocity = {}, {}
            for ridge in RIDGES:
                fasts[ridge], velocity[ridge] = [], []
                for site in model.memories:
                    zero = torch.zeros((1, 90, 90), dtype=torch.float32)
                    precision = torch.eye(90, dtype=torch.float32).unsqueeze(0) / ridge
                    params = [zero.clone(), zero.clone(), precision]
                    fasts[ridge].append(params)
                    velocity[ridge].append([torch.zeros_like(x) for x in params])
            addresses = [[] for _ in model.memories]
            targets = []
            batch_fast, batch_velocity = None, None
            checkpoints = []

            def score(matrices, count):
                rows = []
                for i, event in enumerate(events[:count]):
                    vector = torch.stack([matrices[j] @ addresses[j][i] for j in range(2)]).mean(0)
                    try:
                        decoded = codec.decode(vector, event['metadata'])
                        correct = (decoded['identity']['token_ids'] == event['identity'] and
                                   decoded['fact']['token_ids'] == event['fact'])
                        error = None
                    except ValueError as exc:
                        correct, error = False, str(exc)
                    rows.append({'event_time': i, 'person': event['person'], 'version': event['version'],
                                 'chunk': event['chunk'], 'exact': correct, 'decode_error': error})
                return {'count': len(rows), 'exact': sum(row['exact'] for row in rows),
                        'failures': [row for row in rows if not row['exact']]}

            with torch.no_grad():
                for index, event in enumerate(events):
                    key = torch.tensor([event['key']], dtype=torch.long)
                    mode, when = torch.ones_like(key), torch.full_like(key, index)
                    _, bound = model.representation(key, mode, when)
                    addresses_for_event = [model.memory_address(site, bound, key, mode) for site in model.memories]
                    for site_index, address in enumerate(addresses_for_event):
                        addresses[site_index].append(address[0].detach().clone())
                    target = event['packet'].reshape(1, -1)
                    targets.append(event['packet'])
                    for ridge in RIDGES:
                        for site_index, site in enumerate(model.memories):
                            fasts[ridge][site_index], velocity[ridge][site_index] = site.write_temporal(
                                fasts[ridge][site_index], velocity[ridge][site_index],
                                addresses_for_event[site_index], target, torch.ones(1), mode, False)
                    if batch_fast is not None:
                        for site_index, site in enumerate(model.memories):
                            batch_fast[site_index], batch_velocity[site_index] = site.write_temporal(
                                batch_fast[site_index], batch_velocity[site_index],
                                addresses_for_event[site_index], target, torch.ones(1), mode, False)
                    if index + 1 == 62:
                        batch_fast, batch_velocity = [], []
                        y = torch.stack(targets)
                        for site_index, site in enumerate(model.memories):
                            q = torch.stack(addresses[site_index])
                            mat = solve_replay(q, y, 0.001).unsqueeze(0)
                            gram = q.double().T @ q.double() + .001*torch.eye(90, dtype=torch.float64)
                            precision = torch.linalg.inv(gram).float().unsqueeze(0)
                            params = [torch.zeros_like(mat), mat, precision]
                            batch_fast.append(params)
                            batch_velocity.append([torch.zeros_like(x) for x in params])
                    if index + 1 in CHECKPOINTS:
                        point = {'events': index+1, 'online': {str(r): score([v[1][0] for v in fasts[r]], index+1)
                                                               for r in RIDGES}}
                        if batch_fast is not None:
                            point['batch_then_online_0.001'] = score([v[1][0] for v in batch_fast], index+1)
                        checkpoints.append(point)
                        print(json.dumps({'seed': seed, 'events': index+1,
                                          'online_0.01': point['online']['0.01']['exact'],
                                          'online_0.001': point['online']['0.001']['exact'],
                                          'batch_0.001': point.get('batch_then_online_0.001', {}).get('exact')}), flush=True)
            results.append({'seed': seed, 'checkpoints': checkpoints})
        if frozen_digest(model, store.state) != digest:
            raise ValueError('Source state changed')
    finally:
        store.close()
    if any(runtime.sha(path) != hashes[name] for name, path in paths.items()):
        raise ValueError('Protected source changed')
    report = {'status': 'fixed_ridge_unseen_turns_research_probe', 'results': results,
              'selected_cases': [{'case_id': row['case_id'], 'text_sha256': row['text_sha256']} for row in inputs],
              'source_sha256': hashes, 'seconds_after_codec': time.perf_counter()-start,
              'limitations': ['11 previously uninspected English benchmark turns; three synthetic write orders.',
                              'Source texts may have appeared in model pretraining; no claims of true distribution independence.',
                              'Exact identities, source values, token boundaries, event times and numeric addresses supplied.',
                              'History matrix only: no semantics, provenance, natural retrieval, generation or durable recovery.',
                              '0.001 was chosen from an earlier failure; three seeds do not establish reliability.',
                              'Batch reconstruction fits all 62 past events; only the final two writes follow the fit.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'completed_seeds': len(results)}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
