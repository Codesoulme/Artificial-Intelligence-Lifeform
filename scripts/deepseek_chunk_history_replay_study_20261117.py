"""Offline, oracle-addressed replay study on a copied SPORA history matrix.

Old packet targets come from decoding the pre-update source, not from gold
answers. New packet targets are explicitly supplied. No candidate is deployed.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch

from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_chunk_variable_crash_pilot_20261117 import (
    SOURCE, INPUT, NAMES, desired, resources, target_versions)
from scripts.deepseek_spora_chunked_memory_pilot_20261117 import first_user_turn
from scripts.longmemeval_no_experience_diagnostic_20261117 import select_cases
from spora.language_training import frozen_digest
from spora.research_chunk_history_store import ResearchChunkHistoryStore
from spora.research_chunked_text import encode_chunks

CLEAN = ROOT / 'runs/deepseek-spora-chunk-clean-control-20261117/clean.sqlite'
OUT = ROOT / 'runs/deepseek-spora-chunk-history-replay-continuation-20261117.json'
RIDGES = (0.01, 0.001, 0.0001, 0.00001)


def solve_replay(addresses, targets, ridge):
    """Solve ridge least squares in float64; return float32 persisted matrix."""
    if (not isinstance(addresses, torch.Tensor) or not isinstance(targets, torch.Tensor)
            or addresses.ndim != 2 or targets.ndim != 2 or addresses.shape[0] != targets.shape[0]
            or not addresses.shape[0] or not addresses.shape[1] or not targets.shape[1]
            or not isinstance(ridge, (int, float)) or isinstance(ridge, bool) or not 0 < ridge < 1
            or not bool(torch.isfinite(addresses).all()) or not bool(torch.isfinite(targets).all())):
        raise ValueError('Invalid replay inputs')
    q, y = addresses.double(), targets.double()
    gram = q.T @ q + float(ridge) * torch.eye(q.shape[1], dtype=q.dtype, device=q.device)
    return torch.linalg.solve(gram, q.T @ y).T.float()


def read_packet(store, codec, event_time):
    event = store.metadata['events'][event_time]
    key = torch.tensor([event['key']], dtype=torch.long)
    mode, when = torch.ones_like(key), torch.full_like(key, event_time)
    with torch.no_grad():
        _, bound = store.model.representation(key, mode, when)
        vectors = [site.retrieve([store.state.pages[0][i][0]],
                                 store.model.memory_address(site, bound, key, mode))[0]
                   for i, site in enumerate(store.model.memories)]
        return codec.decode(torch.stack(vectors).mean(0), event['metadata'])


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = {'source': SOURCE, 'clean': CLEAN, 'input': INPUT}
    hashes = {name: runtime.sha(path) for name, path in paths.items()}
    tokenizer, codec = resources()
    original, short, long = target_versions(tokenizer)
    truth = [original[0] + [short, long]] + original[1:]
    identities = [tokenizer.encode(name, add_special_tokens=False) for name in NAMES]
    new_tokens = desired(tokenizer)
    new_packets = encode_chunks(codec, identities[0], new_tokens, max_chunks=9)
    start = time.perf_counter()
    source = ResearchChunkHistoryStore.open(SOURCE, codec)
    clean = ResearchChunkHistoryStore.open(CLEAN, codec)
    try:
        if (source.state.cursor != 53 or clean.state.cursor != 62
                or source.metadata['events'] != clean.metadata['events'][:53]):
            raise ValueError('Source snapshots do not match the recorded scenario')
        source_digest = frozen_digest(source.model, source.state)
        clean_digest = frozen_digest(clean.model, clean.state)
        events = []
        for event_time, event in enumerate(source.metadata['events']):
            rec, ver, chunk = event['record'], event['version'], event['chunk']
            decoded = read_packet(source, codec, event_time)
            expected_fact = truth[rec][ver][chunk*4:chunk*4+4]
            if decoded['identity']['token_ids'] != identities[rec] or decoded['fact']['token_ids'] != expected_fact:
                raise ValueError('Pre-update historical source is not exact')
            packet, metadata = codec.encode(decoded['identity']['token_ids'], decoded['fact']['token_ids'])
            if metadata != event['metadata']:
                raise ValueError('Reconstructed boundaries differ')
            events.append({'key': event['key'], 'time': event_time, 'record': rec, 'version': ver,
                           'chunk': chunk, 'packet': packet.detach(), 'metadata': metadata,
                           'identity': identities[rec], 'fact': expected_fact})
        keys = [0, 1, 2, 3, 4, 5, 30, 31, 32]
        if len(new_packets) != len(keys):
            raise ValueError('Unexpected new packet count')
        for index, (key, packet) in enumerate(zip(keys, new_packets)):
            event_time = 53 + index
            event = clean.metadata['events'][event_time]
            if (event['key'], event['record'], event['version'], event['chunk']) != (key, 0, 5, index):
                raise ValueError('Committed update metadata mismatch')
            events.append({'key': key, 'time': event_time, 'record': 0, 'version': 5,
                           'chunk': index, 'packet': packet.packet.detach(), 'metadata': packet.metadata,
                           'identity': identities[0], 'fact': new_tokens[index*4:index*4+4]})
        addresses = [[] for _ in source.model.memories]
        with torch.no_grad():
            for event in events:
                key = torch.tensor([event['key']], dtype=torch.long)
                mode, when = torch.ones_like(key), torch.full_like(key, event['time'])
                _, bound = source.model.representation(key, mode, when)
                for i, site in enumerate(source.model.memories):
                    addresses[i].append(source.model.memory_address(site, bound, key, mode)[0].detach())
        addresses = [torch.stack(q).float() for q in addresses]
        targets = torch.stack([event['packet'].float() for event in events])

        def evaluate(matrices, evaluated_events=None, evaluated_addresses=None):
            evaluated_events = events if evaluated_events is None else evaluated_events
            evaluated_addresses = addresses if evaluated_addresses is None else evaluated_addresses
            rows = []
            for index, event in enumerate(evaluated_events):
                reconstructed = torch.stack([matrix @ evaluated_addresses[i][index]
                                               for i, matrix in enumerate(matrices)]).mean(0)
                try:
                    decoded = codec.decode(reconstructed, event['metadata'])
                    exact = (decoded['identity']['token_ids'] == event['identity'] and
                             decoded['fact']['token_ids'] == event['fact'])
                    reason = None
                except ValueError as exc:
                    exact, reason = False, str(exc)
                rows.append({'event_time': event['time'], 'record': event['record'],
                             'version': event['version'], 'chunk': event['chunk'],
                             'exact': exact, 'decode_error': reason})
            return {'events': len(rows), 'exact': sum(r['exact'] for r in rows),
                    'old_exact': sum(r['exact'] for r in rows[:53]),
                    'new_exact': sum(r['exact'] for r in rows[53:]),
                    'failures': [r for r in rows if not r['exact']]}

        baseline = [site[0][0].detach().float() for site in clean.state.pages[0]]
        results = {'online_original': evaluate(baseline)}
        diagnostics = []
        for i, q in enumerate(addresses):
            singular = torch.linalg.svdvals(q.double())
            diagnostics.append({'site': i, 'events': q.shape[0], 'address_width': q.shape[1],
                                'largest_singular': float(singular.max()),
                                'smallest_singular': float(singular.min()),
                                'condition': float(singular.max()/singular.min())})
        for ridge in RIDGES:
            matrices = [solve_replay(q, targets, ridge) for q in addresses]
            result = evaluate(matrices)
            errors = [float(((q @ matrix.T - targets)**2).mean()) for q, matrix in zip(addresses, matrices)]
            result['per_site_mse'] = errors
            results[f'batch_float64_ridge_{ridge}'] = result
            print(json.dumps({'ridge': ridge, 'exact': result['exact'], 'events': result['events']}), flush=True)
        # Two additional, fixed input chunks are not included in the replay
        # solve. This is an online-continuation diagnostic, not independent
        # evaluation of the post-hoc ridge choice.
        text = first_user_turn(select_cases(INPUT)[2])
        extra_ids = tokenizer.encode(text, add_special_tokens=False)[56:64]
        if len(extra_ids) != 8:
            raise ValueError('Continuation tokens unavailable')
        extra_packets = encode_chunks(codec, identities[0], extra_ids, max_chunks=2)
        if len(extra_packets) != 2:
            raise ValueError('Unexpected continuation extent')
        added = []
        for i, packet in enumerate(extra_packets):
            added.append({'key': 33+i, 'time': 62+i, 'record': 0, 'version': 6,
                          'chunk': i, 'packet': packet.packet.detach(), 'metadata': packet.metadata,
                          'identity': identities[0], 'fact': extra_ids[i*4:i*4+4]})
        all_events = events + added
        all_addresses = [torch.cat((q, torch.zeros((2, q.shape[1]), dtype=q.dtype)), dim=0)
                         for q in addresses]
        with torch.no_grad():
            for j, event in enumerate(added):
                keys = torch.tensor([event['key']], dtype=torch.long)
                modes, times = torch.ones_like(keys), torch.full_like(keys, event['time'])
                _, bound = source.model.representation(keys, modes, times)
                for i, site in enumerate(source.model.memories):
                    all_addresses[i][len(events)+j] = source.model.memory_address(site, bound, keys, modes)[0]
        continuation = {}
        for name in ('online_original', 'batch_ridge_0.001'):
            fast, velocity = [], []
            for i, site in enumerate(source.model.memories):
                if name == 'online_original':
                    matrix = clean.state.pages[0][i][0].detach().clone()
                    precision = clean.state.pages[0][i][1].detach().clone()
                else:
                    q = addresses[i].double()
                    gram = q.T @ q + 0.001*torch.eye(q.shape[1], dtype=torch.float64)
                    matrix = solve_replay(addresses[i], targets, 0.001).unsqueeze(0)
                    precision = torch.linalg.inv(gram).float().unsqueeze(0)
                params = [torch.zeros_like(matrix), matrix, precision]
                fast.append(params)
                velocity.append([torch.zeros_like(x) for x in params])
            before = evaluate([site[1][0] for site in fast])
            with torch.no_grad():
                for j, event in enumerate(added):
                    for i, site in enumerate(source.model.memories):
                        fast[i], velocity[i] = site.write_temporal(
                            fast[i], velocity[i], all_addresses[i][len(events)+j].reshape(1, -1),
                            event['packet'].reshape(1, -1), torch.ones(1), torch.ones(1, dtype=torch.long), False)
            after = evaluate([site[1][0] for site in fast], all_events, all_addresses)
            continuation[name] = {'before': before, 'after_two_writes': after}
            print(json.dumps({'continuation': name, 'before': before['exact'],
                              'after': after['exact'], 'events': after['events']}), flush=True)
        if (frozen_digest(source.model, source.state) != source_digest or
                frozen_digest(clean.model, clean.state) != clean_digest):
            raise ValueError('Source individual changed')
    finally:
        source.close()
        clean.close()
    if any(runtime.sha(path) != hashes[name] for name, path in paths.items()):
        raise ValueError('Protected source changed')
    report = {'status': 'offline_replay_diagnostic_not_deployed', 'results': results,
              'continuation': continuation,
              'address_diagnostics': diagnostics, 'source_sha256': hashes,
              'seconds_after_codec': time.perf_counter()-start,
              'limitations': ['One previously observed failure and four post-hoc ridge settings.',
                              'Historical content is decoded with known addresses and exact timestamps.',
                              'Replay reads all historical contents; requires working pre-update source.',
                              'No current matrix change, durable transaction, language routing or generation.',
                              'Batch float64 solve with float32 result; fitted on the first 62 evaluated events.',
                              'Two additional event writes use exact manual addresses and one post-hoc ridge choice.',
                              'No claim that this method corrects already-corrupt or unidentifiable memories.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'baseline': results['online_original']['exact'],
                      'best': max(results[k]['exact'] for k in results if k != 'online_original')}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
