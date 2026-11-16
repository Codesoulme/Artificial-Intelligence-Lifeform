"""Controlled vector writes to current and historical SPORA memory matrices.

Historical event times and all addresses are supplied by the experiment, not
inferred from a question. Spans are simulated revisions, not factual corrections.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch

from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_spora_chunked_memory_pilot_20261117 import first_user_turn
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.longmemeval_no_experience_diagnostic_20261117 import INPUT, select_cases
from spora.language_training import frozen_digest
from spora.research_chunked_text import encode_chunks
from spora.structured_text_memory import StructuredTextMemoryCodec

OUT = ROOT / 'runs/deepseek-spora-chunked-history-20261117.json'
IDENTITIES = ('Alice', 'Bob', 'Carol', 'David', 'Eve')
CHUNKS = 6
TOKENS = CHUNKS * 4


def summarize(rows):
    return {'packets': len(rows), 'identity_exact': sum(r['identity_exact'] for r in rows),
            'fact_exact': sum(r['fact_exact'] for r in rows),
            'both_exact': sum(r['identity_exact'] and r['fact_exact'] for r in rows),
            'decode_errors': sum(r['decode_error'] is not None for r in rows)}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    cases = select_cases(INPUT)
    paths = {'input': INPUT, 'base': runtime.MODEL / 'model.safetensors', 'checkpoint': runtime.CHECKPOINT}
    hashes = {name: runtime.sha(path) for name, path in paths.items()}
    if hashes['base'] != runtime.WEIGHT_SHA:
        raise ValueError('Base provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    phases = []
    with tempfile.TemporaryDirectory(prefix='spora-chunk-history-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        model = chat.organism.eval()
        digest = frozen_digest(model, chat.state)
        device = next(model.parameters()).device
        if (model.config.key_count < 30 or model.config.time_count < 42 or
                model.config.temporal_kind != 'split' or model.config.history_writer != 'rls'):
            raise ValueError('Unexpected individual capacity or history configuration')
        codec = StructuredTextMemoryCodec(chat.base.model.embed_tokens.weight.detach().cpu(),
                    memory_width=model.config.memory_width, identity_slots=3, fact_slots=5,
                    identity_slot_width=15, fact_slot_width=9, seed=29071,
                    eos_token_id=chat.tokenizer.eos_token_id).to(device)
        texts = [first_user_turn(case) for case in cases]
        if any(text is None for text in texts):
            raise ValueError('Missing user turn')
        sources = [chat.tokenizer.encode(text, add_special_tokens=False) for text in texts]
        # case-0054 has enough tokens for three distinct 24-token spans.
        target = sources[2]
        if len(target) < 3*TOKENS:
            raise ValueError('Target is too short')
        versions = [target[i*TOKENS:(i+1)*TOKENS] for i in range(3)]
        if len({tuple(x) for x in versions}) != 3:
            raise ValueError('Simulated versions are not distinct')
        distractors = [sources[i][:TOKENS] for i in (0, 1, 3, 4)]
        if any(len(tokens) != TOKENS for tokens in distractors):
            raise ValueError('Distractor is too short')
        identities = [chat.tokenizer.encode(name, add_special_tokens=False) for name in IDENTITIES]
        if len({tuple(ids) for ids in identities}) != len(identities):
            raise ValueError('Identity collision')
        fast, velocity = model.initial_state(1)
        for site in fast:
            site[0] = torch.zeros_like(site[0])
            site[1] = torch.zeros_like(site[1])
        events, current = [], {}

        def write(person, version, tokens):
            chunks = encode_chunks(codec, identities[person], tokens, max_chunks=CHUNKS)
            if len(chunks) != CHUNKS:
                raise ValueError('Unexpected chunk count')
            for index, chunk in enumerate(chunks):
                key, event_time = person*CHUNKS+index, len(events)
                keys = torch.tensor([key], dtype=torch.long, device=device)
                times = torch.tensor([event_time], dtype=torch.long, device=device)
                with torch.inference_mode():
                    for mode in (0, 1):
                        modes = torch.full_like(keys, mode)
                        _, bound = model.representation(keys, modes, times)
                        for site_index, site in enumerate(model.memories):
                            address = model.memory_address(site, bound, keys, modes)
                            fast[site_index], velocity[site_index] = site.write_temporal(
                                fast[site_index], velocity[site_index], address,
                                chunk.packet.reshape(1, -1).to(device), torch.ones(1, device=device),
                                modes, False)
                event = {'key': key, 'person': person, 'version': version, 'chunk': index,
                         'event_time': event_time, 'packet': chunk, 'identity': identities[person],
                         'fact': tokens[index*4:index*4+4]}
                events.append(event)
                current[key] = event

        def read(event, mode):
            keys = torch.tensor([event['key']], dtype=torch.long, device=device)
            modes = torch.full_like(keys, mode)
            times = torch.full_like(keys, event['event_time'])
            with torch.inference_mode():
                _, bound = model.representation(keys, modes, times)
                vectors = [site.retrieve([fast[i][mode]], model.memory_address(site, bound, keys, modes))[0]
                           for i, site in enumerate(model.memories)]
                try:
                    decoded = codec.decode(torch.stack(vectors).mean(0), event['packet'].metadata)
                    ident = decoded['identity']['token_ids'] == event['identity']
                    fact = decoded['fact']['token_ids'] == event['fact']
                    error = None
                except ValueError as exc:
                    ident, fact, error = False, False, str(exc)
            return {'key': event['key'], 'person': event['person'], 'version': event['version'],
                    'chunk': event['chunk'], 'event_time': event['event_time'],
                    'identity_exact': ident, 'fact_exact': fact, 'decode_error': error}

        def record_phase(version):
            curr = [read(current[key], 0) for key in sorted(current)]
            hist = [read(event, 1) for event in events]
            item = {'version': version, 'current': summarize(curr), 'history_oracle': summarize(hist),
                    'history_target_by_version': {str(v): summarize([r for r in hist
                                                if r['person'] == 0 and r['version'] == v])
                                                  for v in range(version+1)},
                    'current_rows': curr, 'history_rows': hist}
            phases.append(item)
            print(json.dumps({'version': version, 'current': item['current'],
                              'history_oracle': item['history_oracle']}), flush=True)

        write(0, 0, versions[0])
        for person, tokens in enumerate(distractors, 1):
            write(person, 0, tokens)
        record_phase(0)
        for version in (1, 2):
            write(0, version, versions[version])
            record_phase(version)
        if frozen_digest(model, chat.state) != digest:
            raise ValueError('Protected individual changed')
    if any(runtime.sha(path) != hashes[name] for name, path in paths.items()):
        raise ValueError('Protected input changed')
    report = {'status': 'isolated_current_history_vector_pilot_not_deployed', 'phases': phases,
              'source_sha256': hashes, 'seconds': time.perf_counter()-started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Three spans of one turn are simulated updates, not real corrections.',
                              'Explicit person identity, numeric addresses, exact event times and chunk boundaries.',
                              'Historical times are oracle-provided; no event selection or natural question is tested.',
                              'No normal write guards, durable events, crash recovery, inference output or training.',
                              'One fixed event sequence and five previously inspected benchmark cases.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'completed': True, 'current': phases[-1]['current'],
                      'history_oracle': phases[-1]['history_oracle']}), flush=True)


if __name__ == '__main__':
    main()
