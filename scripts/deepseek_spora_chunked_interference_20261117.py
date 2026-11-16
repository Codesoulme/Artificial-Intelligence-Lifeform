"""Multi-person chunk coexistence and overwrite pilot; controlled addresses only."""
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

OUT = ROOT / 'runs/deepseek-spora-chunked-interference-20261117.json'
PEOPLE = ('Alice', 'Bob', 'Carol', 'David', 'Eve')
PREFIX = 24
CHUNKS = 6


def summarize(rows):
    """Count identity/fact packet correctness without hiding decode errors."""
    return {'packets': len(rows), 'identity_exact': sum(row['identity_exact'] for row in rows),
            'fact_exact': sum(row['fact_exact'] for row in rows),
            'both_exact': sum(row['identity_exact'] and row['fact_exact'] for row in rows),
            'decode_errors': sum(row['decode_error'] is not None for row in rows)}


def count_other_losses(before, after, updated_key):
    old = {row['key']: row for row in before}
    new = {row['key']: row for row in after}
    if len(old) != len(before) or len(new) != len(after) or set(old) != set(new) or updated_key not in old:
        raise ValueError('Inconsistent comparison keys')
    return sum(bool(row['identity_exact'] and row['fact_exact']) and
               not (new[key]['identity_exact'] and new[key]['fact_exact'])
               for key, row in old.items() if key != updated_key)


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
    start = time.perf_counter()
    phases = []
    with tempfile.TemporaryDirectory(prefix='spora-chunk-interference-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        model = chat.organism.eval()
        digest = frozen_digest(model, chat.state)
        device = next(model.parameters()).device
        if model.config.key_count < len(PEOPLE)*CHUNKS:
            raise ValueError('Insufficient controlled addresses')
        codec = StructuredTextMemoryCodec(chat.base.model.embed_tokens.weight.detach().cpu(),
                    memory_width=model.config.memory_width, identity_slots=3, fact_slots=5,
                    identity_slot_width=15, fact_slot_width=9, seed=29071,
                    eos_token_id=chat.tokenizer.eos_token_id).to(device)
        items = []
        for person, case in enumerate(cases):
            text = first_user_turn(case)
            if text is None:
                raise ValueError('No user turn')
            source = chat.tokenizer.encode(text, add_special_tokens=False)
            identity = chat.tokenizer.encode(PEOPLE[person], add_special_tokens=False)
            tokens = source[:PREFIX]
            if len(tokens) != PREFIX:
                raise ValueError('Unexpected short input')
            packets = encode_chunks(codec, identity, tokens, max_chunks=CHUNKS)
            if len(packets) != CHUNKS:
                raise ValueError('Unexpected chunk count')
            for index, packet in enumerate(packets):
                items.append({'person': person, 'chunk': index, 'key': person*CHUNKS+index,
                              'packet': packet, 'identity': identity, 'fact': tokens[4*index:4*index+4]})
        if len({tuple(row['identity']) for row in items}) != len(PEOPLE):
            raise ValueError('Identity tokens collide')
        fast, velocity = model.initial_state(1)
        for site in fast:
            site[0] = torch.zeros_like(site[0])

        def write(item):
            with torch.inference_mode():
                for index, site in enumerate(model.memories):
                    basis = site.current_basis[item['key']:item['key']+1].to(device)
                    fast[index], velocity[index] = site.write_temporal(
                        fast[index], velocity[index], basis, item['packet'].packet.reshape(1, -1).to(device),
                        torch.ones(1, device=device), torch.zeros(1, dtype=torch.long, device=device), False)

        def read(item):
            with torch.inference_mode():
                vectors = [site.retrieve([fast[i][0]], site.current_basis[item['key']:item['key']+1].to(device))[0]
                           for i, site in enumerate(model.memories)]
                try:
                    decoded = codec.decode(torch.stack(vectors).mean(0), item['packet'].metadata)
                    ident = decoded['identity']['token_ids'] == item['identity']
                    fact = decoded['fact']['token_ids'] == item['fact']
                    error = None
                except ValueError as exc:
                    ident, fact, error = False, False, str(exc)
            return {'person': item['person'], 'chunk': item['chunk'], 'key': item['key'],
                    'identity_exact': ident, 'fact_exact': fact, 'decode_error': error}

        written = []
        for chunk in range(CHUNKS):
            for item in (row for row in items if row['chunk'] == chunk):
                write(item)
                written.append(item)
            rows = [read(item) for item in written]
            phase = {'round': chunk + 1, 'summary': summarize(rows), 'rows': rows}
            phases.append(phase)
            print(json.dumps({'round': chunk + 1, **phase['summary']}), flush=True)
        before = phases[-1]['rows']
        original = items[0]
        replacement_fact = items[CHUNKS]['fact']
        if replacement_fact == original['fact']:
            raise ValueError('Replacement matches original')
        replacement_packet = encode_chunks(codec, original['identity'], replacement_fact, max_chunks=1)[0]
        replacement = {**original, 'packet': replacement_packet, 'fact': replacement_fact}
        write(replacement)
        after_rows = [read(replacement if item['key'] == original['key'] else item) for item in items]
        after = {'summary': summarize(after_rows), 'rows': after_rows,
                 'updated_chunk_exact': next(r for r in after_rows if r['key'] == original['key'])['fact_exact'],
                 'other_previously_correct_lost': count_other_losses(before, after_rows, original['key'])}
        if frozen_digest(model, chat.state) != digest:
            raise ValueError('Protected individual changed')
    if any(runtime.sha(path) != hashes[name] for name, path in paths.items()):
        raise ValueError('Protected input changed')
    report = {'status': 'controlled_interference_pilot_not_deployed', 'phases': phases, 'after_update': after,
              'source_sha256': hashes, 'seconds': time.perf_counter()-start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Five 24-token prefixes from previously inspected benchmark cases.',
                              'Identities, fact boundaries, numeric addresses, chunk order and lengths supplied explicitly.',
                              'Only current transient matrix; no natural routing, historical versions, persistence or QA.',
                              'One fixed order and one overwrite; no population generalization or capacity proof.',
                              'No parameter training; codec and metadata are auxiliary components.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'final': phases[-1]['summary'], 'after_update': after['summary'],
                      'updated_chunk_exact': after['updated_chunk_exact'],
                      'other_previously_correct_lost': after['other_previously_correct_lost']}), flush=True)


if __name__ == '__main__':
    main()
