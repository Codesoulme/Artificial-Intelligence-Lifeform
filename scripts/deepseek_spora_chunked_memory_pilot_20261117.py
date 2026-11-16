"""Bounded real-model chunk storage pilot; no natural routing or QA is tested."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch

from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.longmemeval_no_experience_diagnostic_20261117 import INPUT, select_cases
from spora.language_training import frozen_digest
from spora.research_chunked_text import TextChunk, decode_chunks, encode_chunks
from spora.structured_text_memory import StructuredTextMemoryCodec

OUT = ROOT / 'runs/deepseek-spora-chunked-memory-pilot-20261117.json'
PREFIX_TOKENS = 24


def first_user_turn(case):
    return next((turn['content'] for session in case['sessions'] for turn in session['turns']
                 if turn['role'] == 'user' and turn['content'].strip()), None)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    cases = select_cases(INPUT)
    hashes = {'input': runtime.sha(INPUT), 'base': runtime.sha(runtime.MODEL / 'model.safetensors'),
              'checkpoint': runtime.sha(runtime.CHECKPOINT)}
    if hashes['base'] != runtime.WEIGHT_SHA:
        raise ValueError('Base provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix='spora-chunked-pilot-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        model = chat.organism.eval()
        digest = frozen_digest(model, chat.state)
        device = next(model.parameters()).device
        codec = StructuredTextMemoryCodec(chat.base.model.embed_tokens.weight.detach().cpu(),
                    memory_width=model.config.memory_width, identity_slots=3, fact_slots=5,
                    identity_slot_width=15, fact_slot_width=9, seed=29071,
                    eos_token_id=chat.tokenizer.eos_token_id).to(device)
        identity = chat.tokenizer.encode('user', add_special_tokens=False)
        for case in cases:
            text = first_user_turn(case)
            if text is None:
                raise ValueError('No nonempty user turn')
            all_ids = chat.tokenizer.encode(text, add_special_tokens=False)
            chosen = all_ids[:PREFIX_TOKENS]
            chunks = encode_chunks(codec, identity, chosen, max_chunks=model.config.key_count)
            direct = decode_chunks(codec, chunks, max_chunks=model.config.key_count)
            fast, velocity = model.initial_state(1)
            for site in fast:
                site[0] = torch.zeros_like(site[0])
            with torch.inference_mode():
                for key, chunk in enumerate(chunks):
                    target = chunk.packet.reshape(1, -1).to(device)
                    for i, site in enumerate(model.memories):
                        basis = site.current_basis[key:key + 1].to(device)
                        fast[i], velocity[i] = site.write_temporal(
                            fast[i], velocity[i], basis, target, torch.ones(1, device=device),
                            torch.zeros(1, dtype=torch.long, device=device), False)
                recovered = []
                for key, chunk in enumerate(chunks):
                    vectors = [site.retrieve([fast[i][0]], site.current_basis[key:key + 1].to(device))[0]
                               for i, site in enumerate(model.memories)]
                    recovered.append(TextChunk(chunk.index, chunk.total, torch.stack(vectors).mean(0),
                                               dict(chunk.metadata)))
                try:
                    neural = decode_chunks(codec, recovered, max_chunks=model.config.key_count)
                    exact = neural['identity_ids'] == identity and neural['fact_ids'] == chosen
                    error = None
                except ValueError as exc:
                    exact, error = False, str(exc)
            row = {'case_id': case['case_id'], 'full_turn_tokens': len(all_ids),
                   'prefix_tokens': len(chosen), 'chunk_count': len(chunks),
                   'direct_roundtrip_exact': direct['identity_ids'] == identity and direct['fact_ids'] == chosen,
                   'neural_roundtrip_exact': exact, 'neural_error': error}
            rows.append(row)
            print(json.dumps(row), flush=True)
        if frozen_digest(model, chat.state) != digest:
            raise ValueError('Protected individual changed')
        key_count = model.config.key_count
    if (runtime.sha(INPUT) != hashes['input'] or runtime.sha(runtime.MODEL / 'model.safetensors') != hashes['base']
            or runtime.sha(runtime.CHECKPOINT) != hashes['checkpoint']):
        raise ValueError('Protected input changed')
    report = {'status': 'isolated_chunked_memory_pilot_not_deployed', 'rows': rows,
              'key_count': key_count, 'fact_tokens_per_packet': 4,
              'source_sha256': hashes, 'seconds': time.perf_counter()-started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Five previously inspected benchmark cases; only first user-turn prefix up to 24 tokens.',
                              'Explicit identity, numeric addresses, chunk order and packet lengths supplied externally.',
                              'No natural fact extraction, semantic addressing, persistent storage, generation or QA.',
                              'The codec and chunk metadata are external machinery; this is not a trained model weight update.',
                              'Each case uses fresh transient matrices and no interference from other conversations.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'direct_exact': sum(r['direct_roundtrip_exact'] for r in rows),
                      'neural_exact': sum(r['neural_roundtrip_exact'] for r in rows), 'cases': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
