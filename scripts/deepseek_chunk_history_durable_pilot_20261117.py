"""Two-process durable test for a copied SPORA temporal core, never production state."""
import argparse
import json
import struct
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from transformers import AutoTokenizer

from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_spora_chunked_memory_pilot_20261117 import first_user_turn
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_prefix_features import read_bf16_tensor
from scripts.longmemeval_no_experience_diagnostic_20261117 import INPUT, select_cases
from spora.language_training import frozen_digest
from spora.research_chunk_history_store import ResearchChunkHistoryStore
from spora.structured_text_memory import StructuredTextMemoryCodec
from spora.temporal import TemporalSPORA

DIR = ROOT / 'runs/deepseek-spora-chunk-history-durable-20261117'
DATABASE = DIR / 'research.sqlite'
WRITE_REPORT = DIR / 'write.json'
READ_REPORT = DIR / 'read.json'
NAMES = ('Alice', 'Bob', 'Carol', 'David', 'Eve')


def codec_from_embedding(embedding, tokenizer):
    return StructuredTextMemoryCodec(embedding, memory_width=90, identity_slots=3, fact_slots=5,
                                     identity_slot_width=15, fact_slot_width=9, seed=29071,
                                     eos_token_id=tokenizer.eos_token_id)


def expected_rows(tokenizer):
    cases = select_cases(INPUT)
    sources = [tokenizer.encode(first_user_turn(c), add_special_tokens=False) for c in cases]
    if len(sources[2]) < 72 or any(len(sources[i]) < 24 for i in (0, 1, 3, 4)):
        raise ValueError('Unexpected input lengths')
    versions = [sources[2][i*24:(i+1)*24] for i in range(3)]
    others = [[sources[i][:24]] for i in (0, 1, 3, 4)]
    return [versions] + others


def write_phase():
    if DIR.exists():
        raise FileExistsError(DIR)
    paths = {'base': runtime.MODEL / 'model.safetensors', 'checkpoint': runtime.CHECKPOINT, 'input': INPUT}
    hashes = {name: runtime.sha(path) for name, path in paths.items()}
    if hashes['base'] != runtime.WEIGHT_SHA:
        raise ValueError('Base mismatch')
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory(prefix='spora-durable-copy-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        original = frozen_digest(chat.organism, chat.state)
        device = next(chat.organism.parameters()).device
        # Copy only the temporal model's parameters; do not transfer the
        # real individual's controllers, runtime state, or existing memories.
        model = TemporalSPORA(chat.organism.config).to(device).eval()
        source = chat.organism.state_dict()
        model.load_state_dict({k: source[k] for k in model.state_dict()}, strict=True)
        model.requires_grad_(False)
        codec = codec_from_embedding(chat.base.model.embed_tokens.weight.detach().cpu(), chat.tokenizer).to(device)
        expected = expected_rows(chat.tokenizer)
        identities = [chat.tokenizer.encode(name, add_special_tokens=False) for name in NAMES]
        DIR.mkdir(parents=True)
        store = ResearchChunkHistoryStore.create(DATABASE, model, codec)
        writes = []
        try:
            for person in range(5):
                writes.append(store.write(person, list(range(person*6, person*6+6)),
                                          identities[person], expected[person][0]))
            for version in (1, 2):
                writes.append(store.write(0, list(range(6)), identities[0], expected[0][version], allow_update=True))
            state_digest = frozen_digest(store.model, store.state)
            cursor, revision = store.state.cursor, store.store.revision
        finally:
            store.close()
        if frozen_digest(chat.organism, chat.state) != original:
            raise ValueError('Original individual changed')
    if any(runtime.sha(path) != hashes[name] for name, path in paths.items()):
        raise ValueError('Protected input changed')
    report = {'status': 'isolated_copied_core_written', 'source_sha256': hashes,
              'state_digest': state_digest, 'cursor': cursor, 'revision': revision,
              'writes': writes, 'seconds': time.perf_counter()-started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
    with WRITE_REPORT.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'status': report['status'], 'cursor': cursor, 'revision': revision}), flush=True)


def read_phase():
    if READ_REPORT.exists():
        raise FileExistsError(READ_REPORT)
    report = json.loads(WRITE_REPORT.read_text(encoding='utf-8'))
    weights = runtime.MODEL / 'model.safetensors'
    if (runtime.sha(weights) != report['source_sha256']['base']
            or runtime.sha(INPUT) != report['source_sha256']['input']
            or runtime.sha(runtime.CHECKPOINT) != report['source_sha256']['checkpoint']):
        raise ValueError('Source changed')
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(runtime.MODEL, local_files_only=True, trust_remote_code=False)
    with weights.open('rb') as stream:
        header_size = struct.unpack('<Q', stream.read(8))[0]
        if header_size > 100_000_000 or header_size > weights.stat().st_size - 8:
            raise ValueError('Invalid header')
        header = json.loads(stream.read(header_size))
        embedding = read_bf16_tensor(stream, header, 8 + header_size, weights.stat().st_size,
                                     'model.embed_tokens.weight').to(torch.float16)
    codec = codec_from_embedding(embedding, tokenizer)
    del embedding
    expected = expected_rows(tokenizer)
    identities = [tokenizer.encode(name, add_special_tokens=False) for name in NAMES]
    store = ResearchChunkHistoryStore.open(DATABASE, codec, device='cpu')
    try:
        digest = frozen_digest(store.model, store.state)
        if digest != report['state_digest'] or store.store.revision != report['revision']:
            raise ValueError('Reloaded individual differs from committed state')
        rows = []
        for person, versions in enumerate(expected):
            current = store.read(person)
            rows.append({'record': person, 'mode': 'current',
                         'exact': current == {'identity_ids': identities[person], 'fact_ids': versions[-1]}})
            for version, tokens in enumerate(versions):
                result = store.read(person, version)
                rows.append({'record': person, 'mode': 'history', 'version': version,
                             'exact': result == {'identity_ids': identities[person], 'fact_ids': tokens}})
        if frozen_digest(store.model, store.state) != digest:
            raise ValueError('Read modified individual')
    finally:
        store.close()
    result = {'status': 'separate_process_reload_diagnostic', 'rows': rows,
              'digest_matches': True, 'seconds': time.perf_counter()-started,
              'limitations': ['A copied uncontrolled temporal core, not the production individual.',
                              'Numeric addresses, identity, boundaries and event versions supplied by experiment.',
                              'No language routing, natural writes, open QA, or true semantic correction.',
                              'Five inspected cases; model weights not trained.']}
    with READ_REPORT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'current': sum(r['exact'] for r in rows if r['mode'] == 'current'),
                      'current_total': sum(r['mode'] == 'current' for r in rows),
                      'history': sum(r['exact'] for r in rows if r['mode'] == 'history'),
                      'history_total': sum(r['mode'] == 'history' for r in rows)}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('write', 'read'))
    args = parser.parse_args()
    write_phase() if args.phase == 'write' else read_phase()
