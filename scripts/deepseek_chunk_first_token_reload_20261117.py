"""Independent-process reload of both frozen-LM first-token bridge candidates."""
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch

from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_chunk_first_token_bridge_20261117 import (
    CORPUS, MANIFEST, PROMPT, SEED, TRAIN_TYPES, HELD_TYPES, EVAL_TYPES,
    FirstTokenBridge, actual_memory_code, chinese_token_counts, split_token_types)
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.language_training import frozen_digest
from spora.structured_text_memory import StructuredTextMemoryCodec

SOURCES = {
    'longer': ROOT / 'runs/deepseek-spora-chunk-first-token-longer-20261117',
    'broader': ROOT / 'runs/deepseek-spora-chunk-first-token-broader-20261117',
}
OUT = ROOT / 'runs/deepseek-spora-chunk-first-token-reload-20261117.json'


def predict(bridge, head, h0, codes, device):
    outputs = []
    with torch.no_grad():
        for i in range(0, len(codes), 16):
            hidden = h0 + bridge(codes[i:i+16].to(device))
            outputs.extend(head(hidden.to(head.weight.dtype)).argmax(-1).cpu().tolist())
    return outputs


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    base_sha = runtime.sha(runtime.MODEL / 'model.safetensors')
    checkpoint_sha = runtime.sha(runtime.CHECKPOINT)
    corpus_sha = runtime.sha(CORPUS)
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if base_sha != runtime.WEIGHT_SHA or corpus_sha != manifest['files'][CORPUS.name]:
        raise ValueError('Base or corpus mismatch')
    source = {}
    for name, directory in SOURCES.items():
        result = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        path = directory / 'candidate.pt'
        if (runtime.sha(path) != result['candidate_sha256'] or result['source_sha256'] !=
                {'base': base_sha, 'integrated': checkpoint_sha, 'corpus': corpus_sha}):
            raise ValueError('Candidate provenance mismatch')
        source[name] = (path, result)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = {}
    with tempfile.TemporaryDirectory(prefix='spora-token-bridge-reload-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        device = next(chat.organism.parameters()).device
        digest = frozen_digest(chat.organism, chat.state)
        counts = chinese_token_counts(chat.tokenizer, CORPUS)
        _, held = split_token_types(counts, TRAIN_TYPES, HELD_TYPES, SEED)
        held_hash = hashlib.sha256(json.dumps(held, separators=(',', ':')).encode('ascii')).hexdigest()
        broader_hash = source['broader'][1]['held_token_types_sha256']
        if held_hash != broader_hash:
            raise ValueError('Held-out split mismatch')
        codec = StructuredTextMemoryCodec(chat.base.model.embed_tokens.weight.detach().cpu(),
                    memory_width=chat.organism.config.memory_width, identity_slots=3, fact_slots=5,
                    identity_slot_width=15, fact_slot_width=9, seed=29071,
                    eos_token_id=chat.tokenizer.eos_token_id).to(device)
        prompt = closed_think_inputs(chat.tokenizer, PROMPT, 'cuda')
        prompt_hash = hashlib.sha256(prompt['input_ids'].detach().cpu().numpy().tobytes()).hexdigest()
        with torch.inference_mode():
            h0 = chat.base.model(**prompt, use_cache=False, return_dict=True).last_hidden_state[:, -1].float().detach()
        head = chat.base.lm_head
        target = held[:EVAL_TYPES]
        direct_codes = codec.fact_codebook.index_select(0, torch.tensor(target, dtype=torch.long, device=device)).detach()
        identity = chat.tokenizer.encode('用户', add_special_tokens=False)
        memory_codes, storage = [], []
        for value in target:
            code, exact = actual_memory_code(chat.organism, codec, identity, value, device)
            memory_codes.append(code)
            storage.append(exact)
        memory_codes = torch.stack(memory_codes)
        for name, (path, recorded) in source.items():
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if (payload.get('format') != 'spora-first-token-hidden-bridge-research-v1'
                    or payload.get('base_sha256') != base_sha or payload.get('integrated_sha256') != checkpoint_sha
                    or payload.get('prompt_sha256') != prompt_hash or payload.get('codec_config') != codec.config_dict()
                    or payload.get('seed') != SEED or payload.get('steps') != 4000):
                raise ValueError('Candidate metadata mismatch')
            bridge = FirstTokenBridge(codec.fact_slot_width, h0.shape[-1]).to(device).eval()
            bridge.load_state_dict(payload['state'], strict=True)
            direct = predict(bridge, head, h0, direct_codes, device)
            native = predict(bridge, head, h0, memory_codes, device)
            direct_exact = sum(p == t for p, t in zip(direct, target))
            native_exact = sum(p == t for p, t in zip(native, target))
            paired = sum(native[i] == target[i] and native[i+1] == target[i+1]
                         for i in range(0, EVAL_TYPES, 2))
            recorded_native = recorded['held_native_memory']
            output[name] = {'direct_predictions_equal': direct == recorded['held_direct_code']['predicted_ids'],
                            'direct_exact': direct_exact,
                            'native_exact': native_exact, 'native_exact_matches': native_exact == recorded_native['exact'],
                            'native_pairs_correct': paired,
                            'native_pair_count_matches': paired == recorded_native['pair_both_correct'],
                            'storage_exact': sum(storage),
                            'storage_count_matches': sum(storage) == recorded_native['storage_exact']}
            print(json.dumps({'candidate': name, **output[name]}), flush=True)
        if frozen_digest(chat.organism, chat.state) != digest:
            raise ValueError('Original individual changed')
    if (runtime.sha(runtime.MODEL / 'model.safetensors') != base_sha or runtime.sha(runtime.CHECKPOINT) != checkpoint_sha
            or runtime.sha(CORPUS) != corpus_sha or
            any(runtime.sha(path) != recorded['candidate_sha256'] for path, recorded in source.values())):
        raise ValueError('Protected input changed')
    report = {'status': 'separate_process_bridge_reload', 'results': output, 'held_types_sha256': held_hash,
              'prompt_sha256': prompt_hash, 'seconds': time.perf_counter()-started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Only existing fixed prompt and 128 held token types.',
                              'Native prediction arrays were not saved originally; only aggregate counts are compared.',
                              'No multi-token generation, new facts, open dialogue, persistence or real user data.',
                              'Fixed oracle address; reconstructed memory is a temporary individual.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'completed': len(output)}), flush=True)


if __name__ == '__main__':
    main()
