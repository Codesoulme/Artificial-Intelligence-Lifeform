"""Fixed output-space-aligned memory representation: first-token research probe.

This is a matched-filter-style architectural upper bound, not learned language
understanding. Only the existing frozen LM head produces output logits; the
token is encoded at write time from its pretrained output-head row. Identity,
memory address and whether to enable the residual are supplied by the caller.
"""
import hashlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch.nn import functional as F

from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_chunk_first_token_bridge_20261117 import (
    CORPUS, MANIFEST, PROMPT, SEED, TRAIN_TYPES, HELD_TYPES,
    FirstTokenBridge, actual_memory_code, chinese_token_counts, split_token_types)
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.language_training import frozen_digest
from spora.structured_text_memory import StructuredTextMemoryCodec

OUT = ROOT / 'runs/deepseek-spora-output-aligned-memory-pilot-20261117.json'
CANDIDATE = ROOT / 'runs/deepseek-spora-chunk-first-token-longer-20261117'
PROJECTION_SEED = 28017
SCALES = (0., 1., 3., 10., 30., 100., 300., 1000.)
CALIBRATION = 64
TEST = 128
ORDINARY = ('今天下雨了，出门应该带什么？', '请计算三加四等于多少。', '请用一句话介绍你自己。')


def make_projection(hidden_width, memory_width, seed):
    if any(type(x) is not int or x < 1 for x in (hidden_width, memory_width)) or type(seed) is not int:
        raise ValueError('Invalid projection dimensions')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return torch.randn((hidden_width, memory_width), generator=generator, dtype=torch.float32) / math.sqrt(hidden_width)


def encode_output_rows(rows, projection):
    if (rows.ndim != 2 or projection.ndim != 2 or rows.shape[1] != projection.shape[0]
            or not bool(torch.isfinite(rows).all()) or not bool(torch.isfinite(projection).all())):
        raise ValueError('Invalid output rows or projection')
    projected = rows.float() @ projection.float()
    if bool((projected.norm(dim=-1) < 1e-8).any()):
        raise ValueError('Degenerate projected output row')
    return F.normalize(projected, dim=-1)


def actual_vector(model, value, device):
    """Controlled single-address native write/read; no text or logits are stored."""
    if value.ndim != 1 or value.numel() != model.config.memory_width:
        raise ValueError('Invalid continuous value')
    fast, velocity = model.initial_state(1)
    for site in fast:
        site[0] = torch.zeros_like(site[0])
    key = 1
    with torch.no_grad():
        for i, site in enumerate(model.memories):
            address = site.current_basis[key:key+1].to(device)
            fast[i], velocity[i] = site.write_temporal(
                fast[i], velocity[i], address, value.reshape(1, -1).to(device),
                torch.ones(1, device=device), torch.zeros(1, dtype=torch.long, device=device), False)
        return torch.stack([site.retrieve([fast[i][0]], site.current_basis[key:key+1].to(device))[0]
                            for i, site in enumerate(model.memories)]).mean(0).detach()


def choose_scale(scored):
    if not scored:
        raise ValueError('Empty calibration')
    return min(scored, key=lambda item: (-item['exact'], item['scale']))['scale']


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    files = {'base': runtime.MODEL / 'model.safetensors', 'integrated': runtime.CHECKPOINT,
             'corpus': CORPUS, 'candidate': CANDIDATE / 'candidate.pt'}
    hashes = {name: runtime.sha(path) for name, path in files.items()}
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    prior = json.loads((CANDIDATE / 'result.json').read_text(encoding='utf-8'))
    if (hashes['base'] != runtime.WEIGHT_SHA or hashes['corpus'] != manifest['files'][CORPUS.name]
            or hashes['candidate'] != prior['candidate_sha256']):
        raise ValueError('Source provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='spora-output-aligned-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        device = next(chat.organism.parameters()).device
        digest = frozen_digest(chat.organism, chat.state)
        counts = chinese_token_counts(chat.tokenizer, CORPUS)
        train, held = split_token_types(counts, TRAIN_TYPES, HELD_TYPES, SEED)
        calibration, test = train[:CALIBRATION], held[128:128+TEST]
        if len(test) != TEST or set(calibration) & set(test):
            raise ValueError('Calibration and new held-out types overlap')
        projection = make_projection(chat.base.lm_head.weight.shape[1], chat.organism.config.memory_width,
                                     PROJECTION_SEED).to(device)
        projection_sha = hashlib.sha256(projection.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        head = chat.base.lm_head.eval()
        prompt = closed_think_inputs(chat.tokenizer, PROMPT, 'cuda')
        with torch.inference_mode():
            h0 = chat.base.model(**prompt, use_cache=False, return_dict=True).last_hidden_state[:, -1].float().detach()

        def codes_for(ids):
            idx = torch.tensor(ids, dtype=torch.long, device=device)
            return encode_output_rows(head.weight.detach().index_select(0, idx), projection)

        def predict(codes, scale, hidden=h0):
            predictions = []
            with torch.no_grad():
                for index in range(0, len(codes), 16):
                    delta = codes[index:index+16].float() @ projection.T
                    logits = head((hidden + float(scale)*delta).to(head.weight.dtype))
                    predictions.extend(logits.argmax(-1).cpu().tolist())
            return predictions

        cal_codes = codes_for(calibration)
        scored = []
        for scale in SCALES:
            predictions = predict(cal_codes, scale)
            scored.append({'scale': scale, 'exact': sum(p == t for p, t in zip(predictions, calibration))})
        chosen = choose_scale(scored)
        print(json.dumps({'calibration': scored, 'selected_scale': chosen}), flush=True)
        direct_codes = codes_for(test)
        direct_predictions = predict(direct_codes, chosen)
        native_vectors = torch.stack([actual_vector(chat.organism, code, device) for code in direct_codes])
        cosine = F.cosine_similarity(native_vectors, direct_codes, dim=-1)
        native_codes = F.normalize(native_vectors.float(), dim=-1)
        native_predictions = predict(native_codes, chosen)
        zero_predictions = predict(torch.zeros_like(native_codes), chosen)

        def pair_metrics(predictions):
            return {'total': TEST, 'exact': sum(p == t for p, t in zip(predictions, test)),
                    'pairs': TEST//2,
                    'both_correct': sum(predictions[i] == test[i] and predictions[i+1] == test[i+1]
                                        for i in range(0, TEST, 2)),
                    'outputs_differ': sum(predictions[i] != predictions[i+1] for i in range(0, TEST, 2)),
                    'predicted_ids': predictions}

        # Previous learned 9-D bridge on the same fresh half of held-out types.
        payload = torch.load(CANDIDATE / 'candidate.pt', map_location='cpu', weights_only=True)
        if (payload.get('format') != 'spora-first-token-hidden-bridge-research-v1'
                or payload.get('base_sha256') != hashes['base'] or payload.get('integrated_sha256') != hashes['integrated']):
            raise ValueError('Learned bridge provenance mismatch')
        codec = StructuredTextMemoryCodec(chat.base.model.embed_tokens.weight.detach().cpu(),
                    memory_width=chat.organism.config.memory_width, identity_slots=3, fact_slots=5,
                    identity_slot_width=15, fact_slot_width=9, seed=29071,
                    eos_token_id=chat.tokenizer.eos_token_id).to(device)
        if codec.config_dict() != payload['codec_config']:
            raise ValueError('Learned bridge codec mismatch')
        bridge = FirstTokenBridge(codec.fact_slot_width, h0.shape[-1]).to(device).eval()
        bridge.load_state_dict(payload['state'], strict=True)
        identity = chat.tokenizer.encode('用户', add_special_tokens=False)
        learned_codes, storage = [], []
        for token in test:
            code, exact = actual_memory_code(chat.organism, codec, identity, token, device)
            learned_codes.append(code)
            storage.append(exact)
        learned_predictions = []
        with torch.no_grad():
            learned_codes = torch.stack(learned_codes)
            for index in range(0, TEST, 16):
                hidden = h0 + bridge(learned_codes[index:index+16])
                learned_predictions.extend(head(hidden.to(head.weight.dtype)).argmax(-1).cpu().tolist())

        ordinary = []
        for question in ORDINARY:
            encoded = closed_think_inputs(chat.tokenizer, question, 'cuda')
            with torch.inference_mode():
                hidden = chat.base.model(**encoded, use_cache=False, return_dict=True).last_hidden_state[:, -1].float()
                plain = int(head(hidden.to(head.weight.dtype)).argmax(-1)[0])
                forced = int(head((hidden + chosen*(native_codes[0:1] @ projection.T)).to(head.weight.dtype)).argmax(-1)[0])
            ordinary.append({'question': question, 'first_token_changed_when_forced': plain != forced})
        if frozen_digest(chat.organism, chat.state) != digest:
            raise ValueError('Original individual changed')
    if any(runtime.sha(path) != hashes[name] for name, path in files.items()):
        raise ValueError('Protected input changed')
    result = {'status': 'output_aligned_memory_research_only_not_deployed',
              'calibration_types': CALIBRATION, 'test_types': TEST, 'calibration_scores': scored,
              'chosen_scale': chosen, 'projection_seed': PROJECTION_SEED, 'projection_sha256': projection_sha,
              'aligned_direct': pair_metrics(direct_predictions), 'aligned_native': pair_metrics(native_predictions),
              'zero_memory': pair_metrics(zero_predictions), 'learned_9d_native': pair_metrics(learned_predictions),
              'learned_storage_exact': sum(storage),
              'native_vector_cosine': {'minimum': float(cosine.min()), 'mean': float(cosine.mean())},
              'ordinary_forced': ordinary, 'source_sha256': hashes,
              'seconds': time.perf_counter()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Fixed output-head projection acts as a matched-filter vocabulary decoder.',
                              'Write-time token identity selects a pretrained output-head row; not autonomous learning.',
                              'One token per 90-dimensional memory vector, using explicit address and oracle activation.',
                              'First-token prediction only, one prompt, no semantic QA or multi-token generation.',
                              'Scale chosen on training types; test types are the previously unused held-out half.',
                              'Different representation width and architecture from the learned 9-D baseline.',
                              'Ordinary-prompt test forces memory on; no trained relevance or safety gate.',
                              'No trained model weights, production state changes, or deployment.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'chosen_scale': chosen, 'direct': result['aligned_direct']['exact'],
                      'native': result['aligned_native']['exact'],
                      'learned': result['learned_9d_native']['exact'],
                      'pairs': result['aligned_native']['both_correct'],
                      'ordinary_changed': sum(r['first_token_changed_when_forced'] for r in ordinary)}), flush=True)


if __name__ == '__main__':
    main()
