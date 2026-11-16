"""Learn a fixed input-embedding to output-space memory bridge.

Token IDs select frozen input embeddings at write time. The trainable mapping
is fitted on disjoint token types to reconstruct projected frozen output rows.
This is a token-level probe, not natural language question answering.
"""
import json
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
    chinese_token_counts, extend_train_types, split_token_types)
from scripts.deepseek_output_aligned_holdout_20261117 import grouped_vectors, metrics, ORDINARY
from scripts.deepseek_output_aligned_memory_pilot_20261117 import encode_output_rows, make_projection
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.language_training import frozen_digest

OUT = ROOT / 'runs/deepseek-spora-input-output-memory-pilot-v2-20261117.json'
INPUT_SEED, OUTPUT_SEED = 28201, 28017
TRAIN_TOTAL, CAL, TEST, GROUP = 12000, 64, 128, 32
RIDGE = 0.1
SCALES = (0., 10., 30., 100., 300., 1000.)


def fit_projection(inputs, targets, ridge):
    if (inputs.ndim != 2 or targets.ndim != 2 or inputs.shape[0] != targets.shape[0]
            or inputs.shape[0] < 2 or ridge <= 0 or not torch.isfinite(inputs).all()
            or not torch.isfinite(targets).all()):
        raise ValueError('Invalid fit inputs')
    x, y = inputs.float(), targets.float()
    return torch.linalg.solve(x.T @ x + ridge * torch.eye(x.shape[1], device=x.device), x.T @ y)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = {'base': runtime.MODEL/'model.safetensors', 'integrated': runtime.CHECKPOINT, 'corpus': CORPUS}
    hashes = {k: runtime.sha(v) for k, v in paths.items()}
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if hashes['base'] != runtime.WEIGHT_SHA or hashes['corpus'] != manifest['files'][CORPUS.name]:
        raise ValueError('Source mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='spora-input-output-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder)/'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        device = next(chat.organism.parameters()).device
        digest = frozen_digest(chat.organism, chat.state)
        counts = chinese_token_counts(chat.tokenizer, CORPUS)
        train, held = split_token_types(counts, TRAIN_TYPES, HELD_TYPES, SEED)
        train = extend_train_types(counts, train, held, TRAIN_TOTAL, SEED)
        calibration, train = train[:CAL], train[CAL:]
        test = held[:TEST]
        if set(train) & set(calibration + test) or set(calibration) & set(test):
            raise ValueError('Token types overlap')
        head = chat.base.lm_head.eval()
        embed = chat.base.model.embed_tokens.weight.detach()
        p_in = make_projection(embed.shape[1], chat.organism.config.memory_width, INPUT_SEED).to(device)
        p_out = make_projection(head.weight.shape[1], chat.organism.config.memory_width, OUTPUT_SEED).to(device)

        def representations(ids):
            idx = torch.tensor(ids, dtype=torch.long, device=device)
            return (encode_output_rows(embed.index_select(0, idx), p_in),
                    encode_output_rows(head.weight.detach().index_select(0, idx), p_out))

        x_train, y_train = representations(train)
        transform = fit_projection(x_train, y_train, RIDGE)
        x_cal, _ = representations(calibration)
        x_test, y_test = representations(test)
        prompt = closed_think_inputs(chat.tokenizer, PROMPT, 'cuda')
        with torch.inference_mode():
            h0 = chat.base.model(**prompt, use_cache=False, return_dict=True).last_hidden_state[:, -1].float()

        def convert(x):
            return F.normalize(x.float() @ transform, dim=-1)

        def predict(codes, scale, hidden=h0):
            predictions = []
            with torch.no_grad():
                for i in range(0, len(codes), 16):
                    delta = convert(codes[i:i+16]) @ p_out.T
                    predictions.extend(head((hidden+scale*delta).to(head.weight.dtype)).argmax(-1).cpu().tolist())
            return predictions

        scored = [{'scale': s, 'exact': sum(p == t for p, t in zip(predict(x_cal, s), calibration))}
                  for s in SCALES]
        scale = min(scored, key=lambda r: (-r['exact'], r['scale']))['scale']
        direct = predict(x_test, scale)
        native = torch.cat([grouped_vectors(chat.organism, x_test[i:i+GROUP], device)
                            for i in range(0, TEST, GROUP)])
        actual = predict(F.normalize(native.float(), dim=-1), scale)
        # If every calibration scale ties at zero correct, the conservative
        # selector chooses zero. Also report the predeclared matched-filter
        # scale so this does not silently test only the empty intervention.
        fixed_predictions = predict(F.normalize(native.float(), dim=-1), 1000.)
        cosine = F.cosine_similarity(native, x_test, dim=-1)
        reconstructed = F.cosine_similarity(convert(x_test), y_test, dim=-1)
        zero = predict(torch.zeros_like(x_test), scale)
        ordinary = []
        for q in ORDINARY:
            inputs = closed_think_inputs(chat.tokenizer, q, 'cuda')
            with torch.inference_mode():
                h = chat.base.model(**inputs, use_cache=False, return_dict=True).last_hidden_state[:, -1].float()
                base_id = int(head(h.to(head.weight.dtype)).argmax(-1)[0])
            ordinary.append({'question': q,
                             'calibrated_changed': predict(x_test[:1], scale, h)[0] != base_id,
                             'fixed_1000_changed': predict(x_test[:1], 1000., h)[0] != base_id})
        if frozen_digest(chat.organism, chat.state) != digest:
            raise ValueError('Original individual changed')
    if any(runtime.sha(v) != hashes[k] for k, v in paths.items()):
        raise ValueError('Protected source changed')
    result = {'status': 'input_projection_ridge_research_not_deployed', 'training_types': len(train),
              'calibration_types': CAL, 'test_types': TEST, 'ridge': RIDGE, 'scales': scored,
              'chosen_scale': scale, 'direct': metrics(direct, test), 'native': metrics(actual, test),
              'zero': metrics(zero, test), 'fixed_1000_native': metrics(fixed_predictions, test),
              'mean_predicted_target_cosine': float(reconstructed.mean()),
              'min_native_cosine': float(cosine.min()), 'ordinary_forced': ordinary,
              'source_sha256': hashes, 'seconds': time.perf_counter()-started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Token identity selects frozen input embedding, no learned experience semantics.',
                              'Training uses frozen output-head rows as regression targets.',
                              'Token types disjoint in this corpus; base pretraining exposure unknown.',
                              'First token and one fixed prompt only; no natural question or automatic routing.',
                              'Four groups of 32 explicit addresses; no persistence or history.',
                              'Single ridge and input projection seed, selected before evaluation.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'scale': scale, 'direct': result['direct']['exact'], 'native': result['native']['exact'],
                      'mean_cosine': result['mean_predicted_target_cosine']}), flush=True)


if __name__ == '__main__':
    main()
