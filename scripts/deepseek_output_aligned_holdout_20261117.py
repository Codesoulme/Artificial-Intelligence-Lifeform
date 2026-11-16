"""Fixed matched-filter representation on corpus-disjoint Chinese token types.

This evaluates one-token copying with manually supplied memory addresses. The
pretrained output head itself provides the vocabulary geometry, so this is not
evidence that the model learned the token-to-meaning relation from experience.
"""
import hashlib
import json
import random
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
from scripts.deepseek_chunk_first_token_bridge_20261117 import CORPUS, MANIFEST, PROMPT, chinese_token_counts
from scripts.deepseek_output_aligned_memory_pilot_20261117 import (
    actual_vector, encode_output_rows, make_projection)
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.language_training import frozen_digest

TEST_CORPUS = ROOT / 'runs/clean-zh-pretrain-v1/pretrain-test.jsonl'
PRIOR = ROOT / 'runs/deepseek-spora-output-aligned-memory-pilot-20261117.json'
OUT = ROOT / 'runs/deepseek-spora-output-aligned-holdout-20261117.json'
SEED = 37191
COUNT = 128
GROUP = 32
ORDINARY = ('请把十减三的结果写出来。', '春天常见的天气是什么样？', '请问怎样泡一杯绿茶？',
            '用一句话解释什么是地图。', '请说一个简短的问候语。', '路上遇到红灯应该怎么做？')


def choose_unseen(train_counts, test_counts, count, seed):
    if not isinstance(train_counts, dict) or not isinstance(test_counts, dict) or type(count) is not int or count < 2:
        raise ValueError('Invalid selection inputs')
    candidates = sorted(k for k, v in test_counts.items() if type(k) is int and k >= 0
                        and type(v) is int and v > 0 and k not in train_counts)
    if len(candidates) < count:
        raise ValueError('Insufficient corpus-disjoint token types')
    random.Random(seed).shuffle(candidates)
    return candidates[:count], len(candidates)


def grouped_vectors(model, codes, device):
    """Write up to 32 distinct values concurrently to explicit orthogonal addresses."""
    if codes.ndim != 2 or not 1 <= len(codes) <= model.config.key_count:
        raise ValueError('Invalid group')
    fast, velocity = model.initial_state(1)
    for state in fast:
        state[0] = torch.zeros_like(state[0])
    with torch.no_grad():
        for key, value in enumerate(codes):
            for site_index, site in enumerate(model.memories):
                fast[site_index], velocity[site_index] = site.write_temporal(
                    fast[site_index], velocity[site_index], site.current_basis[key:key+1].to(device),
                    value.reshape(1, -1).to(device), torch.ones(1, device=device),
                    torch.zeros(1, dtype=torch.long, device=device), False)
        return torch.stack([torch.stack([
            site.retrieve([fast[site_index][0]], site.current_basis[key:key+1].to(device))[0]
            for site_index, site in enumerate(model.memories)]).mean(0)
            for key in range(len(codes))]).detach()


def metrics(predictions, targets):
    if len(predictions) != len(targets) or len(targets) % 2:
        raise ValueError('Paired targets require equal even lengths')
    return {'total': len(targets), 'exact': sum(p == t for p, t in zip(predictions, targets)),
            'pair_total': len(targets)//2,
            'pair_both_correct': sum(predictions[i] == targets[i] and predictions[i+1] == targets[i+1]
                                     for i in range(0, len(targets), 2)),
            'pair_outputs_differ': sum(predictions[i] != predictions[i+1] for i in range(0, len(targets), 2)),
            'predicted_ids': predictions}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    if prior['chosen_scale'] != 1000.0:
        raise ValueError('Unexpected previously fixed scale')
    files = {'base': runtime.MODEL / 'model.safetensors', 'integrated': runtime.CHECKPOINT,
             'train_corpus': CORPUS, 'test_corpus': TEST_CORPUS, 'prior': PRIOR}
    hashes = {name: runtime.sha(path) for name, path in files.items()}
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if (hashes['base'] != runtime.WEIGHT_SHA or hashes['train_corpus'] != manifest['files'][CORPUS.name]
            or hashes['test_corpus'] != manifest['files'][TEST_CORPUS.name]):
        raise ValueError('Source mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='spora-output-holdout-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        device = next(chat.organism.parameters()).device
        digest = frozen_digest(chat.organism, chat.state)
        train_counts = chinese_token_counts(chat.tokenizer, CORPUS)
        test_counts = chinese_token_counts(chat.tokenizer, TEST_CORPUS)
        selected, eligible = choose_unseen(train_counts, test_counts, COUNT, SEED)
        projection = make_projection(chat.base.lm_head.weight.shape[1], chat.organism.config.memory_width,
                                     prior['projection_seed']).to(device)
        projection_sha = hashlib.sha256(projection.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        if projection_sha != prior['projection_sha256']:
            raise ValueError('Projection differs from calibration run')
        head = chat.base.lm_head.eval()
        idx = torch.tensor(selected, dtype=torch.long, device=device)
        codes = encode_output_rows(head.weight.detach().index_select(0, idx), projection)
        isolated = torch.stack([actual_vector(chat.organism, code, device) for code in codes])
        coexisting = torch.cat([grouped_vectors(chat.organism, codes[i:i+GROUP], device)
                                for i in range(0, COUNT, GROUP)])
        isolated_cos = F.cosine_similarity(isolated, codes, dim=-1)
        coexist_cos = F.cosine_similarity(coexisting, codes, dim=-1)
        prompt = closed_think_inputs(chat.tokenizer, PROMPT, 'cuda')
        with torch.inference_mode():
            h0 = chat.base.model(**prompt, use_cache=False, return_dict=True).last_hidden_state[:, -1].float()
        scale = prior['chosen_scale']

        def predict(vectors):
            out = []
            with torch.no_grad():
                for i in range(0, len(vectors), 16):
                    delta = F.normalize(vectors[i:i+16].float(), dim=-1) @ projection.T
                    out.extend(head((h0 + scale*delta).to(head.weight.dtype)).argmax(-1).cpu().tolist())
            return out

        direct = predict(codes)
        singles = predict(isolated)
        group = predict(coexisting)
        zero = predict(torch.zeros_like(codes))
        with torch.no_grad():
            residual_norm = (scale*(F.normalize(coexisting.float(), dim=-1) @ projection.T)).norm(dim=-1)
            ratio = residual_norm / h0.norm(dim=-1)
        ordinary = []
        for question in ORDINARY:
            encoded = closed_think_inputs(chat.tokenizer, question, 'cuda')
            with torch.inference_mode():
                hidden = chat.base.model(**encoded, use_cache=False, return_dict=True).last_hidden_state[:, -1].float()
                normal = int(head(hidden.to(head.weight.dtype)).argmax(-1)[0])
                forced = int(head((hidden + scale*(F.normalize(coexisting[:1].float(), dim=-1) @ projection.T))
                                  .to(head.weight.dtype)).argmax(-1)[0])
            ordinary.append({'question': question, 'first_token_changed_when_forced': normal != forced})
        if frozen_digest(chat.organism, chat.state) != digest:
            raise ValueError('Original individual changed')
    if any(runtime.sha(path) != hashes[name] for name, path in files.items()):
        raise ValueError('Protected input changed')
    result = {'status': 'output_aligned_corpus_disjoint_holdout_not_deployed',
              'eligible_unseen_types': eligible, 'test_types': COUNT, 'selection_seed': SEED,
              'selected_types_sha256': hashlib.sha256(json.dumps(selected, separators=(',', ':')).encode('ascii')).hexdigest(),
              'projection_sha256': projection_sha, 'fixed_scale': scale, 'source_sha256': hashes,
              'direct': metrics(direct, selected), 'isolated_native': metrics(singles, selected),
              'coexisting_native': metrics(group, selected), 'zero_memory': metrics(zero, selected),
              'minimum_cosine_isolated': float(isolated_cos.min()), 'minimum_cosine_coexisting': float(coexist_cos.min()),
              'residual_to_base_norm': {'min': float(ratio.min()), 'mean': float(ratio.mean()), 'max': float(ratio.max())},
              'ordinary_forced': ordinary, 'seconds': time.perf_counter()-started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Test token types absent from the inspected Wikipedia train shard; base pretraining unknown.',
                              'Projection and scale fixed from earlier development; only one projection seed.',
                              'Matched filter uses output-head row of the token at write time, not learned semantic recall.',
                              'One token per 90-dim vector, explicit addresses and oracle activation.',
                              'Coexistence covers four separate groups of 32, not 128 simultaneous records.',
                              'First-token only; no language QA, identity, autonomous routing, history or production writes.',
                              'Ordinary prompts intentionally force activation and compare only first token.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'direct': result['direct']['exact'], 'single': result['isolated_native']['exact'],
                      'coexist': result['coexisting_native']['exact'],
                      'pairs': result['coexisting_native']['pair_both_correct'],
                      'ordinary_changed': sum(r['first_token_changed_when_forced'] for r in ordinary)}), flush=True)


if __name__ == '__main__':
    main()
