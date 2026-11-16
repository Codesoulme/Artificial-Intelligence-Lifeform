"""Compare one versus two native sites for output-aligned token vectors.

This remains a fixed matched-filter probe with oracle addresses and activation;
it is not a learned semantic or conversation system.
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
from scripts.deepseek_chunk_first_token_bridge_20261117 import CORPUS, MANIFEST, PROMPT, chinese_token_counts
from scripts.deepseek_output_aligned_holdout_20261117 import (
    TEST_CORPUS, PRIOR, OUT as PREVIOUS_HOLDOUT, COUNT, GROUP, ORDINARY, choose_unseen, grouped_vectors, metrics)
from scripts.deepseek_output_aligned_memory_pilot_20261117 import encode_output_rows, make_projection
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.language_training import frozen_digest

OUT = ROOT / 'runs/deepseek-spora-dual-site-aligned-pilot-20261117.json'
SECOND_SEED = 28018


def dual_group_vectors(model, codes, device):
    """Write one independent 90-D value per native site at each explicit key."""
    if (codes.ndim != 3 or codes.shape[1] != len(model.memories) or
            codes.shape[2] != model.config.memory_width or not 1 <= len(codes) <= model.config.key_count):
        raise ValueError('Invalid per-site memory group')
    fast, velocity = model.initial_state(1)
    for state in fast:
        state[0] = torch.zeros_like(state[0])
    with torch.no_grad():
        for key, per_site in enumerate(codes):
            for i, site in enumerate(model.memories):
                fast[i], velocity[i] = site.write_temporal(
                    fast[i], velocity[i], site.current_basis[key:key+1].to(device),
                    per_site[i].reshape(1, -1).to(device), torch.ones(1, device=device),
                    torch.zeros(1, dtype=torch.long, device=device), False)
        return torch.stack([torch.stack([
            site.retrieve([fast[i][0]], site.current_basis[key:key+1].to(device))[0]
            for i, site in enumerate(model.memories)]) for key in range(len(codes))]).detach()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    previous = json.loads(PREVIOUS_HOLDOUT.read_text(encoding='utf-8'))
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    if previous['fixed_scale'] != 1000.0 or prior['chosen_scale'] != 1000.0:
        raise ValueError('Unexpected fixed calibration')
    paths = {'base': runtime.MODEL / 'model.safetensors', 'integrated': runtime.CHECKPOINT,
             'train_corpus': CORPUS, 'test_corpus': TEST_CORPUS,
             'prior': PRIOR, 'previous_holdout': PREVIOUS_HOLDOUT}
    hashes = {name: runtime.sha(path) for name, path in paths.items()}
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if (hashes['base'] != runtime.WEIGHT_SHA or hashes['train_corpus'] != manifest['files'][CORPUS.name]
            or hashes['test_corpus'] != manifest['files'][TEST_CORPUS.name]):
        raise ValueError('Source mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='spora-dual-site-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        device = next(chat.organism.parameters()).device
        digest = frozen_digest(chat.organism, chat.state)
        if len(chat.organism.memories) != 2:
            raise ValueError('Expected two memory sites')
        train_counts = chinese_token_counts(chat.tokenizer, CORPUS)
        test_counts = chinese_token_counts(chat.tokenizer, TEST_CORPUS)
        all_types, eligible = choose_unseen(train_counts, test_counts, COUNT*2, previous['selection_seed'])
        former_sha = hashlib.sha256(json.dumps(all_types[:COUNT], separators=(',', ':')).encode('ascii')).hexdigest()
        if former_sha != previous['selected_types_sha256']:
            raise ValueError('Previous selection mismatch')
        targets = all_types[COUNT:COUNT*2]
        if set(targets) & set(all_types[:COUNT]):
            raise ValueError('Holdout types overlap')
        head = chat.base.lm_head.eval()
        projection0 = make_projection(head.weight.shape[1], chat.organism.config.memory_width,
                                      prior['projection_seed']).to(device)
        p0_sha = hashlib.sha256(projection0.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        if p0_sha != prior['projection_sha256']:
            raise ValueError('Prior projection mismatch')
        projection1 = make_projection(head.weight.shape[1], chat.organism.config.memory_width, SECOND_SEED).to(device)
        p1_sha = hashlib.sha256(projection1.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        rows = head.weight.detach().index_select(0, torch.tensor(targets, dtype=torch.long, device=device))
        code0, code1 = encode_output_rows(rows, projection0), encode_output_rows(rows, projection1)
        dual_codes = torch.stack((code0, code1), dim=1)
        single_vectors = torch.cat([grouped_vectors(chat.organism, code0[i:i+GROUP], device)
                                    for i in range(0, COUNT, GROUP)])
        dual_vectors = torch.cat([dual_group_vectors(chat.organism, dual_codes[i:i+GROUP], device)
                                  for i in range(0, COUNT, GROUP)])
        single_cos = F.cosine_similarity(single_vectors, code0, dim=-1)
        dual_cos = F.cosine_similarity(dual_vectors, dual_codes, dim=-1)
        single_vectors = F.normalize(single_vectors.float(), dim=-1)
        dual_vectors = F.normalize(dual_vectors.float(), dim=-1)
        prompt = closed_think_inputs(chat.tokenizer, PROMPT, 'cuda')
        with torch.inference_mode():
            h0 = chat.base.model(**prompt, use_cache=False, return_dict=True).last_hidden_state[:, -1].float()
        scale = prior['chosen_scale']

        def delta(single, dual=False):
            if dual:
                return (single[:, 0] @ projection0.T + single[:, 1] @ projection1.T) / math.sqrt(2)
            return single @ projection0.T

        def predict(vectors, *, dual=False):
            output = []
            with torch.no_grad():
                for i in range(0, len(vectors), 16):
                    hidden = h0 + scale*delta(vectors[i:i+16], dual=dual)
                    output.extend(head(hidden.to(head.weight.dtype)).argmax(-1).cpu().tolist())
            return output

        single = predict(single_vectors)
        dual = predict(dual_vectors, dual=True)
        direct_dual = predict(dual_codes, dual=True)
        zero = predict(torch.zeros_like(single_vectors))
        ordinary = []
        for question in ORDINARY:
            encoded = closed_think_inputs(chat.tokenizer, question, 'cuda')
            with torch.inference_mode():
                hidden = chat.base.model(**encoded, use_cache=False, return_dict=True).last_hidden_state[:, -1].float()
                normal = int(head(hidden.to(head.weight.dtype)).argmax(-1)[0])
                forced_one = int(head((hidden+scale*delta(single_vectors[:1])).to(head.weight.dtype)).argmax(-1)[0])
                forced_two = int(head((hidden+scale*delta(dual_vectors[:1], dual=True)).to(head.weight.dtype)).argmax(-1)[0])
            ordinary.append({'question': question, 'single_changed': normal != forced_one,
                             'dual_changed': normal != forced_two})
        if frozen_digest(chat.organism, chat.state) != digest:
            raise ValueError('Original individual changed')
    if any(runtime.sha(path) != hashes[name] for name, path in paths.items()):
        raise ValueError('Protected inputs changed')
    result = {'status': 'dual_site_fixed_projection_research_not_deployed',
              'test_types': COUNT, 'eligible_unseen_types': eligible,
              'new_types_sha256': hashlib.sha256(json.dumps(targets, separators=(',', ':')).encode('ascii')).hexdigest(),
              'first_projection_sha256': p0_sha, 'second_projection_sha256': p1_sha,
              'second_projection_seed': SECOND_SEED, 'fixed_scale': scale,
              'single_native': metrics(single, targets), 'dual_native': metrics(dual, targets),
              'dual_direct': metrics(direct_dual, targets), 'zero_memory': metrics(zero, targets),
              'minimum_single_cosine': float(single_cos.min()), 'minimum_dual_site_cosine': float(dual_cos.min()),
              'ordinary_forced': ordinary, 'source_sha256': hashes,
              'seconds': time.perf_counter()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Output-head matched filter, not learned semantic reading.',
                              'Second set of 128 types absent from inspected train corpus and disjoint from prior 128.',
                              'Four independent groups of 32, not 128 concurrent entities.',
                              'Two sites store different vectors at one manually specified address.',
                              'Fixed prompt, first token, fixed scale; no natural QA, provenance or history.',
                              'Forced ordinary-prompt injection tests only first-token changes, not answer quality.',
                              'Dual representation consumes both sites and doubles value-vector capacity per token.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'single': result['single_native']['exact'], 'dual': result['dual_native']['exact'],
                      'dual_pairs': result['dual_native']['pair_both_correct'],
                      'ordinary_single': sum(r['single_changed'] for r in ordinary),
                      'ordinary_dual': sum(r['dual_changed'] for r in ordinary)}), flush=True)


if __name__ == '__main__':
    main()
